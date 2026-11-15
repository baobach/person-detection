# Evaluate person detectors on the COCO val2017 person subset.
#
# Compares the fine-tuned single-class model with the stock COCO-pretrained YOLOv8s
# (restricted to its "person" class), and breaks the official COCO AP down by
# person size and by how crowded the image is.
#
# Usage (from repo root, after src/prepare_coco.py has made the val split):
#   uv run python src/evaluate.py
#   uv run python src/evaluate.py --device 0 --out report/eval
#
# --device is a PyTorch CUDA index. PyTorch orders GPUs fastest first, which can differ
# from nvidia-smi; set CUDA_DEVICE_ORDER=PCI_BUS_ID to make the two agree.
#
# Outputs in --out:
#   results.json   every number below, per model
#   results.md     the same tables in Markdown, ready for the report
#
# Two metric families are reported:
#   - Ultralytics val (precision, recall, mAP@0.5, mAP@0.5:0.95). These match the
#     training log and the numbers in the README.
#   - pycocotools COCOeval (the official COCO protocol) for AP by size and by crowding.
#     Its AP@0.5:0.95 is about 1 point higher than Ultralytics' because the two use
#     different precision-recall interpolation.

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSON_CAT_ID = 1  # COCO category id for "person"

# Crowding bins: number of (non-crowd) ground-truth people in the image.
CROWD_BINS = [("1 person", 1, 1), ("2-5 people", 2, 5), ("6+ people", 6, 10**9)]
COCO_STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                   "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def ultralytics_val(model, data_dir: Path, work_dir: Path, name: str, person_class: int | None,
                    device: str, imgsz: int, batch: int, workers: int) -> dict:
    """Run Ultralytics val and return precision, recall, mAP50, mAP50-95 for the person class.

    person_class is None for the single-class model. For the 80-class pretrained model it is 0:
    the data yaml then lists all 80 COCO names (labels only use class 0) and predictions are
    filtered to that class, so both models are scored on the same person boxes.
    """
    yaml_path = work_dir / f"data_{name}.yaml"
    names = {0: "person"} if person_class is None else model.names
    lines = [f"path: {data_dir.resolve()}", "train: images/val2017", "val: images/val2017", "names:"]
    lines += [f"  {k}: {v}" for k, v in names.items()]
    yaml_path.write_text("\n".join(lines) + "\n")

    r = model.val(data=str(yaml_path), split="val", imgsz=imgsz, batch=batch, device=device, workers=workers,
                  classes=None if person_class is None else [person_class],
                  project=str(work_dir / "val_runs"), name=name, exist_ok=True,
                  plots=False, verbose=False)
    if person_class is None:
        p, rec, m50, m5095 = r.box.mean_results()
    else:
        idx = list(r.box.ap_class_index).index(person_class)
        p, rec, m50, m5095 = r.box.class_result(idx)
    return {"precision": float(p), "recall": float(rec), "mAP50": float(m50), "mAP50-95": float(m5095)}


def predict_coco(model, images: list[Path], person_class: int | None, device: str, imgsz: int,
                 batch: int) -> list[dict]:
    """Predict every image and return detections in COCO results format (person only).

    Uses the same low confidence threshold as Ultralytics val (0.001), since AP needs
    the whole precision-recall curve. File stems are the COCO image ids.
    """
    dets = []
    classes = None if person_class is None else [person_class]
    for i in range(0, len(images), batch):
        chunk = [str(p) for p in images[i:i + batch]]
        for r in model.predict(chunk, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                               classes=classes, device=device, verbose=False):
            image_id = int(Path(r.path).stem)
            for (x1, y1, x2, y2), score in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                dets.append({"image_id": image_id, "category_id": PERSON_CAT_ID,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    return dets


def coco_eval(gt, dets: list[dict], img_ids: list[int]) -> dict:
    """Official COCO bbox evaluation for the person category on img_ids. Returns the 12 stats."""
    with contextlib.redirect_stdout(io.StringIO()):  # COCOeval prints a lot
        dt = gt.loadRes(dets) if dets else gt.loadRes([{"image_id": img_ids[0], "category_id": 1,
                                                        "bbox": [0, 0, 1, 1], "score": 0.0}])
        from pycocotools.cocoeval import COCOeval
        ev = COCOeval(gt, dt, "bbox")
        ev.params.catIds = [PERSON_CAT_ID]
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return dict(zip(COCO_STAT_NAMES, (float(x) for x in ev.stats)))


def dataset_stats(gt, img_ids: list[int]) -> dict:
    """Counts of ground-truth people by COCO size bucket and images by crowding bin."""
    anns = gt.loadAnns(gt.getAnnIds(imgIds=img_ids, catIds=[PERSON_CAT_ID], iscrowd=False))
    areas = [a["area"] for a in anns]
    per_image = {i: 0 for i in img_ids}
    for a in anns:
        per_image[a["image_id"]] += 1
    return {
        "images": len(img_ids),
        "people": len(anns),
        "small (<32^2 px)": sum(a < 32**2 for a in areas),
        "medium": sum(32**2 <= a < 96**2 for a in areas),
        "large (>=96^2 px)": sum(a >= 96**2 for a in areas),
        "per_image": per_image,
    }


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def to_markdown(results: dict, stats: dict) -> str:
    """Format the results as Markdown tables for the report."""
    models = list(results)
    out = ["# Evaluation on COCO val2017, person class", "",
           f"{stats['images']:,} images, {stats['people']:,} people "
           f"(small {stats['small (<32^2 px)']:,}, medium {stats['medium']:,}, "
           f"large {stats['large (>=96^2 px)']:,}).", "",
           "## Overall (Ultralytics val)", "",
           "| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |", "| --- | --- | --- | --- | --- |"]
    for m in models:
        u = results[m]["ultralytics"]
        out.append(f"| {m} | {pct(u['precision'])} | {pct(u['recall'])} | {pct(u['mAP50'])} | {pct(u['mAP50-95'])} |")

    out += ["", "## By person size (COCO AP, pycocotools)", "",
            "Small: area < 32x32 px. Medium: up to 96x96 px. Large: above.", "",
            "| Model | AP | AP50 | AP75 | AP small | AP medium | AP large | AR100 | AR small |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in models:
        c = results[m]["coco"]["all"]
        out.append(f"| {m} | {c['AP']:.3f} | {c['AP50']:.3f} | {c['AP75']:.3f} | {c['AP_small']:.3f} | "
                   f"{c['AP_medium']:.3f} | {c['AP_large']:.3f} | {c['AR100']:.3f} | {c['AR_small']:.3f} |")

    out += ["", "## By crowding (COCO AP, pycocotools)", "",
            "Images grouped by the number of ground-truth people.", "",
            "| Model | " + " | ".join(f"{b} ({results[models[0]]['coco'][b]['images']} img) AP"
                                     for b, _, _ in CROWD_BINS) + " |",
            "| --- |" + " --- |" * len(CROWD_BINS)]
    for m in models:
        out.append(f"| {m} | " + " | ".join(f"{results[m]['coco'][b]['AP']:.3f}" for b, _, _ in CROWD_BINS) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate person detectors on COCO val2017 persons.")
    ap.add_argument("--finetuned", type=Path, default=ROOT / "runs" / "yolov8s_coco_person" / "weights" / "best.pt")
    ap.add_argument("--pretrained", default="yolov8s.pt", help="stock COCO checkpoint used as the baseline")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "coco_person", help="YOLO dataset from prepare_coco.py")
    ap.add_argument("--ann", type=Path, default=ROOT / "data" / "coco_raw" / "annotations" / "instances_val2017.json")
    ap.add_argument("--out", type=Path, default=ROOT / "report" / "eval")
    ap.add_argument("--device", default="0", help="cuda device index, or cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers for Ultralytics val")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    images = sorted((args.data / "images" / "val2017").glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {args.data / 'images' / 'val2017'}; run src/prepare_coco.py --splits val")
    img_ids = [int(p.stem) for p in images]

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(args.ann))
    stats = dataset_stats(gt, img_ids)
    bins = {b: [i for i, n in stats["per_image"].items() if lo <= n <= hi] for b, lo, hi in CROWD_BINS}

    # (label, weights, person class index or None for the single-class model)
    models = [("Fine-tuned YOLOv8s (best.pt)", str(args.finetuned), None),
              ("Pretrained YOLOv8s, person class", args.pretrained, 0)]

    results = {}
    # Scratch space for the temporary data yamls and Ultralytics' val folders.
    work_dir = Path(tempfile.mkdtemp(prefix="person_eval_"))
    for label, weights, person_class in models:
        print(f"== {label}", flush=True)
        model = YOLO(weights)
        tag = "finetuned" if person_class is None else "pretrained"
        # Store the checkpoint path relative to the repo, so results.json has no local paths.
        shown = Path(weights)
        if shown.is_absolute() and shown.is_relative_to(ROOT):
            shown = shown.relative_to(ROOT)
        res = {"weights": str(shown),
               "ultralytics": ultralytics_val(model, args.data, work_dir, tag, person_class,
                                              args.device, args.imgsz, args.batch, args.workers)}
        dets = predict_coco(model, images, person_class, args.device, args.imgsz, args.batch)
        res["coco"] = {"all": coco_eval(gt, dets, img_ids)}
        for b, ids in bins.items():
            res["coco"][b] = {"images": len(ids), **coco_eval(gt, dets, ids)}
        results[label] = res
        u = res["ultralytics"]
        print(f"   P {pct(u['precision'])}  R {pct(u['recall'])}  mAP50 {pct(u['mAP50'])}  "
              f"mAP50-95 {pct(u['mAP50-95'])}  COCO AP {res['coco']['all']['AP']:.3f}", flush=True)

    stats_out = {k: v for k, v in stats.items() if k != "per_image"}
    (args.out / "results.json").write_text(json.dumps({"dataset": stats_out, "models": results}, indent=2))
    md = to_markdown(results, stats_out)
    (args.out / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
