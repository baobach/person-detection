# Download and prepare the person-only dataset from COCO 2017 dataset. It will  prepare the train/test split with YOLO label ready for training.
#   In:  COCO annotation JSON + COCO images
#   Out: data/coco_person/{images,labels}/{train2017,val2017}/  (.jpg + matching .txt)
#
# Usage (from repo root):
#   uv run python src/prepare_coco.py                      # full person subset (~64k train, ~2.7k val images)
#   uv run python src/prepare_coco.py --max-images 200     # small smoke-test subset per split
#
# Notes:
#   - Only the annotation zip (~240MB) is downloaded in bulk. Images are fetched one by one
#     from their coco_url, so only person images are downloaded (not the full 18GB train zip).
#   - Reruns resume: existing images are skipped, partial downloads are written to .part first.
#   - COCO test2017 has no public labels, so val2017 serves as the evaluation split.
#   - Crowd boxes (iscrowd=1) are dropped, as in the Ultralytics COCO converter. Images whose
#     only person annotations are crowd boxes are skipped (they would act as false negatives).

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
PERSON_CAT_ID = 1  # COCO category id for "person"
YOLO_CLASS = 0  # single-class YOLO index


def download(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> None:
    """Download url to dest atomically (via .part file), with retries."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(dest)
            return
        except Exception as e:  # network errors, timeouts
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {e}") from e
            time.sleep(2 * attempt)


def get_annotations(raw_dir: Path) -> Path:
    """Make sure instances_{train,val}2017.json exist under raw_dir/annotations."""
    ann_dir = raw_dir / "annotations"
    needed = [ann_dir / f"instances_{s}2017.json" for s in ("train", "val")]
    if all(p.exists() for p in needed):
        return ann_dir
    zip_path = raw_dir / "annotations_trainval2017.zip"
    print(f"Downloading annotations (~240MB) to {zip_path} ...")
    download(ANN_URL, zip_path, timeout=300)
    print("Extracting instances_*.json ...")
    with zipfile.ZipFile(zip_path) as z:
        for p in needed:
            z.extract(f"annotations/{p.name}", raw_dir)
    return ann_dir


def coco_to_yolo(bbox, img_w: int, img_h: int):
    """COCO [x_min, y_min, w, h] in pixels -> YOLO (cx, cy, w, h) normalized, clipped to image."""
    x, y, w, h = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return ((x1 + w / 2) / img_w, (y1 + h / 2) / img_h, w / img_w, h / img_h)


def build_split(ann_file: Path):
    """Return {image_id: (image_info, [yolo boxes])} for images with >=1 non-crowd person box."""
    print(f"Loading {ann_file.name} ...")
    with open(ann_file) as f:
        data = json.load(f)
    images = {im["id"]: im for im in data["images"]}
    boxes = defaultdict(list)
    for a in data["annotations"]:
        if a["category_id"] != PERSON_CAT_ID or a.get("iscrowd", 0):
            continue
        im = images[a["image_id"]]
        b = coco_to_yolo(a["bbox"], im["width"], im["height"])
        if b is not None:
            boxes[a["image_id"]].append(b)
    return {i: (images[i], bs) for i, bs in boxes.items()}


def write_label(path: Path, boxes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{YOLO_CLASS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes]
    path.write_text("\n".join(lines) + "\n")


def prepare_split(split: str, ann_dir: Path, out_dir: Path, max_images: int | None, workers: int, seed: int) -> None:
    subset = build_split(ann_dir / f"instances_{split}2017.json")
    ids = sorted(subset)
    if max_images is not None and max_images < len(ids):
        random.Random(seed).shuffle(ids)
        ids = sorted(ids[:max_images])
    n_boxes = sum(len(subset[i][1]) for i in ids)
    print(f"[{split}2017] {len(ids)} person images, {n_boxes} person boxes")

    img_dir = out_dir / "images" / f"{split}2017"
    lbl_dir = out_dir / "labels" / f"{split}2017"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Labels are cheap: write all of them first.
    for i in ids:
        info, bs = subset[i]
        write_label(lbl_dir / (Path(info["file_name"]).stem + ".txt"), bs)

    # Images: parallel download, skipping ones already on disk.
    todo = [subset[i][0] for i in ids if not (img_dir / subset[i][0]["file_name"]).exists()]
    print(f"[{split}2017] downloading {len(todo)} images ({len(ids) - len(todo)} already present)")
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download, im["coco_url"], img_dir / im["file_name"]): im for im in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:
                failed.append(futs[fut]["file_name"])
                print(f"  ! {e}")
            if k % 500 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}")

    # Drop labels whose image failed, so every label has an image.
    for name in failed:
        (lbl_dir / (Path(name).stem + ".txt")).unlink(missing_ok=True)
    if failed:
        print(f"[{split}2017] {len(failed)} images failed; rerun the script to retry them")


def write_data_yaml(cfg: Path, out_dir: Path) -> None:
    """Write the Ultralytics data yaml with a repo-relative dataset path (shared across contributors).

    Ultralytics resolves a relative `path` against the current working directory,
    so train/eval must be run from the repo root.
    """
    out_dir = out_dir.resolve()
    try:
        path = out_dir.relative_to(ROOT).as_posix()
    except ValueError:  # custom --out-dir outside the repo: no relative form exists
        path = out_dir.as_posix()
        print(f"Warning: {out_dir} is outside the repo; writing an absolute path to {cfg}")
    cfg.write_text(
        "# YOLO dataset config for the COCO person subset (written by src/prepare_coco.py)\n"
        "# `path` is relative to the repo root: run training/eval from there.\n"
        f"path: {path}\n"
        "train: images/train2017\n"
        "val: images/val2017\n"
        "names:\n"
        "  0: person\n"
    )
    print(f"Wrote {cfg}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare the COCO 2017 person subset in YOLO format.")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "coco_raw", help="where COCO annotations go")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "coco_person", help="YOLO dataset output")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--max-images", type=int, default=None, help="cap images per split (random, seeded)")
    ap.add_argument("--workers", type=int, default=16, help="parallel image downloads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "coco_person.yaml", help="data yaml to write")
    args = ap.parse_args()

    ann_dir = get_annotations(args.raw_dir)
    for split in args.splits:
        prepare_split(split, ann_dir, args.out_dir, args.max_images, args.workers, args.seed)
    write_data_yaml(args.config, args.out_dir)


if __name__ == "__main__":
    main()
