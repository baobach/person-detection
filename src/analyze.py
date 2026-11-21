# Error analysis and a test-time scale / augmentation experiment for the fine-tuned detector.
#
# Covers the conditions named in the project proposal:
#   - person size        (COCO small / medium / large)
#   - crowded scenes     (number of people in the image)
#   - partial occlusion  (share of a person's labeled keypoints that COCO marks as not visible)
#   - lighting           (mean image brightness: dark / normal / bright)
# and breaks the detections at the deployed threshold (confidence 0.25) into error types.
#
# The experiment re-runs the same weights at larger input sizes (960, 1280), with
# Ultralytics test-time augmentation (TTA: flips and rescales), and with the 640 and 960
# predictions merged by NMS, to test whether more resolution helps the small people,
# which are the weakest case.
#
# Usage (from repo root, after src/prepare_coco.py has made the val split):
#   uv run python src/analyze.py --device 0
#   uv run python src/analyze.py --device 0 --skip-experiment   # analysis only
#
# Needs data/coco_raw/annotations/{instances,person_keypoints}_val2017.json
# (both are in the annotation zip that prepare_coco.py downloads).
#
# Outputs in --out (default report/analysis/):
#   results.json, results.md        all numbers
#   fig_breakdown.png               AP / recall by size, crowding, lighting, occlusion
#   fig_errors.png                  detection outcomes and missed people by size
#   fig_scale.png                   AP by input size and TTA
#   examples/*.jpg                  failure cases: ground truth vs predictions

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from evaluate import CROWD_BINS, PERSON_CAT_ID, coco_eval, predict_coco

ROOT = Path(__file__).resolve().parents[1]

CONF = 0.25          # deployed confidence threshold (same as the report figures)
IOU_MATCH = 0.5      # a detection counts as correct at IoU >= 0.5
IOU_LOC = 0.1        # 0.1 <= IoU < 0.5 with a person counts as a localization error
DARK, BRIGHT = 70, 160  # mean gray level (0-255) bounds for the lighting bins
SIZE_BINS = [("small", 0, 32**2), ("medium", 32**2, 96**2), ("large", 96**2, float("inf"))]
OCC_BINS = [("0% hidden", 0.0, 0.0), ("1-25% hidden", 1e-9, 0.25),
            ("26-50% hidden", 0.25 + 1e-9, 0.5), (">50% hidden", 0.5 + 1e-9, 1.0)]
LIGHT_BINS = ["dark", "normal", "bright"]
ERROR_TYPES = ["correct", "duplicate", "localization", "background", "crowd region (ignored)"]


# ---------------------------------------------------------------- ground truth

def load_ground_truth(ann_dir: Path, img_ids: list[int]):
    """Load COCO GT for the person class and attach per-person attributes.

    Returns (coco, people, crowds):
      people: {image_id: [dict(id, box xyxy, area, size, occ)]}  non-crowd persons
      crowds: {image_id: [box xyxy]}                                crowd regions
    occ is the share of labeled keypoints marked "labeled but not visible" (v=1),
    or None when COCO has no keypoints for that person (all small people, a few others).
    """
    from pycocotools.coco import COCO

    kp_file = ann_dir / "person_keypoints_val2017.json"
    if not kp_file.exists():
        zip_path = ann_dir.parent / "annotations_trainval2017.zip"
        with zipfile.ZipFile(zip_path) as z:
            z.extract(f"annotations/{kp_file.name}", ann_dir.parent)
    with open(kp_file) as f:
        keypoints = {a["id"]: a["keypoints"][2::3] for a in json.load(f)["annotations"]}

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ann_dir / "instances_val2017.json"))
    people, crowds = defaultdict(list), defaultdict(list)
    for a in coco.loadAnns(coco.getAnnIds(imgIds=img_ids, catIds=[PERSON_CAT_ID])):
        x, y, w, h = a["bbox"]
        box = [x, y, x + w, y + h]
        if a["iscrowd"]:
            crowds[a["image_id"]].append(box)
            continue
        vis = keypoints.get(a["id"], [])
        labeled = sum(v > 0 for v in vis)
        occ = sum(v == 1 for v in vis) / labeled if labeled else None
        size = next(name for name, lo, hi in SIZE_BINS if lo <= a["area"] < hi)
        people[a["image_id"]].append({"id": a["id"], "box": box, "area": a["area"], "size": size, "occ": occ})
    return coco, people, crowds


def occ_bin(occ: float | None) -> str | None:
    """Return the occlusion bin name for a share of hidden keypoints, or None if the person has no keypoints."""
    if occ is None:
        return None
    return next(name for name, lo, hi in OCC_BINS if lo <= occ <= hi)


def crowd_bin(n: int) -> str:
    """Return the crowding bin name (from CROWD_BINS) for an image with n people."""
    return next(name for name, lo, hi in CROWD_BINS if lo <= n <= hi)


def light_bin(brightness: float) -> str:
    """Return "dark", "normal", or "bright" from an image's mean gray level (thresholds DARK and BRIGHT)."""
    return "dark" if brightness < DARK else "bright" if brightness > BRIGHT else "normal"


# ---------------------------------------------------------------- matching

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def ioa(det: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Share of the detection's area inside each region (det (4,), regions (M,4))."""
    if len(regions) == 0:
        return np.zeros(0)
    x1 = np.maximum(det[0], regions[:, 0]); y1 = np.maximum(det[1], regions[:, 1])
    x2 = np.minimum(det[2], regions[:, 2]); y2 = np.minimum(det[3], regions[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((det[2] - det[0]) * (det[3] - det[1]) + 1e-9)


def match_image(dets: list[dict], people: list[dict], crowds: list) -> tuple[list[str], list[bool]]:
    """Greedy COCO-style matching at IoU 0.5 for one image, highest score first.

    Returns (outcome per detection, matched flag per person). Outcomes:
      correct       IoU >= 0.5 with a person not matched yet
      duplicate     IoU >= 0.5 only with a person already matched by a higher-scoring box
      crowd region  mostly inside a COCO crowd region (ignored, as in COCO evaluation)
      localization  best IoU with any person in [0.1, 0.5): right person, poor box
      background    no person nearby: a false alarm on the background or another object
    """
    dets = sorted(dets, key=lambda d: -d["score"])
    d_boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in (d["bbox"] for d in dets)]).reshape(-1, 4)
    g_boxes = np.array([p["box"] for p in people]).reshape(-1, 4)
    c_boxes = np.array(crowds).reshape(-1, 4)
    ious = iou_matrix(d_boxes, g_boxes)
    matched = [False] * len(people)
    outcomes = []
    for i in range(len(dets)):
        row = ious[i] if len(people) else np.zeros(0)
        free = [j for j in range(len(people)) if not matched[j] and row[j] >= IOU_MATCH]
        if free:
            j = max(free, key=lambda k: row[k])
            matched[j] = True
            outcomes.append("correct")
        elif len(row) and row.max() >= IOU_MATCH:
            outcomes.append("duplicate")
        elif len(c_boxes) and ioa(d_boxes[i], c_boxes).max() >= 0.5:
            outcomes.append("crowd region (ignored)")
        elif len(row) and row.max() >= IOU_LOC:
            outcomes.append("localization")
        else:
            outcomes.append("background")
    return outcomes, matched


# ---------------------------------------------------------------- analysis

def analyze(dets: list[dict], coco, people, crowds, img_ids, brightness) -> dict:
    """All breakdowns for one set of detections (fine-tuned model at 640)."""
    by_image = defaultdict(list)
    for d in dets:
        if d["score"] >= CONF:
            by_image[d["image_id"]].append(d)

    outcome_counts = Counter()
    recall = defaultdict(lambda: [0, 0])          # group -> [found, total]
    per_image = {}
    for iid in img_ids:
        outcomes, matched = match_image(by_image[iid], people[iid], crowds[iid])
        outcome_counts.update(outcomes)
        crowd = crowd_bin(len(people[iid]))
        light = light_bin(brightness[iid])
        for p, m in zip(people[iid], matched):
            groups = ["all", f"size:{p['size']}", f"crowd:{crowd}", f"light:{light}"]
            ob = occ_bin(p["occ"])
            if ob:
                groups.append(f"occ:{ob}")
                groups.append(f"occ:{ob}|{p['size']}")
            for g in groups:
                recall[g][0] += m
                recall[g][1] += 1
        per_image[iid] = {"outcomes": outcomes, "matched": matched,
                          "dets": sorted(by_image[iid], key=lambda d: -d["score"])}

    # COCO AP for image groups (lighting and crowding), all people in those images.
    light_ids = {b: [i for i in img_ids if light_bin(brightness[i]) == b] for b in LIGHT_BINS}
    crowd_ids = {b: [i for i in img_ids if crowd_bin(len(people[i])) == b] for b, _, _ in CROWD_BINS}
    ap = {"all": coco_eval(coco, dets, img_ids)}
    for b, ids in {**{f"light:{k}": v for k, v in light_ids.items()},
                   **{f"crowd:{k}": v for k, v in crowd_ids.items()}}.items():
        ap[b] = {"images": len(ids), **coco_eval(coco, dets, ids)}

    n_det = sum(outcome_counts.values())
    missed_by_size = {s: recall[f"size:{s}"][1] - recall[f"size:{s}"][0] for s, _, _ in SIZE_BINS}
    return {
        "conf": CONF,
        "detections": n_det,
        "outcomes": {k: outcome_counts.get(k, 0) for k in ERROR_TYPES},
        "recall": {g: {"found": f, "total": t, "recall": f / t} for g, (f, t) in sorted(recall.items())},
        "missed_by_size": missed_by_size,
        "ap": ap,
        "_per_image": per_image,  # not saved; used to pick examples
    }


# ---------------------------------------------------------------- experiment

def merge_scales(dets_a: list[dict], dets_b: list[dict], iou: float = 0.7, max_det: int = 300) -> list[dict]:
    """Merge two prediction sets per image with NMS (multi-scale inference).

    iou=0.7 is the model's own NMS threshold (0.5 and 0.6 gave AP 0.614 and 0.620 on val2017).
    """
    import torch
    import torchvision

    by_image = defaultdict(list)
    for d in dets_a + dets_b:
        by_image[d["image_id"]].append(d)
    merged = []
    for ds in by_image.values():
        boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in (d["bbox"] for d in ds)], dtype=torch.float32)
        scores = torch.tensor([d["score"] for d in ds], dtype=torch.float32)
        keep = torchvision.ops.nms(boxes, scores, iou)[:max_det]
        merged += [ds[k] for k in keep.tolist()]
    return merged


def score_setting(label: str, dets: list[dict], ms: float, coco, img_ids, people, crowds) -> dict:
    """COCO AP plus recall and false boxes at the deployed threshold for one inference setting."""
    stats = coco_eval(coco, dets, img_ids)
    by_image = defaultdict(list)
    for d in dets:
        if d["score"] >= CONF:
            by_image[d["image_id"]].append(d)
    found = total = found_small = total_small = fp = 0
    for iid in img_ids:
        outcomes, matched = match_image(by_image[iid], people[iid], crowds[iid])
        fp += sum(o in ("duplicate", "localization", "background") for o in outcomes)
        found += sum(matched)
        total += len(matched)
        for p, m in zip(people[iid], matched):
            if p["size"] == "small":
                found_small += m
                total_small += 1
    return {"setting": label, "ms_per_image": ms,
            "AP": stats["AP"], "AP50": stats["AP50"], "AP_small": stats["AP_small"],
            "AP_medium": stats["AP_medium"], "AP_large": stats["AP_large"],
            "recall@0.25": found / total, "recall_small@0.25": found_small / total_small,
            "false_positives@0.25": fp}


def scale_experiment(model, images, coco, img_ids, people, crowds, device) -> list[dict]:
    """Same weights at several input sizes, with TTA, and with a 640 + 960 merge. One row per setting."""
    settings = [("640", 640, False, 32), ("960", 960, False, 16), ("1280", 1280, False, 8),
                ("640 + TTA", 640, True, 16)]
    rows, kept = [], {}
    for label, imgsz, augment, batch in settings:
        print(f"   experiment: {label}", flush=True)
        t0 = time.time()
        dets = predict_coco(model, images, None, device, imgsz, batch, augment=augment)
        ms = 1000 * (time.time() - t0) / len(images)
        kept[label] = (dets, ms)
        rows.append(score_setting(label, dets, ms, coco, img_ids, people, crowds))

    # Multi-scale: 640 keeps the large people, 960 adds the small ones. Time = both passes.
    print("   experiment: 640 + 960 merged", flush=True)
    merged = merge_scales(kept["640"][0], kept["960"][0])
    rows.append(score_setting("640 + 960 merged", merged, kept["640"][1] + kept["960"][1],
                              coco, img_ids, people, crowds))
    return rows


# ---------------------------------------------------------------- figures

def plot_figures(res: dict, exp: list[dict] | None, out: Path) -> None:
    """Draw the report figures: breakdown by condition, detection outcomes, and (if run) the input-size experiment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap, rec = res["ap"], res["recall"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    panels = [
        ("Recall by person size", [s for s, _, _ in SIZE_BINS], [rec[f"size:{s}"]["recall"] for s, _, _ in SIZE_BINS],
         [rec[f"size:{s}"]["total"] for s, _, _ in SIZE_BINS], "recall @ conf 0.25"),
        ("AP by crowding", [b for b, _, _ in CROWD_BINS], [ap[f"crowd:{b}"]["AP"] for b, _, _ in CROWD_BINS],
         [ap[f"crowd:{b}"]["images"] for b, _, _ in CROWD_BINS], "COCO AP"),
        ("AP by lighting", LIGHT_BINS, [ap[f"light:{b}"]["AP"] for b in LIGHT_BINS],
         [ap[f"light:{b}"]["images"] for b in LIGHT_BINS], "COCO AP"),
        ("Recall by occlusion\n(medium + large people)", [b for b, _, _ in OCC_BINS],
         [rec[f"occ:{b}"]["recall"] for b, _, _ in OCC_BINS], [rec[f"occ:{b}"]["total"] for b, _, _ in OCC_BINS],
         "recall @ conf 0.25"),
    ]
    for ax, (title, labels, vals, counts, ylabel) in zip(axes, panels):
        bars = ax.bar(range(len(vals)), vals, color="#4C72B0")
        ax.set_xticks(range(len(vals)), labels, rotation=15, fontsize=8)
        ax.set_ylim(0, 1.15); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=9)
        for b, v, n in zip(bars, vals, counts):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}\nn={n}", ha="center", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_breakdown.png", dpi=150); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = ERROR_TYPES
    vals = [res["outcomes"][k] for k in labels]
    colors = ["#55A868", "#C44E52", "#DD8452", "#8172B3", "#999999"]
    bars = ax1.barh(range(len(vals)), vals, color=colors)
    ax1.set_yticks(range(len(vals)), labels); ax1.invert_yaxis()
    ax1.set_title(f"Detections at conf {CONF} ({res['detections']:,} boxes)", fontsize=10)
    for b, v in zip(bars, vals):
        ax1.text(v, b.get_y() + b.get_height() / 2, f" {v:,}", va="center", fontsize=8)
    ax1.set_xlim(0, max(vals) * 1.15)
    sizes = [s for s, _, _ in SIZE_BINS]
    miss = [res["missed_by_size"][s] for s in sizes]
    bars = ax2.bar(sizes, miss, color="#C44E52")
    for b, v, s in zip(bars, miss, sizes):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / sum(miss):.0%})", ha="center", va="bottom", fontsize=8)
    ax2.set_ylim(0, max(miss) * 1.25)
    ax2.set_title(f"Missed people by size ({sum(miss):,} of {res['recall']['all']['total']:,})", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig_errors.png", dpi=150); plt.close(fig)

    if exp:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        x = np.arange(len(exp)); w = 0.2
        for k, (key, name) in enumerate([("AP", "AP (all)"), ("AP_small", "AP small"),
                                         ("AP_medium", "AP medium"), ("AP_large", "AP large")]):
            ax.bar(x + (k - 1.5) * w, [r[key] for r in exp], w, label=name)
        ax.set_xticks(x, [r["setting"] for r in exp]); ax.set_ylim(0, 1)
        ax.set_xlabel("inference setting (same fine-tuned weights)"); ax.set_ylabel("COCO AP")
        ax.set_title("Input size and test-time augmentation", fontsize=10)
        ax.legend(fontsize=8, ncol=4, loc="upper left"); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
        fig.tight_layout(); fig.savefig(out / "fig_scale.png", dpi=150); plt.close(fig)


def draw_example(path: Path, info: dict, people: list[dict], out_file: Path) -> None:
    """Ground truth: green (found) or yellow (missed). Predictions: red for errors only.

    Each false box gets a label on a filled red tag: dup (duplicate), loc (localization),
    bac (background), and the confidence. Captions in the report describe each image.
    """
    img = cv2.imread(str(path))
    for p, m in zip(people, info["matched"]):
        x1, y1, x2, y2 = map(int, p["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0) if m else (0, 220, 255), 2)
    for d, o in zip(info["dets"], info["outcomes"]):
        if o in ("duplicate", "localization", "background"):
            x, y, w, h = map(int, d["bbox"])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 230), 2)
            label = f"{o[:3]} {d['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = y - 4 if y - th - 6 > 0 else y + th + 4  # put the tag inside the box at the top edge
            cv2.rectangle(img, (x, ty - th - 3), (x + tw + 4, ty + 3), (0, 0, 230), -1)
            cv2.putText(img, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def save_examples(res: dict, people, brightness, img_dir: Path, out: Path) -> list[dict]:
    """Pick one clear failure per condition and draw it."""
    per = res["_per_image"]

    chosen: set[int] = set()

    def score(fn):
        """Best image for one condition, skipping images already used for another example."""
        best = max((i for i in per if i not in chosen), key=lambda i: fn(i, per[i], people[i]))
        chosen.add(best)
        return best

    sizes: dict[int, tuple[int, int]] = {}

    def box_share(i, p):
        """Share of the image covered by a person's box."""
        if i not in sizes:
            sizes[i] = cv2.imread(str(img_dir / f"{i:012d}.jpg")).shape[:2]
        h, w = sizes[i]
        x1, y1, x2, y2 = p["box"]
        return (x2 - x1) * (y2 - y1) / (w * h)

    def missed(i, info, ppl, cond=lambda p: True):
        """Count people in image i that were not matched by any detection and satisfy cond."""
        return sum(1 for p, m in zip(ppl, info["matched"]) if not m and cond(p))

    picks = {
        "small_people_missed": score(lambda i, info, ppl: missed(i, info, ppl, lambda p: p["size"] == "small")),
        # Most people in one image (capped at 20 so the boxes stay readable).
        "crowded_scene": score(lambda i, info, ppl: len(ppl) if len(ppl) <= 20 else -1),
        # Largest missed person with more than half the keypoints hidden, whose box still fits
        # well inside the image (under 25% of it), so the yellow box is visible.
        "occluded_person_missed": score(lambda i, info, ppl: max(
            [p["area"] for p, m in zip(ppl, info["matched"])
             if not m and p["occ"] is not None and p["occ"] > 0.5 and box_share(i, p) < 0.25],
            default=0)),
        "background_false_alarm": score(lambda i, info, ppl: max(
            [d["score"] for d, o in zip(info["dets"], info["outcomes"]) if o == "background"], default=0)),
        "dark_image": score(lambda i, info, ppl: missed(i, info, ppl) if brightness[i] < DARK else -1),
    }
    ex_dir = out / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, iid in picks.items():
        info, ppl = per[iid], people[iid]
        n_miss = missed(iid, info, ppl)
        n_err = sum(o in ("duplicate", "localization", "background") for o in info["outcomes"])
        draw_example(img_dir / f"{iid:012d}.jpg", info, ppl, ex_dir / f"{name}.jpg")
        rows.append({"example": name, "image_id": iid, "people": len(ppl), "missed": n_miss, "false_boxes": n_err,
                     "false_box_types": [f"{o} {d['score']:.2f}" for d, o in zip(info["dets"], info["outcomes"])
                                         if o in ("duplicate", "localization", "background")]})
    return rows


# ---------------------------------------------------------------- report

def to_markdown(res: dict, exp: list[dict] | None, examples: list[dict]) -> str:
    """Format the analysis, experiment, and failure-example tables as Markdown for results.md."""
    rec, ap = res["recall"], res["ap"]
    L = ["# Error analysis: fine-tuned YOLOv8s on COCO val2017 persons", "",
         f"Recall is measured at the deployed threshold (confidence {CONF}, IoU {IOU_MATCH}). "
         "AP is the official COCO AP@0.5:0.95 (pycocotools).", "",
         "## Recall by person size", "", "| Size | People | Found | Recall |", "| --- | --- | --- | --- |"]
    for s, _, _ in SIZE_BINS:
        r = rec[f"size:{s}"]
        L.append(f"| {s} | {r['total']:,} | {r['found']:,} | {r['recall']:.3f} |")
    r = rec["all"]
    L.append(f"| all | {r['total']:,} | {r['found']:,} | {r['recall']:.3f} |")

    L += ["", "## Crowded scenes", "", "| People in image | Images | People | AP | Recall |", "| --- | --- | --- | --- | --- |"]
    for b, _, _ in CROWD_BINS:
        r = rec[f"crowd:{b}"]
        L.append(f"| {b} | {ap[f'crowd:{b}']['images']} | {r['total']:,} | {ap[f'crowd:{b}']['AP']:.3f} | {r['recall']:.3f} |")

    L += ["", "## Partial occlusion", "",
          "Occlusion = share of a person's labeled keypoints that COCO marks as not visible. COCO has keypoints "
          "only for medium and large people, so small people are not in this table.", "",
          "| Hidden keypoints | People | Recall | Recall, medium only | Recall, large only |",
          "| --- | --- | --- | --- | --- |"]
    for b, _, _ in OCC_BINS:
        r = rec[f"occ:{b}"]
        med = rec.get(f"occ:{b}|medium", {"recall": float("nan")})
        lar = rec.get(f"occ:{b}|large", {"recall": float("nan")})
        L.append(f"| {b} | {r['total']:,} | {r['recall']:.3f} | {med['recall']:.3f} | {lar['recall']:.3f} |")

    L += ["", "## Lighting", "",
          f"Mean gray level of the image (0-255): dark < {DARK}, bright > {BRIGHT}.", "",
          "| Lighting | Images | People | AP | Recall |", "| --- | --- | --- | --- | --- |"]
    for b in LIGHT_BINS:
        r = rec[f"light:{b}"]
        L.append(f"| {b} | {ap[f'light:{b}']['images']} | {r['total']:,} | {ap[f'light:{b}']['AP']:.3f} | {r['recall']:.3f} |")

    n = res["detections"]
    L += ["", f"## Detection outcomes at confidence {CONF}", "", f"{n:,} boxes in total.", "",
          "| Outcome | Boxes | Share |", "| --- | --- | --- |"]
    for k in ERROR_TYPES:
        L.append(f"| {k} | {res['outcomes'][k]:,} | {res['outcomes'][k] / n:.1%} |")
    miss = res["missed_by_size"]
    L += ["", "Missed people by size: " + ", ".join(f"{s} {v:,} ({v / sum(miss.values()):.0%})" for s, v in miss.items()) + "."]

    if exp:
        L += ["", "## Experiment: input size and test-time augmentation", "",
              "Same fine-tuned weights, evaluated at different input sizes, with Ultralytics TTA, and with "
              "the 640 and 960 predictions merged by NMS (IoU 0.7, the model's own NMS threshold). "
              "Time is end-to-end per image on one NVIDIA TITAN RTX; the merge time is both passes.", "",
              "| Setting | AP | AP50 | AP small | AP medium | AP large | Recall | Recall small | False boxes | ms / image |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in exp:
            L.append(f"| {r['setting']} | {r['AP']:.3f} | {r['AP50']:.3f} | {r['AP_small']:.3f} | {r['AP_medium']:.3f} | "
                     f"{r['AP_large']:.3f} | {r['recall@0.25']:.3f} | {r['recall_small@0.25']:.3f} | "
                     f"{r['false_positives@0.25']:,} | {r['ms_per_image']:.1f} |")

    L += ["", "## Failure examples", "",
          "Green: person found. Yellow: person missed. Red: false box (dup = duplicate, loc = localization, "
          "bac = background).", "", "| Example | COCO image id | People | Missed | False boxes |",
          "| --- | --- | --- | --- | --- |"]
    for e in examples:
        L.append(f"| [{e['example']}](examples/{e['example']}.jpg) | {e['image_id']} | {e['people']} | {e['missed']} | {e['false_boxes']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    """Run the error analysis and the input-size / TTA experiment, then save figures, examples, and tables."""
    ap = argparse.ArgumentParser(description="Error analysis and scale/TTA experiment.")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "yolov8s_coco_person" / "weights" / "best.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "coco_person")
    ap.add_argument("--ann-dir", type=Path, default=ROOT / "data" / "coco_raw" / "annotations")
    ap.add_argument("--out", type=Path, default=ROOT / "report" / "analysis")
    ap.add_argument("--device", default="0", help="cuda device index, or cpu")
    ap.add_argument("--skip-experiment", action="store_true", help="skip the input-size / TTA runs")
    args = ap.parse_args()

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    img_dir = args.data / "images" / "val2017"
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {img_dir}; run src/prepare_coco.py --splits val")
    img_ids = [int(p.stem) for p in images]

    # 1. Ground truth (people, keypoint occlusion, crowd regions) and per-image brightness.
    print("loading ground truth and image brightness ...", flush=True)
    coco, people, crowds = load_ground_truth(args.ann_dir, img_ids)
    brightness = {int(p.stem): float(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).mean()) for p in images}

    # 2. Predict at the training size (640) and score every box and every person.
    model = YOLO(str(args.weights))
    print("predicting at 640 ...", flush=True)
    dets = predict_coco(model, images, None, args.device, 640, 32)
    res = analyze(dets, coco, people, crowds, img_ids, brightness)
    examples = save_examples(res, people, brightness, img_dir, args.out)

    # 3. Optional experiment: larger inputs, TTA, and the 640 + 960 merge.
    exp = None if args.skip_experiment else scale_experiment(model, images, coco, img_ids, people, crowds, args.device)

    # 4. Save figures, numbers (JSON), and Markdown tables for the report.
    plot_figures(res, exp, args.out)
    saved = {k: v for k, v in res.items() if not k.startswith("_")}
    (args.out / "results.json").write_text(json.dumps({"analysis": saved, "experiment": exp, "examples": examples}, indent=2))
    md = to_markdown(res, exp, examples)
    (args.out / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
