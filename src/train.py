# Fine-tune a pretrained YOLO model on the COCO person subset.
#
# Usage (from repo root, after src/prepare_coco.py):
#   uv run python src/train.py
#   uv run python src/train.py --model yolov8s.pt --epochs 50 --batch 32
#
# Outputs land in runs/<name>/ (weights, metrics, plots). Ultralytics also
# runs validation at the end of training and writes results.csv there.

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO on the COCO person subset.")
    ap.add_argument("--model", default="yolov8s.pt", help="pretrained checkpoint to fine-tune")
    ap.add_argument("--data", type=Path, default=ROOT / "configs" / "coco_person.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0", help="cuda device index, or cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", type=Path, default=ROOT / "runs")
    ap.add_argument("--name", default="yolov8s_coco_person")
    ap.add_argument("--patience", type=int, default=20, help="early-stop patience in epochs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        patience=args.patience,
        seed=args.seed,
        cos_lr=True,
        plots=True,
        val=True,
        # Ultralytics' automatic mixed-precision probe hangs on these V100s.
        amp=False,
    )
    run_dir = args.project / args.name
    csv_path = run_dir / "results.csv"
    print(f"TRAIN_DONE weights={run_dir / 'weights' / 'best.pt'}", flush=True)
    if csv_path.exists():
        lines = csv_path.read_text().strip().splitlines()
        print("RESULTS_HEADER " + lines[0], flush=True)
        print("RESULTS_LAST " + lines[-1], flush=True)


if __name__ == "__main__":
    main()
