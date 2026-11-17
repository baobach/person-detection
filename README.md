# Person Detection in Images

Course project. Team: Tejas Mahajan, Van Hoang Bao Bach.

## Problem
Detect and localize people in images with bounding boxes. The system must handle multiple people per image, different lighting, backgrounds and scales, and partial occlusion.

## Dataset
[COCO](https://cocodataset.org) (Common Objects in Context), `person` category only (COCO class id 0 in YOLO format).

## Approach
1. Preprocess: filter COCO to images with `person`, convert annotations to YOLO format.
2. Fine-tune a pretrained YOLO model on the person subset.
3. Evaluate: precision, recall, mAP@0.5, mAP@0.5:0.95.
4. Visualize detections on test images.
5. Error analysis: crowded scenes, small persons, occlusion. Try data augmentation and fine-tuning variants.

## Architecture
![Single-class person detection with YOLOv8s](report/person_detection_yolov8s_architecture.png)

YOLOv8s fine-tuned to one class. A C2f and SPPF backbone feeds a bidirectional neck (FPN top-down, PAN bottom-up) and three anchor-free heads at strides 8, 16, and 32.

## Layout
```
configs/    dataset and training configs (YOLO data yaml)
src/        data preparation, training, evaluation, visualization scripts
data/       COCO images and labels (not committed)
runs/       trained YOLOv8s weights and metrics
report/     final report, evaluation and analysis results, figures, sample detections
```

## Result
YOLOv8s was fine-tuned for 50 epochs on the COCO person training images and evaluated on COCO `val2017` (2,693 images, 10,777 people).

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- |
| 83.50% | 72.21% | 81.94% | 58.96% |

The weights are `runs/yolov8s_coco_person/weights/best.pt`. The final report is [`report/final_report.pdf`](report/final_report.pdf) (source: `report/final_report.md`). The earlier two-page results summary is `report/person_detection_report.pdf`. Every validation image with people highlighted, plus the curves and confusion matrices, is in [`full-results/`](full-results/). Train again with:

```bash
uv run python src/train.py --model yolov8s.pt --epochs 50 --batch 32
```

## Evaluation and error analysis
`src/evaluate.py` compares the fine-tuned model with the stock COCO-pretrained YOLOv8s (person class only) on the same val2017 person images. It also breaks the official COCO AP (pycocotools) down by person size and by the number of people in the image. Results go to [`report/eval/results.md`](report/eval/results.md) and `results.json`.

```bash
uv run python src/prepare_coco.py --splits val   # if data/coco_person/images/val2017 is missing
uv run python src/evaluate.py --device 0
```

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| Fine-tuned YOLOv8s | 83.50 | 72.21 | 81.94 | 58.96 |
| Pretrained YOLOv8s, person class | 82.75 | 69.77 | 79.76 | 57.02 |

Small people (under 32x32 px) are 40% of the 10,777 people and the weakest case: COCO AP 0.390 (small) vs 0.685 (medium) and 0.809 (large). AP also falls as images get crowded: 0.799 with one person, 0.656 with 2 to 5, 0.547 with 6 or more. The pretrained model already saw COCO train2017, so fine-tuning gains come from specializing to one class; most of the gain is on small people (+0.027 AP).

`src/analyze.py` covers the remaining conditions from the proposal and one inference experiment. Results, figures, and failure examples go to [`report/analysis/`](report/analysis/results.md).

```bash
uv run python src/analyze.py --device 0          # about 12 minutes on a TITAN RTX
```

* **Occlusion** (share of a person's COCO keypoints marked hidden, medium and large people): recall 0.92 with nothing hidden, 0.81 with 26 to 50% hidden, 0.60 with more than half hidden.
* **Lighting** (mean image brightness): AP 0.578 dark, 0.603 normal, 0.587 bright. A small effect.
* **Error types** at confidence 0.25: 81% of boxes are correct, 5% are localization errors, 4% background, 2% duplicates, and 7% fall in COCO crowd regions (ignored). 75% of the missed people are small.
* **Experiment:** the same weights at a larger input help small people but hurt large ones (AP large 0.809 at 640, 0.517 at 1280). Merging the 640 and 960 predictions with NMS gives the best AP, 0.622 (vs 0.600), with small-person AP 0.445 (vs 0.390).

## Setup
Environment is managed with [uv](https://docs.astral.sh/uv/).
```bash
uv sync                 # creates .venv and uv.lock from pyproject.toml
uv run python src/<script>.py
uv add <package>        # add a dependency
```
