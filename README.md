# Person Detection in Images

CAP 5415 Computer Vision course project. Team: Tejas Mahajan, Van Hoang Bao Bach.

We fine-tune a pretrained YOLOv8s detector on the person images of COCO 2017, evaluate it on COCO val2017, and analyze where it fails: small people, crowded scenes, partial occlusion, and lighting. The full write-up is [`report/final_report.pdf`](report/final_report.pdf).

## Results

Fine-tuned for 50 epochs on 64,115 COCO train2017 person images. Evaluated on the 2,693 val2017 images with people (10,777 people).

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| Pretrained YOLOv8s, person class only | 82.75% | 69.77% | 79.76% | 57.02% |
| **Fine-tuned YOLOv8s (ours)** | **83.50%** | **72.21%** | **81.94%** | **58.96%** |

Main findings (details in the report, Sections 6 and 7):

* **Size** is the main weakness. Recall is 0.94 for large people and 0.50 for small people (under 32x32 px). 75% of the missed people are small.
* **Crowding:** COCO AP is 0.80 in images with one person and 0.55 in images with six or more.
* **Occlusion:** recall stays at 0.92 with up to a quarter of the body hidden, and falls to 0.60 when more than half is hidden.
* **Lighting** has a small effect: AP 0.578 in dark images vs 0.603 in normal ones.
* **False boxes** are mostly loose boxes near a person. Only 4% of boxes are false alarms on the background.
* **Experiment:** a larger input size helps small people but hurts large ones. Merging the predictions at 640 and 960 with NMS raises COCO AP from 0.600 to 0.622, without retraining.

![Single-class person detection with YOLOv8s](report/person_detection_yolov8s_architecture.png)

## Repository layout

```
configs/coco_person.yaml        YOLO dataset config (written by prepare_coco.py)
src/
  prepare_coco.py               download COCO person subset, convert labels to YOLO format
  train.py                      fine-tune YOLOv8s (Ultralytics)
  evaluate.py                   fine-tuned vs pretrained, COCO AP by size and crowding
  analyze.py                    occlusion, lighting, error types, failure examples, input-size experiment
scripts/train_newton.slurm      training job for the UCF Newton cluster (one V100)
runs/yolov8s_coco_person/       trained weights (best.pt, last.pt) and the per-epoch training log
full-results/                   every val2017 image with detections, detections.csv, curves, confusion matrices
report/
  final_report.md, .pdf         final report
  eval/                         output of evaluate.py
  analysis/                     output of analyze.py: tables, figures, failure examples
  samples/                      sample detections used in the report
  person_detection_report.pdf   earlier two-page results summary
data/                           COCO images and labels (not committed, made by prepare_coco.py)
```

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/) (Python 3.11).

```bash
uv sync
```

## Reproduce

Run every command from the repository root.

1. Prepare the data. This downloads the COCO annotations (about 240 MB) and only the images that contain people.

   ```bash
   uv run python src/prepare_coco.py                  # train and val
   uv run python src/prepare_coco.py --splits val     # val only, enough for evaluation
   uv run python src/prepare_coco.py --max-images 200 # small smoke-test subset
   ```

2. Train (optional, the trained weights are in `runs/`). 50 epochs take about 11 hours on one V100.

   ```bash
   uv run python src/train.py --model yolov8s.pt --epochs 50 --batch 32
   sbatch scripts/train_newton.slurm                  # same run on Newton (see Notes)
   ```

3. Evaluate and analyze. Both use `runs/yolov8s_coco_person/weights/best.pt` by default.

   ```bash
   uv run python src/evaluate.py --device 0           # about 5 min, writes report/eval/
   uv run python src/analyze.py --device 0            # about 12 min, writes report/analysis/
   ```

## Notes

* COCO test2017 has no public labels, so val2017 is both the validation set during training and the final test set.
* COCO labels dense groups of people as one *crowd* region. Training drops these boxes, and evaluation ignores detections inside them, as the official COCO evaluation does.
* Two metric families are reported. Ultralytics validation gives precision, recall, and mAP (the table above). The official COCO evaluation (pycocotools) gives AP by size and condition. Its AP is about 1 point higher because it interpolates the precision-recall curve differently.
* `--device` is a PyTorch CUDA index. PyTorch orders GPUs fastest first, which can differ from `nvidia-smi`. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to make them agree.
* Automatic mixed precision is off in `train.py`, because its start-up check hung on the Newton V100 nodes.
* `scripts/train_newton.slurm` calls `prepare_coco.py --from-zip`, an option from a local version of the script that is not committed yet. Until it is, remove that flag or prepare the data before submitting the job.
