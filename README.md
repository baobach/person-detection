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

## Layout
```
configs/    dataset and training configs (YOLO data yaml)
src/        data preparation, training, evaluation, visualization scripts
notebooks/  exploration and result plots
data/       COCO images and labels (not committed)
runs/       training outputs and weights (not committed)
report/     write-up and figures
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
