# Evaluation on COCO val2017, person class

2,693 images, 10,777 people (small 4,308, medium 3,723, large 2,746).

## Overall (Ultralytics val)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| Fine-tuned YOLOv8s (best.pt) | 83.50 | 72.21 | 81.94 | 58.96 |
| Pretrained YOLOv8s, person class | 82.75 | 69.77 | 79.76 | 57.02 |

## By person size (COCO AP, pycocotools)

Small: area < 32x32 px. Medium: up to 96x96 px. Large: above.

| Model | AP | AP50 | AP75 | AP small | AP medium | AP large | AR100 | AR small |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fine-tuned YOLOv8s (best.pt) | 0.600 | 0.839 | 0.652 | 0.390 | 0.685 | 0.809 | 0.703 | 0.530 |
| Pretrained YOLOv8s, person class | 0.588 | 0.822 | 0.636 | 0.363 | 0.681 | 0.804 | 0.688 | 0.498 |

## By crowding (COCO AP, pycocotools)

Images grouped by the number of ground-truth people.

| Model | 1 person (1045 img) AP | 2-5 people (971 img) AP | 6+ people (677 img) AP |
| --- | --- | --- | --- |
| Fine-tuned YOLOv8s (best.pt) | 0.799 | 0.656 | 0.547 |
| Pretrained YOLOv8s, person class | 0.782 | 0.651 | 0.530 |
