# Error analysis: fine-tuned YOLOv8s on COCO val2017 persons

Recall is measured at the deployed threshold (confidence 0.25, IoU 0.5). AP is the official COCO AP@0.5:0.95 (pycocotools).

## Recall by person size

| Size | People | Found | Recall |
| --- | --- | --- | --- |
| small | 4,308 | 2,134 | 0.495 |
| medium | 3,723 | 3,155 | 0.847 |
| large | 2,746 | 2,586 | 0.942 |
| all | 10,777 | 7,875 | 0.731 |

## Crowded scenes

| People in image | Images | People | AP | Recall |
| --- | --- | --- | --- | --- |
| 1 person | 1045 | 1,045 | 0.799 | 0.922 |
| 2-5 people | 971 | 2,863 | 0.656 | 0.783 |
| 6+ people | 677 | 6,869 | 0.547 | 0.680 |

## Partial occlusion

Occlusion = share of a person's labeled keypoints that COCO marks as not visible. COCO has keypoints only for medium and large people, so small people are not in this table.

| Hidden keypoints | People | Recall | Recall, medium only | Recall, large only |
| --- | --- | --- | --- | --- |
| 0% hidden | 2,768 | 0.922 | 0.885 | 0.956 |
| 1-25% hidden | 2,356 | 0.932 | 0.904 | 0.974 |
| 26-50% hidden | 988 | 0.808 | 0.785 | 0.871 |
| >50% hidden | 240 | 0.604 | 0.568 | 0.703 |

## Lighting

Mean gray level of the image (0-255): dark < 70, bright > 160.

| Lighting | Images | People | AP | Recall |
| --- | --- | --- | --- | --- |
| dark | 214 | 788 | 0.578 | 0.712 |
| normal | 2285 | 9,408 | 0.603 | 0.732 |
| bright | 194 | 581 | 0.587 | 0.728 |

## Detection outcomes at confidence 0.25

9,712 boxes in total.

| Outcome | Boxes | Share |
| --- | --- | --- |
| correct | 7,875 | 81.1% |
| duplicate | 233 | 2.4% |
| localization | 503 | 5.2% |
| background | 382 | 3.9% |
| crowd region (ignored) | 719 | 7.4% |

Missed people by size: small 2,174 (75%), medium 568 (20%), large 160 (6%).

## Experiment: input size and test-time augmentation

Same fine-tuned weights, evaluated at different input sizes, with Ultralytics TTA, and with the 640 and 960 predictions merged by NMS (IoU 0.7, the model's own NMS threshold). Time is end-to-end per image on one NVIDIA TITAN RTX; the merge time is both passes.

| Setting | AP | AP50 | AP small | AP medium | AP large | Recall | Recall small | False boxes | ms / image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 640 | 0.600 | 0.839 | 0.390 | 0.685 | 0.809 | 0.731 | 0.495 | 1,118 | 7.2 |
| 960 | 0.606 | 0.856 | 0.455 | 0.702 | 0.719 | 0.766 | 0.583 | 1,484 | 13.3 |
| 1280 | 0.551 | 0.827 | 0.473 | 0.694 | 0.517 | 0.773 | 0.630 | 2,136 | 21.4 |
| 640 + TTA | 0.599 | 0.827 | 0.382 | 0.682 | 0.824 | 0.758 | 0.524 | 2,092 | 11.4 |
| 640 + 960 merged | 0.622 | 0.851 | 0.445 | 0.705 | 0.796 | 0.791 | 0.607 | 2,057 | 20.4 |

## Failure examples

Green: person found. Yellow: person missed. Red: false box (dup = duplicate, loc = localization, bac = background).

| Example | COCO image id | People | Missed | False boxes |
| --- | --- | --- | --- | --- |
| [small_people_missed](examples/small_people_missed.jpg) | 17959 | 13 | 11 | 0 |
| [crowded_scene](examples/crowded_scene.jpg) | 2299 | 13 | 1 | 0 |
| [occluded_person_missed](examples/occluded_person_missed.jpg) | 297427 | 3 | 2 | 1 |
| [background_false_alarm](examples/background_false_alarm.jpg) | 295713 | 7 | 1 | 2 |
| [dark_image](examples/dark_image.jpg) | 303713 | 13 | 10 | 1 |
