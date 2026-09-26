# Rule compliance — HOD Challenge 2026

Rules re-checked on 2026-09-26 against the Kaggle competition record and the organisers' posts
(overview/rules, discussions 727863, 729747, 737136, 739853, 742487, 743224).

| Rule (current organiser wording) | How this repo complies |
|---|---|
| Final CSV from **one trained model / one checkpoint**; no multi-model voting, weighted fusion or cross-model WBF | Every submission CSV is produced by exactly one `.pt` file. `ensemble.py` (multi-model WBF) is kept only for historical reference and must not be used for submissions. |
| Same-checkpoint TTA / multi-scale inference is allowed | `kaggle_kernels/hod-phase2/predict_phase2.py` fuses only passes of the same checkpoint (original + hflip). It is used only if hold-out validation improves. |
| Public pretrained weights allowed if declared | COCO-pretrained Ultralytics `yolo11s.pt` / `yolo26s.pt` (AGPL-3.0, https://github.com/ultralytics/ultralytics). No other external weights. |
| No external labelled data | Training uses only `data_train` annotations. No HOD3K, LivingOptics or other labelled sets. |
| Unlabelled **test** images may be pseudo-labelled (document teacher, thresholds, filtering, iterations) | Not used so far. If used, it will be logged in `experiments.csv`. |
| **Ranking** images are inference-only: no training, pseudo-labels, self-supervision, BN recalibration or state-changing TTA | Ranking images are read only by the inference script. The model is in eval mode; nothing is fitted on them. Per-image intensity scaling (percentile normalisation) is a stateless per-image transform, identical for train/test/ranking. |
| Phase 2 CSV must contain test + ranking (2000 images), `id,image_id,class_id,confidence,x1,y1,x2,y2`, boxes in post-X2Cube pixels | Checked before every submission (schema, id range, class range, coordinates, image coverage). |
| Final selection must be ticked manually ("use for final scoring", up to 2) | The user ticks it on the website. The CLI cannot do this. |
| Phase 2 deadline 2026-09-27 08:00 UTC; top-10 code package by 2026-09-30 08:00 UTC | Tracked in the project notes. |

Excluded images: none. Boxes narrower or shorter than 1 px after clipping to the frame are dropped by the label writer, the same as in validation. The EDA flags 44 tiny or out-of-bounds boxes in total; most of them survive clipping.
