# 16 bands beat any 3-band composite: a single-checkpoint YOLO solution and what hold-out validation missed

**Hyperspectral Object Detection Challenge 2026 · Team: Md. Hamid Hosen**
Code: <https://github.com/hamidhosen42/Hyperspectral-Object-Detection-Challenge-2026>  
Kaggle write-up: <https://www.kaggle.com/w/116724> · DOI: [10.34740/KAGGLE/W/116724](https://doi.org/10.34740/KAGGLE/W/116724)

---

## TL;DR

- **Single checkpoint, no ensembling.** The model is Ultralytics YOLO (11s / 26s) with a **native 16-channel input stem**, trained on all 16 X2Cube bands at 1024 px.
- **All 16 bands vs a 3-band pseudo-RGB composite:** test mAP@[.5:.95] rose from **0.565 to 0.591** with an otherwise identical recipe. The random hold-out showed only +0.003. Best public test score: **0.5955** (YOLO26s, 16 bands).
- **Train, test and ranking are three distinct acquisition domains.** A classifier on simple image statistics separates them with AUC ≈ 0.90. I built a test-like STRESS validation subset; its AP75 predicted the test ranking of my early checkpoints better than plain mAP did.
- **The real/counterfeit pairs are not where points are lost.** Material classification of matched boxes was 100% on validation; localisation at high IoU is the bottleneck.

---

## 1. Data understanding

The dataset forensics ran as a Kaggle kernel over all 4,000 train + test images.

| Property | Finding |
|---|---|
| Raw format | 16-bit greyscale mosaic. Values use about 9 bits (max ≈ 300). `X2Cube` gives ~512×256×16; frame size varies by a few px. |
| Objects | 10,107 boxes, about 3.4 per image. **70% are COCO-small**; median side 24 px, 5th–95th percentile 13–63 px. |
| Labels | 44 tiny or out-of-bounds boxes, 151 truncated, no duplicates |
| Bands | Highly correlated (mean r = 0.82; PCA: 2 components explain 97% of pixel variance) |

**Domain shift.** I trained a gradient-boosted classifier on per-image radiometric features: band means and standard deviations, relative band shape, and percentiles.

| Comparison | Adversarial AUC |
|---|---|
| train vs test | **0.89** |
| train vs ranking | **0.90** |
| test vs ranking | **0.90** |
| val vs train (control) | 0.51 |

The biggest differences are in the *relative* band intensities. Relative to train, ranking images are about +7–13% brighter in bands 0–5 and about −3–5% darker in bands 10–15. Test lies about a third of the way along that same direction. A random hold-out from train therefore cannot see this shift.

**Spectra separate the look-alike materials.** I trained a linear discriminant on the mean spectrum inside each box, cross-validated by image:

| Group | Pseudo-RGB (bands 5/8/13) | All 16 bands | 16 bands, brightness-normalised |
|---|---|---|---|
| apple vs plastic | 0.87 | 0.998 | **1.00** |
| banana vs plastic | 0.68 | 0.93 | **0.996** |
| orange vs plastic | 0.91 | 0.985 | **0.998** |
| egg / plastic / wood | 0.69 | **0.95** | 0.95 |
| car vs toy car | 0.75 | **0.95** | 0.94 |

---

## 2. Validation

Every model was scored with pycocotools, by the same script, on:
- **ALL:** a fixed 300-image hold-out. Near-duplicate rate to train is 1.7%, no higher than within train.
- **STRESS:** the 100 hold-out images the adversarial classifier rates most test-like.
- **Shift-stress:** hold-out images with the measured train→ranking per-band gains applied before preprocessing.

Metrics: mAP50-95, AP50, AP75, AP by size, per-class AP, and a real-vs-counterfeit confusion matrix.

**Did validation predict the leaderboard?** Four early checkpoints had both a hold-out score and a test score:
- **STRESS AP75 ranked all four in the true test order.**
- Plain mAP got 2 of the 6 pairwise comparisons wrong.

That is a small sample, but enough to make STRESS AP75 my main gate.

---

## 3. Model and training

```
16-bit mosaic → X2Cube (4×4) → per-image percentile (0.5–99.5%) → uint8 → 16-page TIFF
             → YOLO (COCO-pretrained, 16-channel stem) @1024, 50 epochs, 2×T4, AMP
```

- **Stem:** Ultralytics copies RGB weights into channels 0–2 and initialises channels 3–15 randomly. That worked; a mean-RGB initialisation for all 16 channels is implemented but was not needed.
- **Augmentation:** Ultralytics defaults (mosaic, scale, flip). HSV jitter is off for more than 3 channels.
- **Inference:** confidence 0.001, NMS IoU 0.6, max_det 300. Same-checkpoint hflip TTA (WBF) was neutral on validation, so it was not used.
- **Two practical traps:**
  1. Ultralytics' predictor reads multi-page TIFF paths as 3-channel images, so you must pass the 16-band arrays yourself.
  2. `cache='disk'` writes `.npy` files into the image folders, so prediction globs must filter by extension.

---

## 4. Results (all measured)

| Run | Model | Input | Val mAP | Val AP75 | STRESS AP75 | Test (public) | Ranking (private)* |
|---|---|---|---|---|---|---|---|
| E0a | YOLO11s | pseudo-RGB | 0.693 | 0.821 | 0.803 | 0.5791 | – |
| E0k | YOLO11s | pseudo-RGB | 0.696 | 0.829 | 0.797 | 0.5649 | – |
| E0b | YOLO26s | pseudo-RGB | **0.704** | **0.842** | **0.835** | 0.5803 | 0.4688 |
| E15 | YOLO26s | pseudo-RGB + test pseudo-labels | 0.701 | 0.840 | 0.824 | – | – |
| E4a | YOLO11s | **16 bands** | 0.699 | 0.838 | 0.829 | 0.5912 | **0.5192** |
| E4c | YOLO26s | **16 bands** | 0.703 | 0.839 | 0.830 | **0.5955** | 0.5186 |

\* My Phase 2 files contained predictions for **841 of the 1,000 ranking images** (see §6), so the ranking scores are lower bounds.

---

## 5. What did not help (or could not be shown to help)

- **Test-set pseudo-labels** (one round, E0b teacher, confidence ≥ 0.5: 956 images, 3,163 boxes): −0.003 mAP and −0.011 STRESS AP75 on the hold-out. Rejected.
- **Same-checkpoint flip TTA:** within ±0.005, inconsistent across models.
- **Per-band normalisation** to cancel the illumination tilt: deprioritised. Applying the measured ranking gains to the hold-out cost less than 0.006 mAP, so a *global* tilt is not what hurts. The val→test drop has other causes, such as scene content, object scale and class mix.
- **YOLO26s vs YOLO11s on 16 bands:** +0.004 on both validation and test. Small, but consistent.

---

## 6. Lessons

1. **Validate on the shift.** The single biggest gain, all 16 bands, was invisible on a random hold-out and only showed up on the shifted test domain.
2. **Secure the data before you need it.** The ranking set was released 48 h before the deadline:
   - The competition data bundle did not mount in notebooks.
   - Per-file API downloads were rate-limited (HTTP 429) after about 840 files.
   - I submitted with 159 ranking images missing, which cost a large share of the Phase 2 score.
3. **Test the inference path as carefully as training.** Both of my multi-channel failures happened *after* hours of training succeeded.

---

## 7. Compliance

- One trained checkpoint per submission.
- COCO-pretrained Ultralytics weights only (AGPL-3.0, declared). No external labelled data.
- Ranking images were used for inference only.
- Details are in `RULE_COMPLIANCE.md` in the repository.

Thanks to the organisers (HotTracking2025) for a well-run competition with a genuinely interesting distribution-shift problem.
