# Ablation study (B-10)

MIT Indoor 67 test set, n = 1,340. Inference only; no model was retrained.

| Configuration | Test top-1 | Test top-5 | Note |
|---|---|---|---|
| ResNet-50 Places365, raw weights | 77.91% | 94.33% | no EMA, no TTA |
| ResNet-50 Places365, EMA (SERVED) | 83.21% | 97.76% | the deployed model |
| ResNet-50 Places365, EMA + TTA | 83.88% | 97.84% | test-time augmentation |
| Fusion (CNN + HOG), full | 82.01% | 96.49% | disabled from serving |
| Fusion, HOG branch zeroed | 80.45% | 96.49% | TEST-TIME ablation, not retrained |
| ResNet-18 ImageNet baseline | 70.82% | 91.57% | CONTEXT: different arch + recipe |

## How to read this

- **HOG row is a test-time ablation.** The fusion head was trained with real
  HOG features; here that branch is zeroed at inference. It measures how much
  the trained head leans on HOG, not how a head trained without HOG would do.
- **Top-1/top-5 are invariant to temperature scaling** (dividing logits by
  T > 0 is monotonic), so calibration is not applied; it moves confidences,
  not accuracy.
- **The ResNet-18 row is context, not a controlled ablation** - it differs in
  architecture and training recipe, so it does not isolate Places365 vs
  ImageNet pretraining. That control needs an identically-trained ImageNet
  ResNet-50.
