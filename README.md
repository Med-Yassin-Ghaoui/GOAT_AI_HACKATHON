# Monocular Depth Estimation — AIGOAT 1.0 (Task 2)

> Predicting a per-pixel depth map from a single RGB image, with a model small and fast enough to win on a size/speed-weighted leaderboard.

**Hackathon:** AIGOAT 1.0 · **Team:** *PPP: PeniParkersPrime* · 🏆 **Final rank: #2** · **Score: 1.47**

---

## The problem

Given a single `448×448` RGB image — no stereo pair, no LiDAR, no camera intrinsics — predict a dense **depth map** of the same resolution, where each pixel encodes its relative distance from the camera. Because absolute scale is ambiguous from one image, predictions are evaluated **scale-invariantly**: both prediction and ground truth are normalized to `[0,1]` before scoring.

The catch is the scoring. The leaderboard doesn't just reward accuracy — it multiplies three components together:

```
Score = Accuracy(RMSE)  ×  SizeScore(MB)  ×  SpeedScore(seconds)
```

So a heavy, ultra-accurate model loses to a model that is *accurate, tiny, and fast all at once*. Hard limits: RMSE < 0.16, size < 100 MB, inference < 10 s. The whole design is an exercise in balancing those three axes.

## Results

| Metric | Value | Notes |
|---|---|---|
| Validation RMSE | **0.1077** | comfortably under the 0.16 cutoff |
| Model size (ONNX, FP16) | **3.71 MB** | vs. a 100 MB ceiling |
| Parameters | 1.93 M | |
| Inference (batch 8, GPU) | ~1.37 s | measured on the eval server |
| **Composite score** | **1.47** | **Rank #3** on the final leaderboard |

## Approach

The model is a compact encoder–decoder ("U-Net style") trained with a mix of techniques to squeeze accuracy out of a deliberately small network:

- **MobileNetV2 encoder + depthwise-separable decoder** — a mobile-grade backbone keeps the parameter count (and therefore the file size) low, which directly boosts the size score.
- **Knowledge distillation from [MiDaS](https://github.com/isl-org/MiDaS)** — the pretrained MiDaS depth model is run once over the whole training set, and its predictions are cached and used as an auxiliary "teacher" target. Our small model learns MiDaS's sense of scene structure on top of the ground-truth labels (the teacher correlated with the labels at Pearson **r = 0.985**).
- **FP16 ONNX export** — storing weights in half precision nearly halves the file size (7.7 MB → **3.71 MB**) with no measurable accuracy loss.
- **Folded ImageNet normalization** — instead of normalizing inputs at runtime, the ImageNet mean/std are baked into the first convolution's weights. The exported model accepts raw `[0,1]` RGB and does zero preprocessing, exactly as the eval server expects.
- **EMA (exponential moving average) of weights** — validation and the final export use smoothed weights for better generalization.
- **Composite loss** — L1 + gradient matching + SSIM + a scheduled scale-invariant term, capturing both per-pixel error and overall scene geometry.
- **Heavy augmentation** — flips, small rotations, random crops, blur, and photometric jitter (gamma/brightness/contrast/saturation/noise) to combat overfitting.

Training: 20 epochs, one-cycle LR, mixed precision (AMP), gradient clipping, early stopping on validation RMSE. Trained on Kaggle with a single Tesla T4 GPU over 9,200 image–depth pairs.

## Repository contents

| File | Description |
|---|---|
| [`depth_estimation.ipynb`](depth_estimation.ipynb) | The full pipeline — data loading, MiDaS teacher pass, model, training, ONNX export, benchmark, and leaderboard submission. **Cell outputs from the original run are preserved.** |
| [`TASK2_DOCS.pdf`](TASK2_DOCS.pdf) | The official competition guidelines (task spec, dataset format, scoring formulas). |

## Reproducing

The notebook was written for the **Kaggle** environment (Tesla T4 GPU, dataset mounted under `/kaggle/input`). It is **not** runnable as-is outside that setup — it needs the competition dataset and a CUDA GPU — which is why the outputs are kept inline so you can read the results without re-running.

To adapt it elsewhere: point `find_dataset_dir()` / `OUTPUT_DIR` at your local paths, ensure the data follows the `{id}_image.png` / `{id}_depth_map.png` naming convention, and install `torch`, `torchvision`, `onnx`, `onnxruntime-gpu`, `pillow`, and `tqdm`.

## Tech stack

`PyTorch` · `torchvision (MobileNetV2)` · `MiDaS` · `ONNX` / `ONNX Runtime` · `NumPy` · `Pillow`
