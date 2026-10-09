# Ultralytics YOLO26 Performance Benchmarks

## Detection (COCO Dataset)

| Model | Size | mAP 50-95 | mAP 50-95 (E2E) | CPU ONNX (ms) | T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|------|-----------|-----------------|---------------|------------------|------------|-----------|
| YOLO26n | 640 | 40.9 | 40.1 | **38.9** | **1.7** | **2.4** | **5.5** |
| YOLO26s | 640 | 48.6 | 47.8 | 87.2 | 2.5 | 9.5 | 20.9 |
| YOLO26m | 640 | 53.1 | 52.5 | 220.0 | 4.7 | 20.4 | 68.4 |
| YOLO26l | 640 | 55.0 | 54.4 | 286.2 | 6.2 | 24.8 | 86.8 |
| YOLO26x | 640 | **57.5** | **56.9** | 525.8 | 11.8 | 55.7 | 194.4 |

**Key Insights:**
- **Best AP-Latency Tradeoff**: YOLO26m (53.1 mAP at 4.7ms)
- **Fastest Inference**: YOLO26n (1.7ms on T4 TensorRT)
- **Smallest Model**: YOLO26n (2.4M parameters)
- **YOLO26n vs YOLO11n**: 43% faster CPU ONNX inference

## Instance Segmentation (COCO Dataset)

| Model | Size | mAP box 50-95 (E2E) | mAP mask 50-95 (E2E) | CPU ONNX (ms) | T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|------|---------------------|----------------------|---------------|------------------|------------|-----------|
| YOLO26n-seg | 640 | 39.6 | 33.9 | **53.3** | **2.1** | **2.7** | **9.3** |
| YOLO26s-seg | 640 | 47.3 | 40.0 | 118.4 | 3.3 | 10.4 | 34.5 |
| YOLO26m-seg | 640 | 52.5 | 44.1 | 328.2 | 6.7 | 23.6 | 121.7 |
| YOLO26l-seg | 640 | 54.4 | 45.5 | 387.0 | 8.0 | 28.0 | 140.1 |
| YOLO26x-seg | 640 | **56.5** | **47.0** | 787.0 | 16.4 | 62.8 | 314.0 |

**Improvements over YOLO11:**
- Box AP: +2.5 mAP
- Mask AP: +3.7 mAP

## Semantic Segmentation (Cityscapes Dataset)

| Model | Input Size | mIoU | RTX3090 PyTorch (ms) | Params (M) | FLOPs (B) |
|-------|------------|------|----------------------|------------|-----------|
| YOLO26n-sem | 1024×2048 | 78.3 | **4.4** | **1.6** | **23.8** |
| YOLO26s-sem | 1024×2048 | 80.8 | 8.4 | 6.5 | 91.0 |
| YOLO26m-sem | 1024×2048 | 82.0 | 19.9 | 14.3 | 305.5 |
| YOLO26l-sem | 1024×2048 | 82.9 | 26.5 | 17.8 | 388.2 |
| YOLO26x-sem | 1024×2048 | **83.6** | 48.9 | 40.1 | 866.9 |

## Depth Estimation (NYU Depth V2 Dataset)

| Model | Size | delta1 | abs_rel | rmse | CPU ONNX (ms) | T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|------|--------|---------|------|---------------|------------------|------------|-----------|
| YOLO26n-depth | 768 | 0.882 | 0.109 | 0.414 | **272.0** | **2.7** | **6.3** | **46.9** |
| YOLO26s-depth | 768 | 0.896 | 0.104 | 0.399 | 393.7 | 3.8 | 13.2 | 68.0 |
| YOLO26m-depth | 768 | 0.921 | 0.089 | 0.364 | 621.5 | 6.0 | 23.3 | 130.4 |
| YOLO26l-depth | 768 | 0.930 | 0.083 | 0.351 | 821.9 | 7.7 | 27.7 | 157.0 |
| YOLO26x-depth | 768 | **0.933** | **0.080** | **0.344** | 1240.9 | 13.6 | 57.0 | 301.7 |

## Classification (ImageNet Dataset)

| Model | Size | top1 acc | top5 acc | CPU ONNX (ms) | T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|------|----------|----------|---------------|------------------|------------|-----------|
| YOLO26n-cls | 224 | 71.4 | 90.1 | **5.0** | **1.1** | **2.8** | **0.4** |
| YOLO26s-cls | 224 | 76.0 | 92.9 | 7.9 | 1.3 | 6.7 | 1.5 |
| YOLO26m-cls | 224 | 78.1 | 94.2 | 17.2 | 2.0 | 11.6 | 4.8 |
| YOLO26l-cls | 224 | 79.0 | 94.6 | 23.2 | 2.8 | 14.1 | 6.0 |
| YOLO26x-cls | 224 | **79.9** | **95.0** | 41.4 | 3.8 | 29.6 | 13.5 |

## Pose Estimation (COCO Dataset)

| Model | Size | mAP pose 50-95 (E2E) | mAP pose 50 (E2E) | CPU ONNX (ms) | T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|------|----------------------|-------------------|---------------|------------------|------------|-----------|
| YOLO26n-pose | 640 | 57.2 | 83.3 | **40.3** | **1.8** | **2.9** | **7.6** |
| YOLO26s-pose | 640 | 63.0 | 86.6 | 85.3 | 2.7 | 10.4 | 24.1 |
| YOLO26m-pose | 640 | 68.8 | 89.6 | 218.0 | 5.0 | 21.5 | 73.3 |
| YOLO26l-pose | 640 | 70.4 | 90.5 | 275.4 | 6.5 | 25.9 | 91.7 |
| YOLO26x-pose | 640 | **71.6** | **91.6** | 565.4 | 12.2 | 57.6 | 202.3 |

**Improvements over YOLO11:**
- Pose AP: +7.2 mAP

## Oriented Detection (DOTA-v1.0 Dataset)

| Model | Size | mAP test 50-95 (E2E) | mAP test 50 (E2E) | CPU ONNX (ms) | T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|------|----------------------|-------------------|---------------|------------------|------------|-----------|
| YOLO26n-obb | 1024 | 52.4 | 78.9 | **97.7** | **2.8** | **2.4** | **14.8** |
| YOLO26s-obb | 1024 | 54.8 | 80.9 | 218.0 | 4.9 | 9.8 | 56.7 |
| YOLO26m-obb | 1024 | 55.3 | 81.0 | 579.2 | 10.2 | 21.2 | 184.9 |
| YOLO26l-obb | 1024 | 56.2 | 81.6 | 735.6 | 13.0 | 25.6 | 232.4 |
| YOLO26x-obb | 1024 | **56.7** | **81.7** | 1485.7 | 30.5 | 57.6 | 520.1 |

**Improvements over YOLO11:**
- OBB mAP: +3.4 mAP

## YOLOE-26 Open-Vocabulary Detection (LVIS Dataset)

| Prompt Type | YOLOE-26x AP | YOLOE-11x AP | Gain |
|-------------|--------------|--------------|------|
| Text Prompting | **40.6** | 37.7 | +2.9 |
| Visual Prompting | **38.5** | 35.9 | +2.6 |
| Prompt-Free | **31.1** | 29.1 | +2.0 |

**Lightweight Variant:**
- YOLOE-26n: 24.7 AP (text), 3.9M parameters

## Comparative Analysis

### Accuracy-Latency Pareto Front

YOLO26 variants sit on or advance the Pareto front over prior real-time detectors, with the strongest AP-latency trade-off at medium, large, and extra-large scales.

### CPU Inference Speedup vs YOLO11

| Model | YOLO11n (ms) | YOLO26n (ms) | Speedup |
|-------|--------------|--------------|---------|
| CPU ONNX | 69.0 | **38.9** | **43% faster** |

---
*Device: Laptop Linux (LTLN) RTX 500 Ada, Intel Core Ultra 7 255H, 32GB 5600MT/s*
*Note: Speed measurements on Intel Xeon CPU @ 2.00 GHz (ONNX) and NVIDIA T4 TensorRT10 FP16*