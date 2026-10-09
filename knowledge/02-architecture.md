# Ultralytics YOLO26 Architecture

## Core Design Philosophy

YOLO26 is built around three design goals:
1. **End-to-end simplicity** - Native NMS-free inference
2. **Deployment efficiency** - Lighter head, simpler exports
3. **Stronger optimization** - MuSGD, Progressive Loss, STAL

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOLO26 Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│  Backbone (Feature Extraction)                                  │
│     └─ Shared across all tasks (detection, segmentation, etc.)  │
├─────────────────────────────────────────────────────────────────┤
│  Neck (Feature Fusion)                                          │
│     └─ Multi-scale feature aggregation                          │
├─────────────────────────────────────────────────────────────────┤
│  Dual Detection Heads                                           │
│  ├─ One-to-One Head (Default)                                   │
││     ├─ NMS-free end-to-end inference                           │
││     ├─ Output: (N, 300, 6) - up to 300 detections              │
││     └─ Progressive Loss emphasis (0.8→0.9 over training)       │
│  └─ One-to-Many Head (Optional)                                 │
│     ├─ Dense prediction with NMS post-processing                │
│     ├─ Output: (N, nc+4, 8400)                                  │
│     └─ Higher accuracy, additional compute                      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Architectural Features

### 1. Dual-Head Design

| Head | Purpose | NMS | Output Shape | Training Emphasis |
|------|---------|-----|--------------|-------------------|
| One-to-One | End-to-end inference | ❌ No | (N, 300, 6) | Progressive (increases) |
| One-to-Many | Maximum accuracy | ✅ Yes | (N, nc+4, 8400) | Progressive (decreases) |

**Usage:**
```python
# Default: one-to-many with NMS
results = model.predict("image.jpg")

# NMS-free: one-to-one head
results = model.predict("image.jpg", nms=False)
```

### 2. DFL-Free Box Regression

**Distribution Focal Loss (DFL) Removal:**
- Traditional YOLO models use DFL: 4 boxes → 4×K logits (K=16)
- YOLO26 uses **direct regression**: 4 scalar values per box
- **Benefits:**
  - -0.3M parameters (12% reduction)
  - -1.4 GFLOPs (20% reduction)
  - Unconstrained regression range (no finite support)
  - Simpler export and deployment

**Why DFL Removal Works:**
- Complementary training objectives compensate:
  - **Progressive Loss** - Better optimization alignment
  - **STAL** - Improved small-object supervision
  - **L1 Loss** - Direct box regression supervision

### 3. Backbone & Neck

- Built on YOLO11 foundation
- Optional P2 (small objects) or P6 (large inputs) additions
- Architecture files: `yolo26-p2.yaml`, `yolo26-p6.yaml`

## Task-Specific Extensions

### Instance Segmentation
- **Multi-Scale Proto Module**: Fuses features from multiple pyramid levels
- **Auxiliary Semantic Loss**: BCE + Dice for better convergence
- **Gains**: +3.7 mask AP over YOLO11 on COCO

### Pose Estimation
- **Residual Log-Likelihood Estimation (RLE)**: Uncertainty-aware keypoint localization
- **Sigma Branch**: Predicts per-axis uncertainty (σx, σy)
- **Gains**: +7.2 pose AP over YOLO11 on COCO

### Oriented Detection (OBB)
- **Long-Edge Angle Definition**: [-45°, 135°) range (vs. [0°, 90°])
- **Dedicated Angle Loss**: Sin²(2Δθ) for square object stability
- **Aspect-Ratio Aware**: Stronger supervision for elongated boxes
- **Gains**: +3.4 mAP over YOLO11 on DOTA-v1.0

### Depth Estimation
- Multi-scale feature fusion
- Pretrained on diverse dataset mix
- Evaluated on NYU Depth V2

### Classification
- Standard Ultralytics Classify head
- Reuses shared backbone
- ImageNet pretrained weights

## Architecture Variants

| Variant | Description | Use Case |
|---------|-------------|----------|
| yolo26-p2.yaml | Adds P2 detection head | Small object detection |
| yolo26-p6.yaml | Adds P6 detection head | Large input resolution |
| yolo26n.pt | Nano scale (2.4M params) | Edge devices |
| yolo26x.pt | Extra-large (55.7M params) | High accuracy needs |

---
*Device: Laptop Linux (LTLN) RTX 500 Ada, Intel Core Ultra 7 255H, 32GB 5600MT/s*