# Ultralytics YOLO26 Training Methodology

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOLO26 Training Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│  Backbone + Neck (Shared)                                       │
│     └─ Feature extraction and multi-scale fusion                │
├─────────────────────────────────────────────────────────────────┤
│  Dual-Head Detection                                            │
│  ├─ One-to-Many Head (Dense)                                    │
││     ├─ TAL Assignment: topk = 10                               │
││     ├─ Loss Weight: α(t) (decreases over training)             │
││     └─ Purpose: Rich supervision, higher accuracy              │
│  └─ One-to-One Head (End-to-End)                                │
│     ├─ TAL Assignment: topk = 7, topk2 = 1                      │
│     ├─ Loss Weight: 1 - α(t) (increases over training)          │
│     └─ Purpose: NMS-free inference, direct decoding             │
├─────────────────────────────────────────────────────────────────┤
│  Label Assignment                                               │
│     └─ STAL: Ensures small-object positive coverage             │
├─────────────────────────────────────────────────────────────────┤
│  Loss Combination                                               │
│     └─ Progressive Loss: α(t)·L_one2many + (1-α(t))·L_one2one   │
├─────────────────────────────────────────────────────────────────┤
│  Optimizer                                                      │
│     └─ MuSGD: Hybrid Muon + SGD optimizer                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Training Innovations

### 1. MuSGD Optimizer

**What is MuSGD?**
- Hybrid optimizer combining **Muon** + **SGD**
- Adapted from large language model training to computer vision
- Applies momentum updates followed by orthogonalization

**How it Works:**
```
For high-dimensional parameters (e.g., conv kernels, linear weights):
  1. Apply Muon update (momentum + orthogonalization)
  2. Apply SGD update (standard momentum)
  3. Combine with weighted mixture

For 1D parameters (biases, normalization scales):
  Use pure SGD (stable scale/shift parameters)
```

**Benefits:**
- **Faster convergence**: Reaches target accuracy in fewer epochs
- **Improved stability**: Better conditioning of update directions
- **Transferable**: Works across detection and classification tasks

**Performance Impact:**
| Optimizer | Epochs | COCO mAP |
|-----------|--------|----------|
| SGD | 600 | 47.0 |
| MuSGD | 500 | **47.4** |

### 2. Progressive Loss

**Problem:**
- Dense one-to-many branch is easier to optimize early in training
- One-to-one branch (end-to-end) needs more epochs to converge
- Fixed loss weights under-utilize this asymmetry

**Solution:**
- Curriculum-style reweighting that shifts emphasis over training
- Early training: Emphasize dense branch for stability
- Late training: Emphasize end-to-end branch for deployment alignment

**Schedule:**
```
α(t) = max(1 - t/(E-1), 0)

where:
  t = current epoch
  E = total epochs
  α_init = 0.8 (one-to-many weight)
  α_final = 0.1 (one-to-many weight)

Total Loss = α(t)·L_one2many + (1-α(t))·L_one2one
```

**Impact:**
- Improves end-to-end mAP from 46.4 to **46.7**
- Better alignment with deployment-time behavior

### 3. Small-Target-Aware Label Assignment (STAL)

**Problem:**
- Task-Aligned Learning (TAL) selects anchors inside ground-truth boxes
- Tiny objects may have no anchor centers after downsampling
- Results in zero positive assignments for small objects

**Solution:**
- Decouple candidate selection from regression geometry
- Use enlarged surrogate boxes for candidate filtering only
- Preserve original boxes for final assignment and regression

**STAL Formula:**
```
For ground-truth box g_i = (x_i, y_i, w_i, h_i):
  s_min = smallest feature pyramid stride
  
  For each dimension d_i ∈ {w_i, h_i}:
    if d_i < s_min:
      d_tilde_i = s_ref  # Reference size (typically next stride)
    else:
      d_tilde_i = d_i
  
  Surrogate box: g_tilde_i = (x_i, y_i, d_tilde_w, d_tilde_h)
```

**Benefits:**
- Guarantees positive label coverage for tiny objects
- +0.2 AP on COCO for small objects (AP_S)
- No impact on localization quality (original boxes preserved)

### 4. DFL-Free Direct Regression

**Traditional DFL:**
- Predicts 4 boxes as distributions over K bins (K=16)
- Output: 4×K = 64 values per location
- Finite regression range: (K-1)×stride pixels

**YOLO26 Approach:**
- Direct scalar regression: 4 values per location
- Unconstrained regression range
- Simpler export and deployment

**Compensation:**
- L1 loss for direct supervision
- Progressive Loss for better optimization
- STAL for small-object coverage

## Training Recipe

### Objects365 Pretraining Stage

| Setting | Value |
|---------|-------|
| Dataset | Objects365-v1 |
| Epochs | 150 |
| Batch Size | 128 |
| Augmentations | Heavy (Mosaic, mixup, copy-paste) |

### COCO Fine-tuning Stage

| Model | Epochs |
|-------|--------|
| YOLO26n | 245 |
| YOLO26s | 70 |
| YOLO26m | 80 |
| YOLO26l | 60 |
| YOLO26x | 40 |

**Global Batch Size:** 128 (across all scales)

### Augmentation Strategy

- **Mosaic**: Heavy during most of training, disabled near end
- **Mixup**: Applied throughout
- **Copy-paste**: Applied throughout
- **Close Mosaic Schedule**: Mosaic disabled in final epochs

### Progressive Loss Schedule

| Training Phase | One-to-Many Weight | One-to-One Weight |
|----------------|-------------------|-------------------|
| Early (epochs 1-100) | 0.8 | 0.2 |
| Late (final epochs) | 0.1 | 0.9 |

## Implementation Details

### Label Assignment

**Task-Aligned Learning (TAL):**
- Matches predictions to ground-truth based on IoU
- One-to-many: topk = 10 candidates
- One-to-one: topk = 7, then topk2 = 1 for inference

### Loss Components

**Detection Loss:**
```
L_detection = L_box + L_cls + L_dfl (if applicable)

YOLO26 uses:
  L_box = L1 loss (direct regression)
  L_cls = classification loss (BCE/CIoU)
```

**Instance Segmentation:**
- Prototype-based mask reconstruction
- Multi-scale proto feature fusion
- Auxiliary semantic segmentation loss (BCE + Dice)

**Pose Estimation:**
- RLE loss for uncertainty-aware keypoint localization
- OKS loss for scale normalization
- Sigma branch for per-axis uncertainty prediction

**Oriented Detection:**
- Angle loss: sin²(2Δθ) for square object stability
- Aspect-ratio-aware supervision
- Long-edge angle definition

## Hyperparameter Tuning

The final YOLO26 configuration includes extensive hyperparameter search:
- Optimizer learning rate schedule
- Loss weights
- Augmentation intensities
- Label assignment thresholds

**Result:**
- YOLO26s achieves **48.6 mAP** on COCO
- +0.8 AP from hyperparameter optimization alone

---
*Device: Laptop Linux (LTLN) RTX 500 Ada, Intel Core Ultra 7 255H, 32GB 5600MT/s*