# YOLO26 Knowledge Base

A comprehensive knowledge base for **Ultralytics YOLO26** - the unified family of real-time vision models.

## Quick Links

- [**Overview**](./01-overview.md) - Model family, quickstart, key features
- [**Architecture**](./02-architecture.md) - Dual-head design, DFL-free regression, task-specific extensions
- [**Performance**](./03-performance.md) - Benchmarks on COCO, Cityscapes, NYU Depth V2, ImageNet, DOTA
- [**Training**](./04-training.md) - MuSGD optimizer, Progressive Loss, STAL, recipe details
- [**Usage**](./05-usage.md) - Python/CLI examples for all tasks
- [**Deployment**](./06-deployment.md) - Export targets, hardware profiles, optimization tips
- [**Device Profile**](./07-device-profile.md) - Your hardware specs (RTX 500 Ada, Core Ultra 7 255H)

## Installation

```bash
# Using UV (recommended)
uv venv
source .venv/bin/activate
uv add ultralytics

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install ultralytics
```

## Quick Start

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo26n.pt")

# Inference
results = model("image.jpg")

# Train
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

## Key Features

- **NMS-Free Inference**: Native end-to-end detection without post-processing
- **DFL-Free**: Lighter head with unconstrained regression range
- **5 Model Scales**: n/s/m/l/x for different accuracy/latency needs
- **7 Tasks**: Detection, segmentation, semantic segmentation, depth, classification, pose, OBB
- **Open-Vocabulary**: YOLOE-26 for text/visual/prompt-free inference

## Performance Highlights

| Metric | Value |
|--------|-------|
| COCO mAP (YOLO26n) | 40.9 |
| COCO mAP (YOLO26x) | 57.5 |
| T4 TensorRT Latency (n) | 1.7ms |
| CPU ONNX Speedup vs YOLO11n | 43% faster |

## Device Profile

**Your Hardware:**
- **GPU**: NVIDIA RTX 500 Ada (8GB VRAM)
- **CPU**: Intel Core Ultra 7 255H (16 cores/threads)
- **RAM**: 32GB DDR5 5600MT/s

**Recommended Models:**
- YOLO26n/26s: Best for edge, <2ms on GPU
- YOLO26m: Balanced performance
- YOLO26l/26x: Maximum accuracy

## References

- **Paper**: https://arxiv.org/abs/2606.03748
- **GitHub**: https://github.com/ultralytics/ultralytics
- **Docs**: https://docs.ultralytics.com/models/yolo26
- **Platform**: https://platform.ultralytics.com/ultralytics/yolo26

---
*Knowledge Base for: Laptop Linux (LTLN) RTX 500 Ada, Intel Core Ultra 7 255H, 32GB 5600MT/s*