# Ultralytics YOLO26 Overview

## Summary

**Ultralytics YOLO26** is a unified family of real-time vision models that redefines state-of-the-art computer vision AI. Published June 2026, it introduces native end-to-end inference, a lighter detection head, an updated training recipe, and task-specific heads for detection, segmentation, pose estimation, classification, and oriented detection.

### Key Highlights

- **Unified Framework**: Single pipeline supporting detection, instance segmentation, semantic segmentation, depth estimation, classification, pose estimation, and oriented detection
- **NMS-Free Inference**: Native end-to-end inference without non-maximum suppression post-processing
- **DFL-Free Design**: Removes Distribution Focal Loss for simpler, faster inference
- **State-of-the-Art Performance**: 40.9-57.5 mAP on COCO at 1.7-11.8 ms T4 TensorRT latency
- **5 Model Scales**: n/s/m/l/x variants for different accuracy/latency tradeoffs
- **Open-Vocabulary Extension**: YOLOE-26 for text/visual/prompt-free inference

## Model Family

| Model | Detection | Segmentation | Semantic Seg | Depth | Classification | Pose | OBB |
|-------|-----------|--------------|--------------|-------|----------------|------|-----|
| yolo26n.pt | ✅ | yolo26n-seg.pt | yolo26n-sem.pt | yolo26n-depth.pt | yolo26n-cls.pt | yolo26n-pose.pt | yolo26n-obb.pt |
| yolo26s.pt | ✅ | yolo26s-seg.pt | yolo26s-sem.pt | yolo26s-depth.pt | yolo26s-cls.pt | yolo26s-pose.pt | yolo26s-obb.pt |
| yolo26m.pt | ✅ | yolo26m-seg.pt | yolo26m-sem.pt | yolo26m-depth.pt | yolo26m-cls.pt | yolo26m-pose.pt | yolo26m-obb.pt |
| yolo26l.pt | ✅ | yolo26l-seg.pt | yolo26l-sem.pt | yolo26l-depth.pt | yolo26l-cls.pt | yolo26l-pose.pt | yolo26l-obb.pt |
| yolo26x.pt | ✅ | yolo26x-seg.pt | yolo26x-sem.pt | yolo26x-depth.pt | yolo26x-cls.pt | yolo26x-pose.pt | yolo26x-obb.pt |

## Quickstart

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo26n.pt")

# Run inference
results = model("image.jpg")

# Train on custom dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

## Reference

- **Paper**: [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models](https://arxiv.org/abs/2606.03748)
- **GitHub**: https://github.com/ultralytics/ultralytics
- **Docs**: https://docs.ultralytics.com/models/yolo26
- **Platform**: https://platform.ultralytics.com/ultralytics/yolo26

---
*Device: Laptop Linux (LTLN) RTX 500 Ada, Intel Core Ultra 7 255H, 32GB 5600MT/s*