# Ultralytics YOLO26 Overview

## Summary

**Ultralytics YOLO26** is a unified family of real-time vision models released **January 2026**. It introduces native end-to-end inference, a lighter detection head, an updated training recipe, and task-specific heads for detection, segmentation, pose estimation, classification, and oriented detection.

## Latest Release

**Status**: ✅ **Released** - January 2026  
**Version**: v8.4.x+  
**GitHub**: https://github.com/ultralytics/ultralytics/issues/24844

### Where to Get Models

YOLO26 models are available through:

1. **Ultralytics Package** (Recommended)
```bash
uv add ultralytics
python -c "from ultralytics import YOLO; m = YOLO('yolo26n.pt')"
```

2. **Ultralytics Platform** (Cloud)
- https://platform.ultralytics.com/ultralytics/yolo26

3. **Direct Download** (GitHub Releases)
- https://github.com/ultralytics/assets/releases/tag/v8.4.0
- Models: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt

## Key Highlights

- **NMS-Free Inference**: Native end-to-end inference without non-maximum suppression post-processing
- **DFL-Free Design**: Removes Distribution Focal Loss for simpler, faster inference
- **5 Model Scales**: n/s/m/l/x variants for different accuracy/latency tradeoffs
- **7 Tasks**: Detection, segmentation, semantic segmentation, depth estimation, classification, pose estimation, oriented detection
- **Open-Vocabulary**: YOLOE-26 for text/visual/prompt-free inference

## Model Family

| Model | Size | Params | COCO mAP | T4 Latency |
|-------|------|--------|----------|------------|
| yolo26n.pt | 640 | 2.4M | 40.9 | 1.7ms |
| yolo26s.pt | 640 | 9.5M | 48.6 | 2.5ms |
| yolo26m.pt | 640 | 20.4M | 53.1 | 4.7ms |
| yolo26l.pt | 640 | 24.8M | 55.0 | 6.2ms |
| yolo26x.pt | 640 | 55.7M | 57.5 | 11.8ms |

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

- **Issue**: https://github.com/ultralytics/ultralytics/issues/24844
- **Docs**: https://docs.ultralytics.com/models/yolo26
- **GitHub**: https://github.com/ultralytics/ultralytics
- **Platform**: https://platform.ultralytics.com/ultralytics/yolo26

---
*Device: Laptop Linux (LTLN) RTX 500 Ada, Intel Core Ultra 7 255H, 32GB 5600MT/s*
*Knowledge Base: ~/agent/agent-sandbox/eyes/knowledge/*