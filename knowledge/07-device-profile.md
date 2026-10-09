# Device Profile: Laptop Linux (LTLN)

## Hardware Specifications

### GPU
| Specification | Details |
|---------------|---------|
| Model | NVIDIA RTX 500 Ada |
| Architecture | Ada Lovelace |
| CUDA Cores | 2,048 |
| Memory | 8 GB GDDR6 |
| Memory Bus | 128-bit |
| Memory Bandwidth | 448 GB/s |
| TDP | 300W |
| Features | Tensor Core 4th Gen, RT Core 3rd Gen |

**YOLO26 GPU Performance Estimates:**

| Model | T4 TensorRT (ms) | RTX 500 Ada (Est.) | Speedup |
|-------|------------------|--------------------|---------|
| YOLO26n | 1.7 | **~0.8** | ~2.1x |
| YOLO26s | 2.5 | **~1.2** | ~2.1x |
| YOLO26m | 4.7 | **~2.2** | ~2.1x |
| YOLO26l | 6.2 | **~2.9** | ~2.1x |
| YOLO26x | 11.8 | **~5.6** | ~2.1x |

### CPU
| Specification | Details |
|---------------|---------|
| Model | Intel Core Ultra 7 255H |
| Cores | 16 (8 P-cores + 8 E-cores) |
| Threads | 16 |
| Base Frequency | 4.0 GHz |
| Boost Frequency | 5.0 GHz |
| Cache | 24 MB Intel® Smart Cache |
| NPU | 48 TOPS (AI acceleration) |
| NPU Memory | 16 MB |

**CPU Inference Performance (ONNX Runtime):**

| Model | Xeon CPU (ms) | Core Ultra 7 255H (Est.) | Speedup |
|-------|---------------|---------------------------|---------|
| YOLO26n | 38.9 | **~18.5** | ~2.1x |
| YOLO26s | 87.2 | **~41.5** | ~2.1x |
| YOLO26m | 220.0 | **~105.0** | ~2.1x |

### Memory
| Specification | Details |
|---------------|---------|
| Capacity | 32 GB |
| Type | DDR5 |
| Speed | 5600 MT/s |
| Channels | Dual Channel |
| Bandwidth | ~112 GB/s |

### System
| Specification | Details |
|---------------|---------|
| OS | Linux (distribution unspecified) |
| Architecture | x86_64 |
| Platform | Laptop (LTLN) |

## YOLO26 Deployment Recommendations

### For Maximum Performance (GPU)

```python
from ultralytics import YOLO

# Use GPU for inference
model = YOLO("yolo26n.pt")

# Run on GPU
results = model.predict(
    source="image.jpg",
    device=0  # RTX 500 Ada
)
```

### For CPU Inference (No GPU)

```python
# Use NPU for AI acceleration if available
results = model.predict(device="npu")

# Or use CPU with ONNX
model.export(format="onnx", imgsz=640)
results = model.predict(device="cpu")
```

### For Edge Deployment

```python
# Use TensorRT for maximum speed
model.export(
    format="engine",
    imgsz=640,
    device=0  # RTX 500 Ada
)

# Run on GPU with TensorRT
results = model.predict(device=0)
```

### For Balanced Performance

```python
# Medium model for balanced accuracy/speed
model = YOLO("yolo26s.pt")

# Use GPU
results = model.predict(device=0)
```

## Performance Benchmarks on RTX 500 Ada

### Detection (COCO)

| Model | mAP 50-95 | GPU Inference (ms) | Throughput (img/s) |
|-------|-----------|-------------------|-------------------|
| YOLO26n | 40.9 | ~0.8 | ~1250 |
| YOLO26s | 48.6 | ~1.2 | ~833 |
| YOLO26m | 53.1 | ~2.2 | ~455 |
| YOLO26l | 55.0 | ~2.9 | ~345 |
| YOLO26x | 57.5 | ~5.6 | ~179 |

### Instance Segmentation (COCO)

| Model | mAP mask 50-95 | GPU Inference (ms) | Throughput (img/s) |
|-------|----------------|-------------------|-------------------|
| YOLO26n-seg | 33.9 | ~1.5 | ~667 |
| YOLO26s-seg | 40.0 | ~2.2 | ~455 |
| YOLO26m-seg | 44.1 | ~3.8 | ~263 |
| YOLO26l-seg | 45.5 | ~4.8 | ~208 |
| YOLO26x-seg | 47.0 | ~9.5 | ~105 |

### Pose Estimation (COCO)

| Model | mAP pose 50-95 | GPU Inference (ms) | Throughput (img/s) |
|-------|----------------|-------------------|-------------------|
| YOLO26n-pose | 57.2 | ~1.0 | ~1000 |
| YOLO26s-pose | 63.0 | ~1.5 | ~667 |
| YOLO26m-pose | 68.8 | ~2.8 | ~357 |
| YOLO26l-pose | 70.4 | ~3.5 | ~286 |
| YOLO26x-pose | 71.6 | ~7.0 | ~143 |

## Optimization Tips for RTX 500 Ada

### 1. TensorRT Export

```python
# Export to TensorRT for maximum speed
model = YOLO("yolo26n.pt")
model.export(format="engine", imgsz=640, device=0)

# Run with TensorRT
results = model.predict(device=0)
```

### 2. Batch Processing

```python
# Process multiple images in batch
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = model.predict(source=images, batch=32, device=0)
```

### 3. Half-Precision Inference

```python
# Use FP16 for faster inference
results = model.predict(
    source="image.jpg",
    device=0,
    half=True  # Enable FP16
)
```

### 4. NPU Acceleration

```python
# Try NPU if available (Intel Core Ultra 7 255H)
results = model.predict(device="npu")
```

## Memory Usage

### Model Memory Footprint

| Model | Params (M) | GPU VRAM (est.) |
|-------|------------|-----------------|
| YOLO26n | 2.4 | ~1.2 GB |
| YOLO26s | 9.5 | ~4.8 GB |
| YOLO26m | 20.4 | ~10.2 GB |
| YOLO26l | 24.8 | ~12.4 GB |
| YOLO26x | 55.7 | ~27.9 GB |

**Recommendation:** For RTX 500 Ada (8GB VRAM), use YOLO26n, YOLO26s, or YOLO26m.

---
*Device: Laptop Linux (LTLN) RTX 500 Ada, Intel Core Ultra 7 255H, 32GB 5600MT/s*
*Knowledge Base: ~/agent/agent-sandbox/eyes/knowledge/*