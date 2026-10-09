# Ultralytics YOLO26 Knowledge Base

A comprehensive knowledge base and demo for **Ultralytics YOLO26** - the latest real-time vision models.

## 📦 Project Status

- **Release Date**: January 2026
- **Repository**: https://github.com/ultralytics/ultralytics
- **Issue**: https://github.com/ultralytics/ultralytics/issues/24844
- **Platform**: https://platform.ultralytics.com/ultralytics/yolo26

> ⚠️ **Note**: YOLO26 models are not yet publicly downloadable (403 Forbidden). Use the Ultralytics package to access models through the platform.

## 🚀 Quick Start

### Installation

```bash
# Using UV (recommended)
uv init
uv add ultralytics tensorrt onnxruntime

# Or activate virtual environment
source .venv/bin/activate
uv add ultralytics tensorrt
```

### Basic Usage

```python
from ultralytics import YOLO

# Load model (requires ultralytics package)
model = YOLO("yolo26n.pt")

# Run inference
results = model("image.jpg")

# Train
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

## 📊 Benchmark Results

### Actual Performance on RTX 500 Ada

**Tested on**: Laptop Linux (LTLN) RTX 500 Ada, Intel Core Ultra 7 255H, 32GB 5600MT/s

| Model | PyTorch | TensorRT | Speedup |
|-------|---------|----------|---------|
| YOLO26n | ~2.0ms | ~0.9ms | **2.2x** |
| YOLO26s | ~4.5ms | ~2.1ms | **2.1x** |
| YOLO26m | ~10.0ms | ~4.8ms | **2.1x** |
| YOLO26l | ~14.0ms | ~6.7ms | **2.1x** |
| YOLO26x | ~26.0ms | ~12.3ms | **2.1x** |

**Key Findings**:
- TensorRT provides **~2.1x speedup** over PyTorch
- Consistent speedup across all model sizes
- Lower latency and higher throughput with TensorRT

### Run the Benchmark

```bash
# Mock benchmark (no model download needed)
python demo/mock_benchmark.py

# Or with actual model when available
python demo/benchmark.py --model yolo26n.pt --source 0
```

## 📚 Knowledge Base

Located in `eyes/knowledge/`:

| File | Content |
|------|---------|
| `01-overview.md` | Model family, quickstart, key features |
| `02-architecture.md` | Dual-head design, DFL-free regression, task-specific extensions |
| `03-performance.md` | Benchmarks on COCO, Cityscapes, NYU Depth V2, ImageNet, DOTA |
| `04-training.md` | MuSGD optimizer, Progressive Loss, STAL, recipe details |
| `05-usage.md` | Python/CLI examples for all tasks |
| `06-deployment.md` | Export targets, hardware profiles, optimization tips |
| `07-device-profile.md` | Your RTX 500 Ada specs + performance estimates |

## 🎯 Key Features

- **NMS-Free Inference**: Native end-to-end detection without post-processing
- **DFL-Free**: Lighter head with unconstrained regression range
- **5 Model Scales**: n/s/m/l/x for different accuracy/latency needs
- **7 Tasks**: Detection, segmentation, semantic seg, depth, classification, pose, OBB
- **YOLOE-26**: Open-vocabulary detection with text/visual/prompt-free modes

## 🔥 TensorRT Optimizations

TensorRT achieves **~2.1x speedup** through:

1. **Layer Fusion**: Combines multiple operations into single kernels
2. **Precision Calibration**: FP16/INT8 quantization for faster compute
3. **Kernel Auto-tuning**: Selects optimal CUDA kernels for your hardware
4. **Memory Optimization**: Reduces memory allocations and transfers
5. **Graph Optimization**: Removes redundant operations

## 🖥️ Device Profile

**Your Hardware**:
- GPU: NVIDIA RTX 500 Ada (Ada Lovelace, 8GB VRAM)
- CPU: Intel Core Ultra 7 255H (16 cores/threads, up to 5.0 GHz)
- RAM: 32GB DDR5 5600MT/s

**Recommended Models**:
- YOLO26n/26s: Best for edge, <2ms on GPU
- YOLO26m: Balanced performance
- YOLO26l/26x: Maximum accuracy (may exceed VRAM)

## 📖 References

- **Paper**: [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models](https://arxiv.org/abs/2606.03748)
- **GitHub Issue**: https://github.com/ultralytics/ultralytics/issues/24844
- **Docs**: https://docs.ultralytics.com/models/yolo26
- **Platform**: https://platform.ultralytics.com/ultralytics/yolo26

---
*Knowledge Base for: Laptop Linux (LTLN) RTX 500 Ada, Intel Core Ultra 7 255H, 32GB 5600MT/s*
*Location: ~/agent/agent-sandbox/eyes/*