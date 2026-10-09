# Ultralytics YOLO26 Usage Examples

## Installation

### Using UV (Recommended)

```bash
# Create virtual environment with UV
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install ultralytics package with UV
uv add ultralytics

# Or with specific version
uv add ultralytics==8.3.0

# Install with dev dependencies
uv add ultralytics[dev]
```

### Alternative: pip (Legacy)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install ultralytics package
pip install ultralytics

# Or with specific version
pip install ultralytics==8.3.0

# Install with dev dependencies
pip install ultralytics[dev]
```

## Basic Inference

### Python API

```python
from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Run inference on an image
results = model("path/to/bus.jpg")

# Print results
results[0].show()
results[0].save("output.jpg")
```

### CLI

```bash
# Run inference
yolo predict model=yolo26n.pt source=path/to/bus.jpg
```

## Supported Tasks

### Object Detection

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Predict
results = model.predict(
    source="image.jpg",
    conf=0.25,
    iou=0.45,
    nms=False  # Use end-to-end head
)

# Show results
results[0].show()
```

### Instance Segmentation

```python
model = YOLO("yolo26n-seg.pt")

results = model.predict(
    source="image.jpg",
    conf=0.25,
    iou=0.45
)

# Access masks
for instance in results[0].masks:
    instance.show()
```

### Pose Estimation

```python
model = YOLO("yolo26n-pose.pt")

results = model.predict(
    source="person.jpg",
    conf=0.25
)

# Access keypoints
for instance in results[0].boxes:
    print(instance.keypoints)
```

### Oriented Detection (OBB)

```python
model = YOLO("yolo26n-obb.pt")

results = model.predict(
    source="dota.jpg",
    conf=0.25,
    iou=0.45
)

# Access oriented boxes
for box in results[0].boxes:
    print(box.xyxy, box.angles)
```

### Classification

```python
model = YOLO("yolo26n-cls.pt")

results = model.predict(
    source="image.jpg"
)

# Get predicted class
pred_class = results[0].cls[0]
print(f"Predicted class: {pred_class}")
```

### Depth Estimation

```python
model = YOLO("yolo26n-depth.pt")

results = model.predict(
    source="scene.jpg"
)

# Get depth map
depth = results[0].depth
depth.show()
```

## Training

### Basic Training

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo26n.pt")

# Train on COCO8 example dataset
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

### Training with Custom Dataset

```python
results = model.train(
    data="path/to/my_dataset.yaml",
    epochs=300,
    imgsz=640,
    batch=32,
    optimizer="sgd",  # or "musegd"
    name="custom_training"
)
```

### CLI Training

```bash
# Basic training
yolo train model=yolo26n.pt data=coco8.yaml epochs=100 imgsz=640

# Training with custom config
yolo train model=yolo26n.pt data=dataset.yaml epochs=300 \
    batch=32 workers=8 device=0:1 imgsz=640 \
    optimizer=sgd name=my_model
```

## Validation

```python
model = YOLO("yolo26n.pt")

# Run validation on dataset
metrics = model.val(
    data="coco.yaml",
    split="val",
    imgsz=640,
    nms=False  # Use end-to-end head
)

print(f"mAP50-95: {metrics.box.map:.2f}")
```

## Export

```python
model = YOLO("yolo26n.pt")

# Export to ONNX
model.export(format="onnx", imgsz=640)

# Export to TensorRT
model.export(format="engine", imgsz=640, device=0)

# Export to TorchScript
model.export(format="torchscript", imgsz=640)

# Export with NMS-free end-to-end head
model.export(format="onnx", imgsz=640, nms=False)
```

### CLI Export

```bash
# Export to ONNX
yolo export model=yolo26n.pt format=onnx imgsz=640

# Export to TensorRT
yolo export model=yolo26n.pt format=engine imgsz=640 device=0

# Export with end-to-end head
yolo export model=yolo26n.pt format=onnx imgsz=640 nms=False
```

## Advanced Usage

### Custom Confidence and IoU Thresholds

```python
results = model.predict(
    source="image.jpg",
    conf=0.5,  # Confidence threshold
    iou=0.7    # IoU threshold for NMS
)
```

### Batch Inference

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Multiple images
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = model.predict(source=images, batch=32)

# From directory
results = model.predict(source="path/to/images/", batch=32)
```

### Video Inference

```python
model = YOLO("yolo26n.pt")

# From file
results = model.predict(source="video.mp4", stream=True)

# From camera
results = model.predict(source=0, stream=True)  # Camera ID 0

# Save results
results.save("output.mp4")
```

### Custom Augmentations

```python
results = model.train(
    data="dataset.yaml",
    epochs=100,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.2,
    scale=0.5,
    shear=0.1,
    perspective=0.001
)
```

### Using Pretrained Weights

```python
# COCO pretrained
model = YOLO("yolo26n.pt")

# Custom pretrained
model = YOLO("path/to/custom.pt")

# Architecture only (train from scratch)
model = YOLO("yolo26n.yaml")
```

## End-to-End Inference (NMS-Free)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Use one-to-one head (NMS-free)
results = model.predict(
    source="image.jpg",
    nms=False  # Critical flag for end-to-end mode
)

# Export with NMS-free head
model.export(format="onnx", nms=False)
```

## Performance Tips

### For Edge Devices

```python
# Use smaller model
model = YOLO("yolo26n.pt")

# Lower resolution
results = model.predict(imgsz=416)

# Use TensorRT if available
results = model.predict(device=0)  # GPU
```

### For High Accuracy

```python
# Use larger model
model = YOLO("yolo26x.pt")

# Higher resolution
results = model.predict(imgsz=1280)

# Use NMS for maximum accuracy
results = model.predict(nms=True)
```

### For Fastest Inference

```python
# Nano model + ONNX
model = YOLO("yolo26n.pt")
model.export(format="onnx")
results = model.predict(device="cpu")

# Or use TensorRT
model.export(format="engine", device=0)
results = model.predict(device=0)
```

## Visualization

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")
results = model.predict(source="image.jpg")

# Access detection results
boxes = results[0].boxes
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"Class: {cls_id}, Conf: {conf:.2f}, Box: [{x1},{y1},{x2},{y2}]")
```

## Citation

```bibtex
@misc{jocher2026ultralyticsyolo26unifiedrealtime,
  title = {Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
  author = {Glenn Jocher and Jing Qiu and Mengyu Liu and Shuai Lyu and Fatih Cagatay Akyon and Muhammet Esat Kalfaoglu},
  year = {2026},
  eprint = {2606.03748},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```

---
*Device: Laptop Linux (LTLN) RTX 500 Ada, Intel Core Ultra 7 255H, 32GB 5600MT/s*