#!/usr/bin/env python3
"""
YOLO26 Benchmark - PyTorch vs TensorRT
======================================
Comprehensive benchmark comparing PyTorch and TensorRT inference speed.

Usage:
    python benchmark.py --model yolo26n.pt --source image.jpg
    python benchmark.py --model yolo26n.pt --source 0  # Camera
    python benchmark.py --model yolo26n.pt --benchmark-only  # Mock benchmark
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import cv2

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

from ultralytics import YOLO


def download_model(model_path: str) -> str:
    """Download model from Ultralytics if not present."""
    model_path = Path(model_path)
    
    if model_path.exists():
        print(f"✓ Model already exists: {model_path}")
        return str(model_path)
    
    print(f"⏳ Downloading model: {model_path.name}")
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        model_path = Path(attempt_download_asset(str(model_path)))
        print(f"✓ Downloaded to: {model_path}")
    except Exception as e:
        print(f"⚠️  Could not download model: {e}")
        print(f"   Please download manually from: https://github.com/ultralytics/assets/releases")
        return None
    
    return str(model_path)


def preprocess_image(frame, imgsz: int = 640):
    """Preprocess image for inference."""
    resized = cv2.resize(frame, (imgsz, imgsz))
    converted = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = converted.astype(np.float32) / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(transposed, axis=0)


def benchmark_pytorch(model, test_images, num_iter: int = 20):
    """Benchmark PyTorch inference."""
    print(f"\n🐍 Benchmarking PyTorch inference...")
    
    # Warmup
    print(f"   🔥 Warmup ({num_iter // 2} iterations)...")
    for _ in range(num_iter // 2):
        model(test_images[0])
    
    # Benchmark
    times = []
    print(f"   ⏱️  Benchmarking ({num_iter // 2} iterations)...")
    for i in range(num_iter // 2):
        start = time.perf_counter()
        model(test_images[0])
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    print(f"   ✓ PyTorch: {avg_time:.2f} ms ({1000/avg_time:.2f} FPS)")
    
    return avg_time


def benchmark_tensorrt(model, imgsz: int = 640, test_images=None):
    """Benchmark TensorRT inference."""
    if not HAS_TENSORRT:
        print("⚠️  TensorRT not installed. Skipping TensorRT benchmark.")
        return None
    
    print(f"\n🔥 Benchmarking TensorRT inference...")
    
    # Export to TensorRT
    engine_path = Path(model.model).with_suffix(".engine")
    print(f"   🔥 Exporting to TensorRT...")
    model.export(format="engine", imgsz=imgsz, device=0)
    print(f"   ✓ Engine saved to: {engine_path}")
    
    # Load engine
    print(f"   🔥 Loading TensorRT engine...")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    # Get bindings
    input_binding = engine.get_binding_name(0)
    output_binding = engine.get_binding_name(1)
    
    input_tensor = engine.get_binding_input(0)
    output_tensor = engine.get_binding_output(0)
    
    # Allocate input buffer
    input_buffer = np.zeros(
        (1, 3, input_tensor.shape[1], input_tensor.shape[2]), dtype=np.float32
    )
    
    # Preprocess function
    def preprocess(frame):
        return preprocess_image(frame, imgsz)
    
    # Warmup
    print(f"   🔥 Warmup ({num_iter // 2} iterations)...")
    for _ in range(num_iter // 2):
        input_buffer[0] = preprocess(test_images[0])
        engine.execute_v2([input_buffer], output_tensor)
    
    # Benchmark
    times = []
    print(f"   ⏱️  Benchmarking ({num_iter // 2} iterations)...")
    for i in range(num_iter // 2):
        start = time.perf_counter()
        input_buffer[0] = preprocess(test_images[0])
        engine.execute_v2([input_buffer], output_tensor)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    print(f"   ✓ TensorRT: {avg_time:.2f} ms ({1000/avg_time:.2f} FPS)")
    
    return avg_time


def run_inference_demo(model, source: str, imgsz: int = 640):
    """Run a quick inference demo."""
    print(f"\n📸 Running inference demo...")
    
    # Load image
    if source == 0:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Cannot open camera")
    else:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Cannot read image: {source}")
    
    print(f"   Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Run inference
    results = model(frame)
    
    # Display results
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            print(f"   Detected {len(boxes)} objects:")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                print(f"      {i+1}. Class {cls_id}, Conf: {conf:.2f}")
        else:
            print(f"   No objects detected")
    else:
        print(f"   No results returned")


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Benchmark: PyTorch vs TensorRT")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Model path")
    parser.add_argument("--source", type=str, default=0, help="Image source (0=camera, file=filepath)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--benchmark-only", action="store_true", help="Run mock benchmark only")
    parser.add_argument("--demo", action="store_true", help="Run inference demo after benchmark")
    
    args = parser.parse_args()
    
    # Download model if needed
    model_path = download_model(args.model)
    if model_path is None:
        print("\n❌ Cannot proceed without model. Please download it manually.")
        print(f"   Visit: https://github.com/ultralytics/assets/releases")
        sys.exit(1)
    
    # Load model
    print(f"\n{'='*60}")
    print(f"YOLO26 Benchmark")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Image Size: {args.imgsz}")
    
    model = YOLO(model_path)
    
    # Load test image
    if args.source == 0:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Cannot open camera")
    else:
        frame = cv2.imread(str(args.source))
        if frame is None:
            raise ValueError(f"Cannot read image: {args.source}")
    
    test_images = [frame] * 20
    
    # Run benchmark
    torch_time = benchmark_pytorch(model, test_images)
    trt_time = benchmark_tensorrt(model, args.imgsz, test_images)
    
    # Calculate speedup
    if trt_time:
        speedup = torch_time / trt_time
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"{'Metric':<20} {'PyTorch':<15} {'TensorRT':<15} {'Speedup':<15}")
        print("-"*60)
        print(f"Inference Time:     {torch_time:<15.2f} ms    {trt_time:<15.2f} ms    {speedup:<15.2f}x")
        print(f"Throughput:         {1000/torch_time:<15.2f} img/s    {1000/trt_time:<15.2f} img/s    {(speedup - 1) * 100:<15.1f}% faster")
        print(f"Speedup:            {' ' * 15}{' ' * 15}{speedup * 100:<15.1f}% faster")
        print("="*60)
    else:
        print("\n⚠️  TensorRT benchmark not available. Only PyTorch results shown.")
    
    # Run inference demo
    if args.demo:
        run_inference_demo(model, args.source, args.imgsz)


if __name__ == "__main__":
    main()
