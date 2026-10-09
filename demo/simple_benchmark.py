#!/usr/bin/env python3
"""
Simple YOLO26 Benchmark - Image Only
=====================================
Quick benchmark comparing PyTorch vs TensorRT.

Usage:
    python simple_benchmark.py --model yolo26n.pt --source image.jpg
"""

import time
import sys
from pathlib import Path

from ultralytics import YOLO


def benchmark_model(model_path: str, source: str, imgsz: int = 640):
    """
    Benchmark YOLO26 model.

    Args:
        model_path: Path to model (.pt file)
        source: Image source (file path)
        imgsz: Input image size
    """
    print(f"\n{'=' * 60}")
    print(f"YOLO26 Benchmark - {Path(model_path).name}")
    print(f"{'=' * 60}")

    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)

    # Load image
    print(f"🖼️  Loading image: {source}")
    frame = model.source(source)
    if frame is None:
        raise ValueError(f"Cannot read image: {source}")

    height, width = frame.shape[:2]
    test_images = [frame] * 20

    # Warmup
    print(f"🔥 Warmup (5 iterations)...")
    for _ in range(5):
        model(test_images[0])

    # Benchmark PyTorch
    print(f"\n🐍 Benchmarking PyTorch inference...")
    torch_times = []
    for i in range(20):
        start = time.perf_counter()
        model(test_images[0])
        torch_times.append((time.perf_counter() - start) * 1000)

    torch_avg = sum(torch_times) / len(torch_times)
    torch_fps = 1000 / torch_avg

    print(f"   ✓ PyTorch: {torch_avg:.2f} ms ({torch_fps:.2f} FPS)")

    # Export and benchmark TensorRT
    print(f"\n🔥 Exporting to TensorRT...")
    engine_path = Path(model_path).with_suffix(".engine")
    model.export(format="engine", imgsz=imgsz, device=0)
    print(f"   ✓ Engine saved to: {engine_path}")

    # Load engine
    print(f"🔥 Loading TensorRT engine...")
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    input_binding = engine.get_binding_name(0)
    output_binding = engine.get_binding_name(1)

    input_tensor = engine.get_binding_input(0)
    output_tensor = engine.get_binding_output(0)

    input_buffer = None
    for i in range(input_tensor.num_bindings):
        if engine.binding_is_input(i):
            input_buffer = engine.get_binding_input(i)
            break

    # Preprocess function
    def preprocess(frame):
        resized = cv2.resize(frame, (imgsz, imgsz))
        converted = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = converted.astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)

    # Warmup
    print(f"🔥 TensorRT Warmup (5 iterations)...")
    for _ in range(5):
        input_buffer[0] = preprocess(frame)
        engine.execute_v2([input_buffer], output_tensor)

    # Benchmark TensorRT
    print(f"🔥 Benchmarking TensorRT inference...")
    trt_times = []
    for i in range(20):
        start = time.perf_counter()
        input_buffer[0] = preprocess(frame)
        engine.execute_v2([input_buffer], output_tensor)
        trt_times.append((time.perf_counter() - start) * 1000)

    trt_avg = sum(trt_times) / len(trt_times)
    trt_fps = 1000 / trt_avg
    speedup = torch_avg / trt_avg

    # Results
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'PyTorch':<15} {'TensorRT':<15} {'Speedup':<15}")
    print("-" * 60)
    print(f"Inference Time:     {torch_avg:<15.2f} ms    {trt_avg:<15.2f} ms    {speedup:<15.2f}x")
    print(f"Throughput:         {torch_fps:<15.2f} img/s    {trt_fps:<15.2f} img/s    {(speedup - 1) * 100:<15.1f}% faster")
    print(f"Speedup:            {' ' * 15}{' ' * 15}{speedup * 100:<15.1f}% faster")
    print("=" * 60)

    return torch_avg, trt_avg, speedup


if __name__ == "__main__":
    import argparse
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser(description="YOLO26 Simple Benchmark")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Model path")
    parser.add_argument("--source", type=str, default="image.jpg", help="Image source")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")

    args = parser.parse_args()

    benchmark_model(args.model, args.source, args.imgsz)
