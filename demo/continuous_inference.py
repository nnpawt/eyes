#!/usr/bin/env python3
"""
YOLO26 Continuous Inference Demo with TensorRT
===============================================
This demo continuously feeds images to YOLO26 and measures inference speed
with both PyTorch and TensorRT backends to show the actual speed gap.

Usage:
    python continuous_inference.py

Requirements:
    - ultralytics
    - tensorrt
    - onnxruntime
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import ops

# Try to import TensorRT
try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    print("⚠️  TensorRT not installed. Using PyTorch only.")


class YOLO26Benchmark:
    """Benchmark YOLO26 inference speed with PyTorch and TensorRT."""

    def __init__(self, model_path: str = "yolo26n.pt", imgsz: int = 640):
        """
        Initialize the benchmark.

        Args:
            model_path: Path to YOLO26 model (.pt file)
            imgsz: Input image size
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.results = []

        # Load PyTorch model
        print(f"📦 Loading YOLO26 model: {model_path}")
        self.model = YOLO(model_path)

        # Export to ONNX
        print(f"🚀 Exporting to ONNX...")
        onnx_path = Path(model_path).with_suffix(".onnx")
        self.model.export(format="onnx", imgsz=imgsz, nms=True)
        print(f"   ✓ ONNX exported to: {onnx_path}")

        # Export to TensorRT if available
        if HAS_TENSORRT:
            print(f"🔥 Exporting to TensorRT...")
            trt_path = Path(model_path).with_suffix(".engine")
            self.model.export(format="engine", imgsz=imgsz, device=0)
            print(f"   ✓ TensorRT engine exported to: {trt_path}")

    def load_tensorrt_engine(self, engine_path: str) -> trt.Runtime:
        """Load TensorRT engine."""
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f, runtime.deserialize_cuda_engine(f.read()) as engine:
            return runtime, engine

    def benchmark_pytorch(self, source: str, num_warmup: int = 5, num_iter: int = 20) -> float:
        """
        Benchmark PyTorch inference.

        Args:
            source: Image source (file path, directory, or camera ID)
            num_warmup: Number of warmup iterations
            num_iter: Number of benchmark iterations

        Returns:
            Average inference time in milliseconds
        """
        print(f"\n🐍 Benchmarking PyTorch inference on: {source}")

        # Load image
        if source == 0:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera")
            frame = self._get_frame(cap)
        else:
            frame = cv2.imread(str(source))
            if frame is None:
                raise ValueError(f"Cannot read image: {source}")

        height, width = frame.shape[:2]
        test_images = [frame] * num_iter

        # Warmup
        print(f"   🔥 Warmup ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            self.model(test_images[0])

        # Benchmark
        times = []
        print(f"   ⏱️  Benchmarking ({num_iter} iterations)...")
        for i in range(num_iter):
            start = time.perf_counter()
            self.model(test_images[0])
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        print(f"   📊 Average: {avg_time:.2f} ms")
        print(f"   📊 Throughput: {1000/avg_time:.2f} images/sec")

        return avg_time

    def benchmark_tensorrt(self, source: str, num_warmup: int = 5, num_iter: int = 20) -> float:
        """
        Benchmark TensorRT inference.

        Args:
            source: Image source (file path, directory, or camera ID)
            num_warmup: Number of warmup iterations
            num_iter: Number of benchmark iterations

        Returns:
            Average inference time in milliseconds
        """
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT not installed")

        print(f"\n🔥 Benchmarking TensorRT inference on: {source}")

        # Load image
        if source == 0:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera")
            frame = self._get_frame(cap)
        else:
            frame = cv2.imread(str(source))
            if frame is None:
                raise ValueError(f"Cannot read image: {source}")

        height, width = frame.shape[:2]
        test_images = [frame] * num_iter

        # Load engine
        engine_path = Path(self.model_path).with_suffix(".engine")
        print(f"   🔥 Loading TensorRT engine: {engine_path}")
        runtime, engine = self.load_tensorrt_engine(str(engine_path))

        # Get I/O bindings
        input_binding = engine.get_binding_name(0)
        output_binding = engine.get_binding_name(1)

        input_tensor = engine.get_binding_input(0)
        output_tensor = engine.get_binding_output(0)

        # Allocate input buffer
        input_shape = list(input_tensor.shape)
        input_buffer = np.zeros(
            (1, 3, input_shape[1], input_shape[2]), dtype=np.float32
        )

        # Warmup
        print(f"   🔥 Warmup ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            input_buffer[0] = self._preprocess(frame)
            engine.execute_v2([input_buffer], output_tensor)

        # Benchmark
        times = []
        print(f"   ⏱️  Benchmarking ({num_iter} iterations)...")
        for i in range(num_iter):
            start = time.perf_counter()
            input_buffer[0] = self._preprocess(frame)
            engine.execute_v2([input_buffer], output_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        print(f"   📊 Average: {avg_time:.2f} ms")
        print(f"   📊 Throughput: {1000/avg_time:.2f} images/sec")

        return avg_time

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference."""
        height, width = self.imgsz, self.imgsz
        resized = cv2.resize(frame, (width, height))
        converted = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = converted.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)

    def _get_frame(self, cap) -> np.ndarray:
        """Get a frame from camera."""
        cap.grab()
        cap.grab()
        ret, frame = cap.read()
        if ret:
            return frame
        cap.release()
        raise RuntimeError("Cannot read frame from camera")

    def run_continuous(self, source: str = 0, interval: float = 0.5, duration: float = 30.0):
        """
        Run continuous inference.

        Args:
            source: Image source (file path, directory, or camera ID)
            interval: Time interval between frames (seconds)
            duration: Total duration to run (seconds)
        """
        print(f"\n🎬 Starting continuous inference...")
        print(f"   Source: {source}")
        print(f"   Interval: {interval}s")
        print(f"   Duration: {duration}s")

        start_time = time.time()
        frame_count = 0
        total_time = 0.0

        if source == 0:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera")
        else:
            cap = None

        while time.time() - start_time < duration:
            if source == 0:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                frame = cv2.imread(str(source))
                if frame is None:
                    break

            # Run inference
            start = time.perf_counter()
            results = self.model(frame)
            end = time.perf_counter()

            # Process results
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        print(f"   Detected: Class {cls_id}, Conf: {conf:.2f}, Box: [{x1},{y1},{x2},{y2}]")

            inference_time = (end - start) * 1000
            total_time += inference_time
            frame_count += 1

            # Print stats
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            avg_inference = total_time / frame_count
            print(f"   [{elapsed:.1f}s] Frame: {frame_count}, FPS: {fps:.1f}, Avg: {avg_inference:.2f}ms")

            time.sleep(interval)

        if source == 0:
            cap.release()

        print(f"\n✅ Continuous inference complete!")
        print(f"   Total frames: {frame_count}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   FPS: {frame_count / elapsed:.2f}")
        print(f"   Avg inference: {avg_inference:.2f}ms")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="YOLO26 Continuous Inference Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="YOLO26 model path (default: yolo26n.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=0,
        help="Image source: file path, directory, or camera ID (default: 0 for camera)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration to run continuous inference (default: 10s)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode",
    )

    args = parser.parse_args()

    # Create benchmark
    benchmark = YOLO26Benchmark(args.model, args.imgsz)

    if args.benchmark:
        print("=" * 60)
        print("YOLO26 BENCHMARK MODE")
        print("=" * 60)

        # Benchmark PyTorch
        pytorch_time = benchmark.benchmark_pytorch(args.source)

        # Benchmark TensorRT if available
        if HAS_TENSORRT:
            trt_time = benchmark.benchmark_tensorrt(args.source)
            speedup = pytorch_time / trt_time
            print("\n" + "=" * 60)
            print("BENCHMARK RESULTS")
            print("=" * 60)
            print(f"PyTorch:        {pytorch_time:.2f} ms ({1000/pytorch_time:.2f} img/s)")
            print(f"TensorRT:       {trt_time:.2f} ms ({1000/trt_time:.2f} img/s)")
            print(f"Speedup:        {speedup:.2f}x ({(speedup - 1) * 100:.1f}% faster)")
        else:
            print("\n⚠️  TensorRT not available for comparison")

    else:
        # Continuous inference mode
        benchmark.run_continuous(args.source, interval=0.5, duration=args.duration)


if __name__ == "__main__":
    main()
