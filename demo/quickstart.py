#!/usr/bin/env python3
"""
YOLO26 QuickStart Demo
======================
Simple demo to get started with YOLO26 inference.

Usage:
    python quickstart.py --model yolo26n.pt --source image.jpg
    python quickstart.py --model yolo26n.pt --source 0  # Camera
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO26 QuickStart Demo")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Model path")
    parser.add_argument("--source", type=str, default=0, help="Image source (0=camera, file=filepath)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("YOLO26 QuickStart Demo")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Image Size: {args.imgsz}")
    print("=" * 60)
    
    # Download/load model
    print(f"\n📦 Loading model: {args.model}")
    try:
        model = YOLO(args.model)
        print(f"✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPlease ensure you have:")
        print("  1. Installed ultralytics: uv add ultralytics")
        print("  2. Model is available or can be downloaded")
        return
    
    # Load image
    print(f"\n🖼️  Loading image: {args.source}")
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
    
    print(f"   Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Run inference
    print(f"\n🔍 Running inference...")
    results = model(frame)
    
    # Display results
    print(f"\n📊 Results:")
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            print(f"   Detected {len(boxes)} objects:")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = f"Class {cls_id}"
                print(f"      {i+1}. {cls_name}: {conf:.2%}")
                print(f"         Box: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
        else:
            print(f"   No objects detected")
    else:
        print(f"   No results returned")
    
    # Save results
    if results and len(results) > 0:
        results[0].save("output.jpg")
        print(f"\n✓ Results saved to: output.jpg")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Try different models: yolo26s.pt, yolo26m.pt, etc.")
    print("  • Run benchmark: python benchmark.py")
    print("  • Read knowledge base: knowledge/01-overview.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
