"""
SAM3 Video Tracking Benchmark with TensorRT Comparison

Comprehensive benchmarking comparing PyTorch vs TensorRT performance.

Usage:
    python benchmarks/video_benchmark_trt.py --video test.mp4 --prompt "person"

Author: Claude Code
Date: 2025-12-26
Phase: 1-B (TensorRT Integration)
"""

import argparse
import sys
from pathlib import Path
import time
import json
import numpy as np
import cv2
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3_video import Sam3VideoSessionManager, create_session_manager, get_version_info


class TensorRTComparativeBenchmark:
    """Comparative benchmark between PyTorch and TensorRT."""

    def __init__(self, video_path: str, prompt: str, device: str = None):
        self.video_path = video_path
        self.prompt = prompt
        self.device = device
        self.results = {}

    def benchmark_pytorch(self, num_frames: int = 100) -> Dict:
        """Benchmark PyTorch inference."""

        print(f"\n{'='*70}")
        print("Benchmarking PyTorch (Baseline)")
        print(f"{'='*70}\n")

        session = Sam3VideoSessionManager(device=self.device)
        session.start_session(text_prompt=self.prompt)

        cap = cv2.VideoCapture(self.video_path)
        frame_times = []
        frame_count = 0

        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()
            masks, detections = session.process_frame(frame)
            end = time.perf_counter()

            frame_times.append(end - start)
            frame_count += 1

            if frame_count % 10 == 0:
                print(f"  Processed {frame_count}/{num_frames} frames...")

        cap.release()
        session.end_session()

        return {
            'mode': 'pytorch',
            'frames': frame_count,
            'frame_times': frame_times,
            'avg_time': np.mean(frame_times),
            'std_time': np.std(frame_times),
            'avg_fps': 1.0 / np.mean(frame_times),
        }

    def benchmark_tensorrt(self, num_frames: int = 100) -> Dict:
        """Benchmark TensorRT inference."""

        print(f"\n{'='*70}")
        print("Benchmarking TensorRT (Accelerated)")
        print(f"{'='*70}\n")

        session = create_session_manager(use_tensorrt=True, device=self.device)

        # Check if TensorRT is actually being used
        if hasattr(session, 'get_inference_info'):
            info = session.get_inference_info()
            if info['mode'] != 'tensorrt':
                print("⚠ TensorRT not available, skipping benchmark")
                return None

        session.start_session(text_prompt=self.prompt)

        cap = cv2.VideoCapture(self.video_path)
        frame_times = []
        frame_count = 0

        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()
            masks, detections = session.process_frame(frame)
            end = time.perf_counter()

            frame_times.append(end - start)
            frame_count += 1

            if frame_count % 10 == 0:
                print(f"  Processed {frame_count}/{num_frames} frames...")

        cap.release()
        session.end_session()

        return {
            'mode': 'tensorrt',
            'frames': frame_count,
            'frame_times': frame_times,
            'avg_time': np.mean(frame_times),
            'std_time': np.std(frame_times),
            'avg_fps': 1.0 / np.mean(frame_times),
        }

    def run_comparison(self, num_frames: int = 100) -> Dict:
        """Run comparative benchmark."""

        pytorch_results = self.benchmark_pytorch(num_frames)
        tensorrt_results = self.benchmark_tensorrt(num_frames)

        return {
            'pytorch': pytorch_results,
            'tensorrt': tensorrt_results,
        }

    def print_comparison_report(self, results: Dict):
        """Print comparative report."""

        pytorch = results['pytorch']
        tensorrt = results.get('tensorrt')

        print(f"\n{'='*70}")
        print("Benchmark Comparison Report")
        print(f"{'='*70}\n")

        print(f"Video: {self.video_path}")
        print(f"Prompt: '{self.prompt}'")
        print(f"Frames tested: {pytorch['frames']}")

        print(f"\n{'Mode':<15} {'FPS':<12} {'Latency (ms)':<15} {'Speedup':<10}")
        print(f"{'-'*70}")

        pytorch_fps = pytorch['avg_fps']
        pytorch_latency = pytorch['avg_time'] * 1000

        print(f"{'PyTorch':<15} {pytorch_fps:<12.2f} {pytorch_latency:<15.2f} {'1.00x':<10}")

        if tensorrt:
            tensorrt_fps = tensorrt['avg_fps']
            tensorrt_latency = tensorrt['avg_time'] * 1000
            speedup = tensorrt_fps / pytorch_fps

            print(f"{'TensorRT':<15} {tensorrt_fps:<12.2f} {tensorrt_latency:<15.2f} {speedup:.2f}x")

            print(f"\n{'='*70}")
            print(f"TensorRT Improvement")
            print(f"{'='*70}")
            print(f"  Speedup: {speedup:.2f}x faster")
            print(f"  Latency reduction: {((1 - tensorrt_latency/pytorch_latency) * 100):.1f}%")
            print(f"  FPS improvement: +{tensorrt_fps - pytorch_fps:.2f} FPS")

            # Performance classification
            if speedup >= 3.0:
                classification = "Excellent (3x+ speedup)"
            elif speedup >= 2.0:
                classification = "Great (2-3x speedup)"
            elif speedup >= 1.5:
                classification = "Good (1.5-2x speedup)"
            elif speedup >= 1.2:
                classification = "Moderate (1.2-1.5x speedup)"
            else:
                classification = "Minimal (< 1.2x speedup)"

            print(f"\n  Classification: {classification}")
        else:
            print(f"{'TensorRT':<15} {'N/A':<12} {'N/A':<15} {'N/A':<10}")
            print(f"\n⚠ TensorRT engines not available")
            print(f"  To enable TensorRT:")
            print(f"    1. python onnxexport_video.py")
            print(f"    2. bash scripts/build_video_engines.sh fp16")

        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Video Tracking TensorRT Benchmark"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person",
        help="Text prompt for detection"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to benchmark"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/trt_comparison.json",
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Validate video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return

    # Show version info
    version_info = get_version_info()
    print(f"SAM3-TensorRT Version: {version_info['version']}")
    print(f"Phase: {version_info['phase']}")
    print(f"TensorRT Support: {'✓' if version_info['tensorrt_support'] else '✗'}")

    # Run benchmark
    benchmark = TensorRTComparativeBenchmark(
        video_path=args.video,
        prompt=args.prompt,
        device=args.device
    )

    results = benchmark.run_comparison(num_frames=args.frames)

    # Print report
    benchmark.print_comparison_report(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove frame times before saving (too large)
    results_to_save = {
        'pytorch': {k: v for k, v in results['pytorch'].items() if k != 'frame_times'},
        'tensorrt': {k: v for k, v in results['tensorrt'].items() if k != 'frame_times'} if results['tensorrt'] else None
    }

    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
