"""
SAM3 Video Tracking Benchmark

Comprehensive benchmarking suite for SAM3 video tracking performance.

Usage:
    python benchmarks/video_benchmark.py --video test.mp4 --prompt "person"

Author: Claude Code
Date: 2025-12-26
"""

import argparse
import sys
from pathlib import Path
import time
import json
import numpy as np
import cv2
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3_video import Sam3VideoSessionManager


class VideoBenchmark:
    """Benchmark suite for SAM3 video tracking."""

    def __init__(self, video_path: str, prompt: str, device: str = None):
        self.video_path = video_path
        self.prompt = prompt
        self.device = device
        self.results = {}

    def benchmark_session(
        self,
        num_frames: int = 100,
        num_runs: int = 3
    ) -> Dict:
        """Benchmark video processing performance."""

        print(f"\n{'='*60}")
        print(f"Benchmarking SAM3 Video Tracking")
        print(f"{'='*60}")
        print(f"Video: {self.video_path}")
        print(f"Prompt: '{self.prompt}'")
        print(f"Frames: {num_frames}")
        print(f"Runs: {num_runs}")
        print(f"Device: {self.device or 'auto'}")
        print(f"{'='*60}\n")

        all_run_results = []

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}...")

            # Initialize session
            session = Sam3VideoSessionManager(device=self.device)
            session.start_session(text_prompt=self.prompt)

            # Open video
            cap = cv2.VideoCapture(self.video_path)

            frame_times = []
            encoder_times = []
            total_detections = 0

            frame_count = 0
            while cap.isOpened() and frame_count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Benchmark single frame
                start = time.perf_counter()
                masks, detections = session.process_frame(frame)
                end = time.perf_counter()

                frame_time = end - start
                frame_times.append(frame_time)
                total_detections += len(detections)

                frame_count += 1

                if frame_count % 10 == 0:
                    print(f"  Processed {frame_count}/{num_frames} frames...")

            cap.release()
            session.end_session()

            # Compute statistics
            run_results = {
                'run': run + 1,
                'frames_processed': frame_count,
                'total_detections': total_detections,
                'frame_times': frame_times,
                'avg_time': np.mean(frame_times),
                'std_time': np.std(frame_times),
                'min_time': np.min(frame_times),
                'max_time': np.max(frame_times),
                'median_time': np.median(frame_times),
                'avg_fps': 1.0 / np.mean(frame_times) if np.mean(frame_times) > 0 else 0,
            }

            all_run_results.append(run_results)

            print(f"  ✓ Avg FPS: {run_results['avg_fps']:.2f}")
            print(f"  ✓ Avg time: {run_results['avg_time']*1000:.2f}ms")

        # Aggregate results
        avg_fps_all = np.mean([r['avg_fps'] for r in all_run_results])
        std_fps_all = np.std([r['avg_fps'] for r in all_run_results])
        avg_time_all = np.mean([r['avg_time'] for r in all_run_results])
        std_time_all = np.std([r['avg_time'] for r in all_run_results])

        results = {
            'video': self.video_path,
            'prompt': self.prompt,
            'num_runs': num_runs,
            'num_frames': num_frames,
            'device': self.device or 'auto',
            'runs': all_run_results,
            'summary': {
                'avg_fps': float(avg_fps_all),
                'std_fps': float(std_fps_all),
                'avg_time_ms': float(avg_time_all * 1000),
                'std_time_ms': float(std_time_all * 1000),
            }
        }

        self.results = results
        return results

    def print_report(self):
        """Print formatted benchmark report."""

        if not self.results:
            print("No results available. Run benchmark first.")
            return

        summary = self.results['summary']

        print(f"\n{'='*60}")
        print("Benchmark Results")
        print(f"{'='*60}\n")

        print(f"Configuration:")
        print(f"  Video: {self.results['video']}")
        print(f"  Prompt: '{self.results['prompt']}'")
        print(f"  Device: {self.results['device']}")
        print(f"  Runs: {self.results['num_runs']}")
        print(f"  Frames per run: {self.results['num_frames']}")

        print(f"\nPerformance:")
        print(f"  Average FPS: {summary['avg_fps']:.2f} ± {summary['std_fps']:.2f}")
        print(f"  Average Latency: {summary['avg_time_ms']:.2f}ms ± {summary['std_time_ms']:.2f}ms")

        # Per-run breakdown
        print(f"\nPer-Run Breakdown:")
        print(f"  {'Run':<6} {'FPS':<10} {'Avg Time':<15} {'Detections':<12}")
        print(f"  {'-'*50}")

        for run in self.results['runs']:
            print(
                f"  {run['run']:<6} "
                f"{run['avg_fps']:<10.2f} "
                f"{run['avg_time']*1000:<15.2f}ms "
                f"{run['total_detections']:<12}"
            )

        # Performance classification
        avg_fps = summary['avg_fps']
        print(f"\nPerformance Classification:")
        if avg_fps >= 30:
            classification = "Excellent (Real-time @ 30+ FPS)"
        elif avg_fps >= 20:
            classification = "Good (Near real-time @ 20+ FPS)"
        elif avg_fps >= 10:
            classification = "Acceptable (Interactive @ 10+ FPS)"
        else:
            classification = "Needs optimization (< 10 FPS)"

        print(f"  {classification}")

        print(f"\n{'='*60}\n")

    def save_results(self, output_path: str):
        """Save results to JSON file."""

        if not self.results:
            print("No results to save.")
            return

        # Remove frame times (too large)
        results_to_save = self.results.copy()
        for run in results_to_save['runs']:
            run.pop('frame_times', None)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 Video Tracking Benchmark")
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
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs"
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
        default="benchmarks/results/benchmark_results.json",
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Validate video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return

    # Run benchmark
    benchmark = VideoBenchmark(
        video_path=args.video,
        prompt=args.prompt,
        device=args.device
    )

    results = benchmark.benchmark_session(
        num_frames=args.frames,
        num_runs=args.runs
    )

    # Print report
    benchmark.print_report()

    # Save results
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
