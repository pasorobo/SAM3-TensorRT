"""
SAM3 Video Tracking Demo with TensorRT

Demonstration of SAM3 video tracking with TensorRT acceleration.
Automatically falls back to PyTorch if TensorRT engines are not available.

Usage:
    python examples/video_tracking_demo_trt.py --video path/to/video.mp4 --prompt "person"

Author: Claude Code
Date: 2025-12-26
Phase: 1-B (TensorRT Integration)
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3_video import create_session_manager, get_version_info


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Video Tracking Demo with TensorRT Acceleration"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person",
        help="Text prompt for object detection (e.g., 'person', 'car', 'dog')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output video (default: input_tracked_trt.mp4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display video while processing"
    )
    parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        help="Disable TensorRT (use PyTorch only)"
    )
    parser.add_argument(
        "--encoder-engine",
        type=str,
        default=None,
        help="Path to encoder TensorRT engine"
    )
    parser.add_argument(
        "--tracker-engine",
        type=str,
        default=None,
        help="Path to tracker TensorRT engine"
    )

    args = parser.parse_args()

    # Setup output path
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_tracked_trt.mp4")

    print("=" * 70)
    print("SAM3 Video Tracking Demo with TensorRT Acceleration")
    print("=" * 70)

    # Show version info
    version_info = get_version_info()
    print(f"\nVersion: {version_info['version']}")
    print(f"Phase: {version_info['phase']}")
    print(f"TensorRT Support: {'✓' if version_info['tensorrt_support'] else '✗'}")

    print(f"\nConfiguration:")
    print(f"  Video: {args.video}")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Output: {args.output}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  TensorRT: {'disabled' if args.no_tensorrt else 'enabled (if available)'}")

    if args.encoder_engine:
        print(f"  Encoder Engine: {args.encoder_engine}")
    if args.tracker_engine:
        print(f"  Tracker Engine: {args.tracker_engine}")

    print("=" * 70)

    # Initialize session manager with TensorRT support
    print("\nInitializing SAM3 Video Session Manager...")
    session = create_session_manager(
        use_tensorrt=not args.no_tensorrt,
        device=args.device,
        encoder_engine_path=args.encoder_engine,
        tracker_engine_path=args.tracker_engine
    )

    # Show inference info
    if hasattr(session, 'get_inference_info'):
        info = session.get_inference_info()
        print(f"\nInference Mode: {info['mode'].upper()}")
        if info.get('tensorrt_available'):
            print(f"  Encoder: {'✓' if info.get('encoder_ready') else '✗'}")
            print(f"  Tracker: {'✓' if info.get('tracker_ready') else '✗'}")

    # Start session
    print("\nStarting video tracking session...")
    session.start_session(
        text_prompt=args.prompt,
        video_source=args.video
    )

    # Process video
    print("\nProcessing video...")
    results = session.process_video(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
        display=args.display
    )

    # End session
    session.end_session()

    # Display results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Frames processed: {results['frames_processed']}")
    print(f"Total detections: {results['total_detections']}")

    stats = results['stats']
    print(f"\nPerformance:")
    print(f"  Average FPS: {stats['avg_fps']:.2f}")
    print(f"  Average processing time: {stats['avg_processing_time']*1000:.2f}ms")
    print(f"  Std dev: {stats['std_processing_time']*1000:.2f}ms")

    # Performance comparison hint
    if hasattr(session, 'get_inference_info'):
        info = session.get_inference_info()
        if info['mode'] == 'pytorch':
            print(f"\n💡 Tip: Enable TensorRT for 2-4x speedup!")
            print(f"   1. Export ONNX: python onnxexport_video.py")
            print(f"   2. Build engines: bash scripts/build_video_engines.sh fp16")
            print(f"   3. Re-run this script")

    memory_stats = stats['memory_stats']
    print(f"\nMemory Bank:")
    print(f"  Recent frames: {memory_stats['recent_frames_count']}/{memory_stats['max_recent']}")
    print(f"  Prompted frames: {memory_stats['prompted_frames_count']}/{memory_stats['max_prompted']}")
    print(f"  Total processed: {memory_stats['total_frames_processed']}")

    print("\n" + "=" * 70)
    print(f"✓ Output saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
