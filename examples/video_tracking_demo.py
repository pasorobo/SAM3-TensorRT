"""
SAM3 Video Tracking Demo

Basic demonstration of SAM3 video tracking capabilities.

Usage:
    python examples/video_tracking_demo.py --video path/to/video.mp4 --prompt "person"

Author: Claude Code
Date: 2025-12-26
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3_video import Sam3VideoSessionManager


def main():
    parser = argparse.ArgumentParser(description="SAM3 Video Tracking Demo")
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
        help="Path to output video (default: input_tracked.mp4)"
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
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype for inference"
    )

    args = parser.parse_args()

    # Setup output path
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_tracked.mp4")

    # Map dtype string to torch dtype
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("SAM3 Video Tracking Demo")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Output: {args.output}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Dtype: {args.dtype}")
    print("=" * 60)

    # Initialize session manager
    print("\nInitializing SAM3 Video Session Manager...")
    session = Sam3VideoSessionManager(
        device=args.device,
        dtype=dtype
    )

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
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Frames processed: {results['frames_processed']}")
    print(f"Total detections: {results['total_detections']}")

    stats = results['stats']
    print(f"\nPerformance:")
    print(f"  Average FPS: {stats['avg_fps']:.2f}")
    print(f"  Average processing time: {stats['avg_processing_time']*1000:.2f}ms")
    print(f"  Std dev: {stats['std_processing_time']*1000:.2f}ms")

    memory_stats = stats['memory_stats']
    print(f"\nMemory Bank:")
    print(f"  Recent frames: {memory_stats['recent_frames_count']}/{memory_stats['max_recent']}")
    print(f"  Prompted frames: {memory_stats['prompted_frames_count']}/{memory_stats['max_prompted']}")
    print(f"  Total processed: {memory_stats['total_frames_processed']}")

    print("\n" + "=" * 60)
    print(f"✓ Output saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
