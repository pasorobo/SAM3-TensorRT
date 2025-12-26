"""
Simple SAM3 Video Tracking Example

Minimal example showing basic usage.

Author: Claude Code
Date: 2025-12-26
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3_video import Sam3VideoSessionManager


def main():
    # Initialize session manager
    session = Sam3VideoSessionManager()

    # Start tracking session
    session.start_session(text_prompt="person")

    # Process video
    results = session.process_video(
        video_path="path/to/your/video.mp4",
        output_path="output_tracked.mp4"
    )

    # End session
    session.end_session()

    # Print results
    print(f"Processed {results['frames_processed']} frames")
    print(f"Found {results['total_detections']} detections")
    print(f"Average FPS: {results['stats']['avg_fps']:.2f}")


if __name__ == "__main__":
    main()
