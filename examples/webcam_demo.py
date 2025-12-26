"""
SAM3 Webcam Real-time Tracking Demo

Real-time object tracking using webcam input.

Usage:
    python examples/webcam_demo.py --prompt "person"

Press 'q' to quit, 'p' to add prompt to current frame.

Author: Claude Code
Date: 2025-12-26
"""

import argparse
import sys
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3_video import Sam3VideoSessionManager


def main():
    parser = argparse.ArgumentParser(description="SAM3 Webcam Tracking Demo")
    parser.add_argument(
        "--prompt",
        type=str,
        default="person",
        help="Text prompt for object detection"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SAM3 Webcam Real-time Tracking Demo")
    print("=" * 60)
    print(f"Camera: {args.camera}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Resolution: {args.width}x{args.height}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'p' - Add prompt to current frame")
    print("  'r' - Reset memory")
    print("=" * 60)

    # Initialize session manager
    print("\nInitializing...")
    session = Sam3VideoSessionManager(device=args.device)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    # Start session
    session.start_session(text_prompt=args.prompt)

    print("\n✓ Ready! Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame
            try:
                masks, detections = session.process_frame(frame)

                # Visualize
                vis_frame = session._visualize(frame, masks, detections)

                # Display FPS
                stats = session.get_stats()
                fps_text = f"FPS: {stats['avg_fps']:.1f}"
                cv2.putText(
                    vis_frame, fps_text, (10, vis_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

                cv2.imshow('SAM3 Webcam Tracking', vis_frame)

            except Exception as e:
                print(f"Error processing frame: {e}")
                cv2.imshow('SAM3 Webcam Tracking', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                print("Adding prompt to current frame...")
                # Reprocess with prompt
                masks, detections = session.process_frame(frame, add_prompt=True)
            elif key == ord('r'):
                print("Resetting memory...")
                session.memory_bank.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        session.end_session()

    print("\n✓ Demo ended")


if __name__ == "__main__":
    main()
