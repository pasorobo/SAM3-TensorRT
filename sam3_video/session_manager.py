"""
SAM3 Video Session Manager

Manages video tracking sessions with SAM3, including frame processing,
memory management, and object tracking state.

Phase 1 Implementation: Uses PyTorch models with Python-based memory management.
Future phases will integrate TensorRT for acceleration.

Author: Claude Code
Date: 2025-12-26
"""

import torch
import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import warnings
from tqdm import tqdm

from .memory_bank import MemoryBank, MemoryBankConfig


class Sam3VideoSessionManager:
    """
    SAM3 Video Session Manager

    Manages video tracking sessions using SAM3 with text prompts.
    Handles frame-by-frame processing, memory management, and tracking state.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: Optional[str] = None,
        memory_config: Optional[Dict] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the session manager.

        Args:
            model_name: Hugging Face model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            memory_config: Memory bank configuration
            dtype: Model dtype (torch.float32, torch.float16, torch.bfloat16)
        """
        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype = dtype

        print(f"Initializing SAM3 Video Session Manager...")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.dtype}")

        # Load model and processor
        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor

            print(f"  Loading model: {model_name}...")
            self.model = Sam3VideoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device, dtype=self.dtype)
            self.model.eval()

            self.processor = Sam3VideoProcessor.from_pretrained(model_name)
            print("  ✓ Model loaded successfully")

        except ImportError as e:
            print(f"  ✗ Error: transformers library issue: {e}")
            print("\n  Please install the latest transformers:")
            print("    pip install git+https://github.com/huggingface/transformers.git")
            raise

        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
            raise

        # Memory bank setup
        if memory_config is None:
            memory_config = MemoryBankConfig.default()

        self.memory_bank = MemoryBank(**memory_config)
        print(f"  ✓ Memory bank initialized: {self.memory_bank}")

        # Session state
        self.is_active = False
        self.frame_count = 0
        self.text_prompt = None
        self.video_source = None

        # Tracking state
        self.tracked_objects: Dict[int, Dict] = {}
        self.next_object_id = 0

        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'processing_times': []
        }

    def start_session(
        self,
        text_prompt: str,
        video_source: Optional[str] = None,
        clear_memory: bool = True
    ):
        """
        Start a new video tracking session.

        Args:
            text_prompt: Text prompt for object detection (e.g., "person", "car")
            video_source: Optional video file path or camera ID
            clear_memory: Whether to clear previous memory
        """
        if self.is_active:
            print("⚠ Session already active. Ending previous session...")
            self.end_session()

        self.is_active = True
        self.frame_count = 0
        self.text_prompt = text_prompt
        self.video_source = video_source

        if clear_memory:
            self.memory_bank.reset()
            self.tracked_objects.clear()
            self.next_object_id = 0

        print(f"\n✓ Session started")
        print(f"  Prompt: '{text_prompt}'")
        if video_source:
            print(f"  Source: {video_source}")

    def process_frame(
        self,
        frame: np.ndarray,
        add_prompt: bool = False,
        return_embeddings: bool = False
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single video frame.

        Args:
            frame: Input frame as numpy array [H, W, 3] in BGR format
            add_prompt: Whether to mark this frame as prompted
            return_embeddings: Whether to return feature embeddings

        Returns:
            masks: Segmentation masks [num_objects, H, W]
            detections: List of detection dictionaries
        """
        if not self.is_active:
            raise RuntimeError("Session not started. Call start_session() first.")

        import time
        start_time = time.time()

        # 1. Preprocess frame
        inputs = self._preprocess_frame(frame)

        # 2. Encode frame
        with torch.no_grad():
            # Get image embeddings
            if hasattr(self.model, 'get_image_embeddings'):
                image_embeddings = self.model.get_image_embeddings(
                    inputs['pixel_values']
                )
            else:
                # Fallback: run full model
                outputs = self.model(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                image_embeddings = outputs.image_embeddings if hasattr(outputs, 'image_embeddings') else None

        # 3. Add to memory bank
        if image_embeddings is not None:
            embeddings_np = image_embeddings.cpu().numpy()
            self.memory_bank.add_frame(
                frame_id=self.frame_count,
                features=embeddings_np,
                has_prompt=add_prompt
            )

        # 4. Run inference with memory
        with torch.no_grad():
            outputs = self.model(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            # Get masks
            if hasattr(outputs, 'pred_masks'):
                pred_masks = torch.sigmoid(outputs.pred_masks)
            else:
                pred_masks = outputs[0]

        # 5. Post-process outputs
        masks_np = pred_masks.cpu().numpy()[0]  # [Q, H, W]
        detections = self._postprocess_outputs(
            masks=masks_np,
            original_size=frame.shape[:2],
            frame_id=self.frame_count
        )

        # 6. Update statistics
        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += len(detections)

        self.frame_count += 1

        if return_embeddings and image_embeddings is not None:
            return masks_np, detections, embeddings_np
        else:
            return masks_np, detections

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        display: bool = False,
        save_detections: bool = True
    ) -> Dict:
        """
        Process an entire video file.

        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            max_frames: Maximum number of frames to process
            display: Whether to display frames during processing
            save_detections: Whether to save detection data

        Returns:
            results: Dictionary with processing results and statistics
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"\nProcessing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Frames: {total_frames}")

        # Setup output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"  Output: {output_path}")

        # Process frames
        all_detections = []

        try:
            with tqdm(total=total_frames, desc="Processing") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if max_frames and self.frame_count >= max_frames:
                        break

                    # Process frame
                    masks, detections = self.process_frame(frame)
                    all_detections.append({
                        'frame_id': self.frame_count - 1,
                        'detections': detections
                    })

                    # Visualize
                    vis_frame = self._visualize(frame, masks, detections)

                    # Save/display
                    if out is not None:
                        out.write(vis_frame)

                    if display:
                        cv2.imshow('SAM3 Video Tracking', vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    pbar.update(1)

        finally:
            cap.release()
            if out is not None:
                out.release()
            if display:
                cv2.destroyAllWindows()

        # Compile results
        results = {
            'video_path': video_path,
            'output_path': output_path,
            'frames_processed': self.frame_count,
            'total_detections': sum(len(d['detections']) for d in all_detections),
            'detections': all_detections if save_detections else None,
            'stats': self.get_stats()
        }

        return results

    def end_session(self):
        """End the current session."""
        if not self.is_active:
            print("⚠ No active session to end.")
            return

        print(f"\n✓ Session ended")
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Memory: {self.memory_bank}")

        self.is_active = False

    def get_stats(self) -> Dict:
        """Get session statistics."""
        stats = self.stats.copy()

        if stats['processing_times']:
            times = np.array(stats['processing_times'])
            stats['avg_processing_time'] = float(np.mean(times))
            stats['std_processing_time'] = float(np.std(times))
            stats['avg_fps'] = 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        else:
            stats['avg_processing_time'] = 0
            stats['std_processing_time'] = 0
            stats['avg_fps'] = 0

        stats['memory_stats'] = self.memory_bank.get_memory_stats()

        return stats

    def _preprocess_frame(self, frame: np.ndarray) -> Dict[str, torch.Tensor]:
        """Preprocess frame for model input."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use processor
        inputs = self.processor(
            images=rgb_frame,
            text=self.text_prompt,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        return inputs

    def _postprocess_outputs(
        self,
        masks: np.ndarray,
        original_size: Tuple[int, int],
        frame_id: int
    ) -> List[Dict]:
        """Post-process model outputs to detections."""
        detections = []

        for i, mask in enumerate(masks):
            # Threshold mask
            binary_mask = (mask > 0.5).astype(np.uint8)

            # Skip empty masks
            if binary_mask.sum() == 0:
                continue

            # Resize to original size
            if binary_mask.shape != original_size:
                binary_mask = cv2.resize(
                    binary_mask,
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # Compute bounding box
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])

                detections.append({
                    'object_id': self._get_or_assign_object_id(i, frame_id),
                    'mask': binary_mask,
                    'bbox': [x, y, w, h],
                    'confidence': float(mask.max()),
                    'frame_id': frame_id
                })

        return detections

    def _get_or_assign_object_id(self, mask_idx: int, frame_id: int) -> int:
        """Assign persistent object IDs (simplified tracking)."""
        # Simplified: use mask index as object ID
        # In production, use IoU matching or feature similarity
        return mask_idx

    def _visualize(
        self,
        frame: np.ndarray,
        masks: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Visualize detections on frame."""
        vis = frame.copy()

        # Color palette
        np.random.seed(42)
        colors = [
            tuple(map(int, np.random.randint(0, 255, 3)))
            for _ in range(len(detections))
        ]

        for i, det in enumerate(detections):
            mask = det['mask']
            color = colors[i]

            # Apply colored mask overlay
            colored_mask = np.zeros_like(vis)
            colored_mask[mask > 0] = color
            vis = cv2.addWeighted(vis, 0.7, colored_mask, 0.3, 0)

            # Draw bounding box
            x, y, w, h = det['bbox']
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)

            # Draw label
            label = f"ID:{det['object_id']} ({det['confidence']:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(vis, (x, y-text_h-10), (x+text_w, y), color, -1)
            cv2.putText(
                vis, label, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        # Add info overlay
        info_text = f"Frame: {self.frame_count} | Prompt: '{self.text_prompt}'"
        cv2.putText(
            vis, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        return vis

    def __repr__(self) -> str:
        status = "Active" if self.is_active else "Inactive"
        return (
            f"Sam3VideoSessionManager("
            f"status={status}, "
            f"device={self.device}, "
            f"frames={self.frame_count})"
        )
