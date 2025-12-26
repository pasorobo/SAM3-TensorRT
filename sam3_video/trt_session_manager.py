"""
TensorRT-Accelerated Video Session Manager

Extended version of Sam3VideoSessionManager with TensorRT acceleration support.
Provides seamless fallback to PyTorch if TensorRT engines are not available.

Author: Claude Code
Date: 2025-12-26
Phase: 1-B (TensorRT Integration)
"""

import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import warnings
import time

from .session_manager import Sam3VideoSessionManager
from .trt_engine import TensorRTEngineManager


class Sam3VideoSessionManagerTRT(Sam3VideoSessionManager):
    """
    TensorRT-Accelerated Video Session Manager

    Extends Sam3VideoSessionManager with TensorRT engine support
    for 2-4x performance improvement.

    Falls back to PyTorch if TensorRT engines are not available.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: Optional[str] = None,
        memory_config: Optional[Dict] = None,
        dtype = None,  # torch.dtype
        # TensorRT specific
        use_tensorrt: bool = True,
        encoder_engine_path: Optional[str] = None,
        tracker_engine_path: Optional[str] = None
    ):
        """
        Initialize TensorRT-enabled session manager.

        Args:
            model_name: Hugging Face model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            memory_config: Memory bank configuration
            dtype: Model dtype (only for PyTorch fallback)
            use_tensorrt: Whether to use TensorRT if available
            encoder_engine_path: Path to encoder TensorRT engine
            tracker_engine_path: Path to tracker TensorRT engine
        """
        # Initialize base class (PyTorch)
        if dtype is None:
            import torch
            dtype = torch.float32

        super().__init__(
            model_name=model_name,
            device=device,
            memory_config=memory_config,
            dtype=dtype
        )

        # TensorRT setup
        self.use_tensorrt = use_tensorrt
        self.trt_manager = None
        self.inference_mode = "pytorch"  # or "tensorrt"

        if use_tensorrt:
            self._setup_tensorrt(encoder_engine_path, tracker_engine_path)

    def _setup_tensorrt(
        self,
        encoder_engine_path: Optional[str],
        tracker_engine_path: Optional[str]
    ):
        """Setup TensorRT engines."""

        # Auto-discover engines if not specified
        if encoder_engine_path is None:
            candidates = [
                "trt_engines/sam3_video_encoder_fp16.plan",
                "sam3_video_encoder_fp16.plan",
                "trt_engines/sam3_video_encoder_int8.plan",
            ]
            for candidate in candidates:
                if Path(candidate).exists():
                    encoder_engine_path = candidate
                    break

        if tracker_engine_path is None:
            candidates = [
                "trt_engines/sam3_video_tracker_fp16.plan",
                "sam3_video_tracker_fp16.plan",
                "trt_engines/sam3_video_tracker_int8.plan",
            ]
            for candidate in candidates:
                if Path(candidate).exists():
                    tracker_engine_path = candidate
                    break

        # Load engines
        if encoder_engine_path or tracker_engine_path:
            print(f"\nTrying to load TensorRT engines...")
            try:
                self.trt_manager = TensorRTEngineManager(
                    encoder_engine_path=encoder_engine_path,
                    tracker_engine_path=tracker_engine_path
                )

                if self.trt_manager.is_ready():
                    self.inference_mode = "tensorrt"
                    print(f"✓ TensorRT acceleration enabled!")
                    print(f"  {self.trt_manager}")
                else:
                    print(f"⚠ TensorRT engines partially loaded")
                    print(f"  {self.trt_manager}")
                    print(f"  Falling back to PyTorch")

            except Exception as e:
                warnings.warn(f"Failed to initialize TensorRT: {e}")
                print(f"  Falling back to PyTorch")
        else:
            print(f"\nNo TensorRT engines found, using PyTorch")
            print(f"  To enable TensorRT:")
            print(f"    1. Export ONNX: python onnxexport_video.py")
            print(f"    2. Build engines: bash scripts/build_video_engines.sh fp16")

    def process_frame(
        self,
        frame: np.ndarray,
        add_prompt: bool = False,
        return_embeddings: bool = False
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single video frame with TensorRT acceleration.

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

        start_time = time.time()

        # Use TensorRT if available
        if self.inference_mode == "tensorrt" and self.trt_manager.is_ready():
            return self._process_frame_tensorrt(frame, add_prompt, return_embeddings)
        else:
            # Fallback to PyTorch
            return super().process_frame(frame, add_prompt, return_embeddings)

    def _process_frame_tensorrt(
        self,
        frame: np.ndarray,
        add_prompt: bool = False,
        return_embeddings: bool = False
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Process frame using TensorRT engines."""

        start_time = time.time()

        # 1. Preprocess frame
        inputs = self._preprocess_frame(frame)

        # Convert PyTorch tensors to numpy
        pixel_values_np = inputs['pixel_values'].cpu().numpy()
        input_ids_np = inputs['input_ids'].cpu().numpy()
        attention_mask_np = inputs['attention_mask'].cpu().numpy()

        # 2. Encode with TensorRT
        image_embeddings_np = self.trt_manager.encode(pixel_values_np)

        # 3. Add to memory bank
        self.memory_bank.add_frame(
            frame_id=self.frame_count,
            features=image_embeddings_np,
            has_prompt=add_prompt
        )

        # 4. Track with TensorRT
        pred_masks_np = self.trt_manager.track(
            image_embeddings=image_embeddings_np,
            input_ids=input_ids_np,
            attention_mask=attention_mask_np
        )

        # Apply sigmoid
        pred_masks_np = 1.0 / (1.0 + np.exp(-pred_masks_np))

        # 5. Post-process outputs
        masks_np = pred_masks_np[0]  # [Q, H, W]
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

        if return_embeddings:
            return masks_np, detections, image_embeddings_np
        else:
            return masks_np, detections

    def get_inference_info(self) -> Dict:
        """Get information about current inference mode."""
        info = {
            'mode': self.inference_mode,
            'use_tensorrt': self.use_tensorrt,
            'tensorrt_available': self.trt_manager is not None,
        }

        if self.trt_manager:
            info['encoder_ready'] = self.trt_manager.has_encoder()
            info['tracker_ready'] = self.trt_manager.has_tracker()

        return info

    def __repr__(self) -> str:
        base = super().__repr__()
        mode_info = f"mode={self.inference_mode}"
        return base.replace(")", f", {mode_info})")
