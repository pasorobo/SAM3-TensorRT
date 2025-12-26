"""
Memory Bank Implementation for SAM3 Video Tracking

Implements the FIFO-based memory management system used by SAM3's
video tracker, maintaining recent frames and prompted frames for
temporal consistency in object tracking.

Author: Claude Code
Date: 2025-12-26
"""

from collections import deque
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings


class MemoryBank:
    """
    SAM3 Memory Bank for Video Tracking

    Maintains two FIFO queues:
    1. Recent Frames: Stores the N most recent frame features
    2. Prompted Frames: Stores M frames where prompts were added

    This allows the tracker to maintain temporal consistency while
    focusing on frames where the user provided explicit guidance.
    """

    def __init__(
        self,
        max_recent_frames: int = 8,
        max_prompted_frames: int = 4,
        feature_dim: int = 256,
        feature_size: Tuple[int, int] = (64, 64)
    ):
        """
        Initialize the memory bank.

        Args:
            max_recent_frames: Maximum number of recent frames to store (N)
            max_prompted_frames: Maximum number of prompted frames to store (M)
            feature_dim: Feature dimension (D)
            feature_size: Spatial size of features (H', W')
        """
        self.max_recent = max_recent_frames
        self.max_prompted = max_prompted_frames
        self.feature_dim = feature_dim
        self.feature_size = feature_size

        # FIFO queues using deque with maxlen
        self.recent_frames: deque = deque(maxlen=max_recent_frames)
        self.prompted_frames: deque = deque(maxlen=max_prompted_frames)

        # Track frame metadata
        self.frame_count = 0
        self.total_frames_processed = 0

    def add_frame(
        self,
        frame_id: int,
        features: np.ndarray,
        has_prompt: bool = False,
        metadata: Optional[Dict] = None
    ):
        """
        Add a new frame to the memory bank.

        Args:
            frame_id: Unique frame identifier
            features: Encoded features, shape [D, H', W'] or [B, D, H', W']
            has_prompt: Whether this frame has an associated prompt
            metadata: Optional additional metadata (e.g., detections, prompts)
        """
        # Ensure features are in the correct shape
        if features.ndim == 4:
            # Remove batch dimension if present
            features = features[0]

        if features.shape[0] != self.feature_dim:
            warnings.warn(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {features.shape[0]}. This may cause issues."
            )

        # Create frame data structure
        frame_data = {
            'frame_id': frame_id,
            'features': features,
            'timestamp': self.total_frames_processed,
            'has_prompt': has_prompt,
            'metadata': metadata or {}
        }

        # Add to recent frames (always)
        self.recent_frames.append(frame_data)

        # Add to prompted frames (only if prompted)
        if has_prompt:
            self.prompted_frames.append(frame_data)

        self.total_frames_processed += 1

    def get_memory_features(self) -> np.ndarray:
        """
        Get combined memory features for the tracker.

        Returns:
            memory_features: Array of shape [N+M, D, H', W']
                            where N is recent frames and M is prompted frames
        """
        all_features = []

        # Collect features from recent frames
        for frame_data in self.recent_frames:
            all_features.append(frame_data['features'])

        # Collect features from prompted frames
        # (avoiding duplicates if a frame is in both queues)
        prompted_ids = {f['frame_id'] for f in self.prompted_frames}
        recent_ids = {f['frame_id'] for f in self.recent_frames}

        for frame_data in self.prompted_frames:
            if frame_data['frame_id'] not in recent_ids:
                all_features.append(frame_data['features'])

        if not all_features:
            # Return empty/dummy features if no frames in memory
            return np.zeros(
                (1, self.feature_dim, *self.feature_size),
                dtype=np.float32
            )

        # Stack features: [N+M, D, H', W']
        memory_features = np.stack(all_features, axis=0)
        return memory_features

    def get_recent_features(self) -> np.ndarray:
        """
        Get only recent frame features.

        Returns:
            recent_features: Array of shape [N, D, H', W']
        """
        if not self.recent_frames:
            return np.zeros(
                (1, self.feature_dim, *self.feature_size),
                dtype=np.float32
            )

        features = [f['features'] for f in self.recent_frames]
        return np.stack(features, axis=0)

    def get_prompted_features(self) -> np.ndarray:
        """
        Get only prompted frame features.

        Returns:
            prompted_features: Array of shape [M, D, H', W']
        """
        if not self.prompted_frames:
            return np.zeros(
                (1, self.feature_dim, *self.feature_size),
                dtype=np.float32
            )

        features = [f['features'] for f in self.prompted_frames]
        return np.stack(features, axis=0)

    def get_memory_stats(self) -> Dict:
        """
        Get statistics about the memory bank.

        Returns:
            stats: Dictionary with memory bank statistics
        """
        return {
            'recent_frames_count': len(self.recent_frames),
            'prompted_frames_count': len(self.prompted_frames),
            'total_frames_processed': self.total_frames_processed,
            'max_recent': self.max_recent,
            'max_prompted': self.max_prompted,
            'memory_utilization_recent': len(self.recent_frames) / self.max_recent,
            'memory_utilization_prompted': (
                len(self.prompted_frames) / self.max_prompted
                if self.max_prompted > 0 else 0
            )
        }

    def clear(self):
        """Clear all frames from the memory bank."""
        self.recent_frames.clear()
        self.prompted_frames.clear()
        self.frame_count = 0
        # Note: total_frames_processed is NOT reset to track lifetime stats

    def reset(self):
        """Reset the memory bank completely, including counters."""
        self.clear()
        self.total_frames_processed = 0

    def __len__(self) -> int:
        """Return total number of unique frames in memory."""
        recent_ids = {f['frame_id'] for f in self.recent_frames}
        prompted_ids = {f['frame_id'] for f in self.prompted_frames}
        return len(recent_ids | prompted_ids)

    def __repr__(self) -> str:
        stats = self.get_memory_stats()
        return (
            f"MemoryBank("
            f"recent={stats['recent_frames_count']}/{self.max_recent}, "
            f"prompted={stats['prompted_frames_count']}/{self.max_prompted}, "
            f"total_processed={self.total_frames_processed})"
        )


class MemoryBankConfig:
    """
    Configuration for MemoryBank.

    Provides preset configurations for different use cases.
    """

    @staticmethod
    def default():
        """Default configuration (balanced)."""
        return {
            'max_recent_frames': 8,
            'max_prompted_frames': 4,
            'feature_dim': 256,
            'feature_size': (64, 64)
        }

    @staticmethod
    def low_memory():
        """Low memory footprint (for resource-constrained devices)."""
        return {
            'max_recent_frames': 4,
            'max_prompted_frames': 2,
            'feature_dim': 256,
            'feature_size': (64, 64)
        }

    @staticmethod
    def high_accuracy():
        """High accuracy (more context, higher memory usage)."""
        return {
            'max_recent_frames': 16,
            'max_prompted_frames': 8,
            'feature_dim': 256,
            'feature_size': (64, 64)
        }

    @staticmethod
    def real_time():
        """Optimized for real-time performance."""
        return {
            'max_recent_frames': 6,
            'max_prompted_frames': 3,
            'feature_dim': 256,
            'feature_size': (64, 64)
        }
