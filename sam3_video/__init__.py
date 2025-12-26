"""
SAM3 Video Tracking Module

This module provides video tracking capabilities for SAM3 using
a hybrid approach: TensorRT for inference acceleration and Python
for memory/state management.

Author: Claude Code
Date: 2025-12-26
"""

from .memory_bank import MemoryBank, MemoryBankConfig
from .session_manager import Sam3VideoSessionManager

# TensorRT components (Phase 1-B)
try:
    from .trt_engine import TensorRTEngine, TensorRTEngineManager
    from .trt_session_manager import Sam3VideoSessionManagerTRT
    _HAS_TRT_SUPPORT = True
except ImportError:
    _HAS_TRT_SUPPORT = False
    TensorRTEngine = None
    TensorRTEngineManager = None
    Sam3VideoSessionManagerTRT = None


__all__ = [
    'MemoryBank',
    'MemoryBankConfig',
    'Sam3VideoSessionManager',
    'Sam3VideoSessionManagerTRT',
    'TensorRTEngine',
    'TensorRTEngineManager',
]

__version__ = '0.2.0'  # Phase 1-B


def create_session_manager(
    use_tensorrt: bool = True,
    **kwargs
):
    """
    Create a video session manager with automatic TensorRT detection.

    Args:
        use_tensorrt: Try to use TensorRT if available
        **kwargs: Arguments passed to session manager

    Returns:
        Session manager instance (TRT or PyTorch)
    """
    if use_tensorrt and _HAS_TRT_SUPPORT:
        return Sam3VideoSessionManagerTRT(**kwargs)
    else:
        return Sam3VideoSessionManager(**kwargs)


def get_version_info() -> dict:
    """Get version and capability information."""
    return {
        'version': __version__,
        'tensorrt_support': _HAS_TRT_SUPPORT,
        'phase': '1-B' if _HAS_TRT_SUPPORT else '1-A',
    }
