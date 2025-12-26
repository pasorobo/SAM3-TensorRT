"""
SAM3 Video Tracking Module

This module provides video tracking capabilities for SAM3 using
a hybrid approach: TensorRT for inference acceleration and Python
for memory/state management.

Author: Claude Code
Date: 2025-12-26
"""

from .memory_bank import MemoryBank
from .session_manager import Sam3VideoSessionManager

__all__ = [
    'MemoryBank',
    'Sam3VideoSessionManager',
]

__version__ = '0.1.0'
