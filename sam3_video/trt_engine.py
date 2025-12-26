"""
TensorRT Inference Engine Wrapper

Provides a unified interface for TensorRT inference with SAM3 video components.
Handles engine loading, memory management, and inference execution.

Author: Claude Code
Date: 2025-12-26
Phase: 1-B (TensorRT Integration)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings


class TensorRTEngine:
    """
    TensorRT Engine wrapper for SAM3 video inference.

    Handles TensorRT engine loading and inference execution with
    automatic memory management.
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        logger_severity: str = 'WARNING'
    ):
        """
        Initialize TensorRT engine.

        Args:
            engine_path: Path to TensorRT engine (.plan file)
            logger_severity: TensorRT logger severity level
        """
        self.engine_path = Path(engine_path)

        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        # Try to import TensorRT
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            self.trt = trt
            self.cuda = cuda
            self.has_tensorrt = True
        except ImportError as e:
            warnings.warn(
                f"TensorRT not available: {e}\n"
                "Install TensorRT for acceleration, falling back to PyTorch"
            )
            self.has_tensorrt = False
            return

        # Setup logger
        severity_map = {
            'VERBOSE': trt.Logger.VERBOSE,
            'INFO': trt.Logger.INFO,
            'WARNING': trt.Logger.WARNING,
            'ERROR': trt.Logger.ERROR,
        }
        self.logger = trt.Logger(severity_map.get(logger_severity, trt.Logger.WARNING))

        # Load engine
        self._load_engine()

        # Setup bindings
        self._setup_bindings()

        # Create CUDA stream
        self.stream = cuda.Stream()

    def _load_engine(self):
        """Load TensorRT engine from file."""
        with open(self.engine_path, 'rb') as f:
            runtime = self.trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {self.engine_path}")

        self.context = self.engine.create_execution_context()

    def _setup_bindings(self):
        """Setup input/output bindings."""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))

            # Allocate GPU memory
            size = self.trt.volume(shape) * dtype.itemsize
            device_mem = self.cuda.mem_alloc(size)

            binding_info = {
                'name': name,
                'shape': shape,
                'dtype': dtype,
                'device_mem': device_mem,
                'size': size
            }

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)

    def infer(
        self,
        input_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on input data.

        Args:
            input_data: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if not self.has_tensorrt:
            raise RuntimeError("TensorRT not available")

        # Transfer input data to GPU
        for inp in self.inputs:
            if inp['name'] not in input_data:
                raise ValueError(f"Missing input: {inp['name']}")

            data = input_data[inp['name']].astype(inp['dtype'])

            # Set dynamic shape if needed
            if -1 in inp['shape']:
                self.context.set_input_shape(inp['name'], data.shape)

            self.cuda.memcpy_htod_async(
                inp['device_mem'],
                np.ascontiguousarray(data),
                self.stream
            )

        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer outputs back to CPU
        outputs = {}
        for out in self.outputs:
            # Get actual output shape (may be dynamic)
            output_shape = self.context.get_tensor_shape(out['name'])
            output_array = np.empty(output_shape, dtype=out['dtype'])

            self.cuda.memcpy_dtoh_async(
                output_array,
                out['device_mem'],
                self.stream
            )

            outputs[out['name']] = output_array

        self.stream.synchronize()
        return outputs

    def get_input_info(self) -> List[Dict]:
        """Get information about input tensors."""
        return [
            {
                'name': inp['name'],
                'shape': inp['shape'],
                'dtype': str(inp['dtype'])
            }
            for inp in self.inputs
        ]

    def get_output_info(self) -> List[Dict]:
        """Get information about output tensors."""
        return [
            {
                'name': out['name'],
                'shape': out['shape'],
                'dtype': str(out['dtype'])
            }
            for out in self.outputs
        ]

    def __del__(self):
        """Cleanup resources."""
        if self.has_tensorrt:
            # Free GPU memory
            for inp in self.inputs:
                if 'device_mem' in inp:
                    inp['device_mem'].free()
            for out in self.outputs:
                if 'device_mem' in out:
                    out['device_mem'].free()

    def __repr__(self) -> str:
        if not self.has_tensorrt:
            return "TensorRTEngine(not available)"

        return (
            f"TensorRTEngine("
            f"path={self.engine_path.name}, "
            f"inputs={len(self.inputs)}, "
            f"outputs={len(self.outputs)})"
        )


class TensorRTEngineManager:
    """
    Manages multiple TensorRT engines for video tracking.

    Handles encoder and tracker engines separately.
    """

    def __init__(
        self,
        encoder_engine_path: Optional[str] = None,
        tracker_engine_path: Optional[str] = None
    ):
        """
        Initialize engine manager.

        Args:
            encoder_engine_path: Path to encoder TensorRT engine
            tracker_engine_path: Path to tracker TensorRT engine
        """
        self.encoder_engine = None
        self.tracker_engine = None

        if encoder_engine_path:
            try:
                self.encoder_engine = TensorRTEngine(encoder_engine_path)
                print(f"✓ Loaded encoder engine: {encoder_engine_path}")
            except Exception as e:
                warnings.warn(f"Failed to load encoder engine: {e}")

        if tracker_engine_path:
            try:
                self.tracker_engine = TensorRTEngine(tracker_engine_path)
                print(f"✓ Loaded tracker engine: {tracker_engine_path}")
            except Exception as e:
                warnings.warn(f"Failed to load tracker engine: {e}")

    def has_encoder(self) -> bool:
        """Check if encoder engine is available."""
        return self.encoder_engine is not None and self.encoder_engine.has_tensorrt

    def has_tracker(self) -> bool:
        """Check if tracker engine is available."""
        return self.tracker_engine is not None and self.tracker_engine.has_tensorrt

    def is_ready(self) -> bool:
        """Check if both engines are ready."""
        return self.has_encoder() and self.has_tracker()

    def encode(self, pixel_values: np.ndarray) -> np.ndarray:
        """
        Encode frame using TensorRT encoder.

        Args:
            pixel_values: Input frame [B, C, H, W]

        Returns:
            features: Encoded features [B, D, H', W']
        """
        if not self.has_encoder():
            raise RuntimeError("Encoder engine not available")

        outputs = self.encoder_engine.infer({
            'pixel_values': pixel_values
        })

        return outputs['features']

    def track(
        self,
        image_embeddings: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Track objects using TensorRT tracker.

        Args:
            image_embeddings: Image features [B, D, H', W']
            input_ids: Text token IDs [B, N]
            attention_mask: Attention mask [B, N]

        Returns:
            pred_masks: Predicted masks [B, Q, H, W]
        """
        if not self.has_tracker():
            raise RuntimeError("Tracker engine not available")

        outputs = self.tracker_engine.infer({
            'image_embeddings': image_embeddings,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })

        return outputs['pred_masks']

    def __repr__(self) -> str:
        encoder_status = "✓" if self.has_encoder() else "✗"
        tracker_status = "✓" if self.has_tracker() else "✗"

        return (
            f"TensorRTEngineManager("
            f"encoder={encoder_status}, "
            f"tracker={tracker_status})"
        )
