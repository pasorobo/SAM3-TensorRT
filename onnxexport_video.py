"""
SAM3 Video Component ONNX Export Script

This script exports SAM3 video components to ONNX format for TensorRT optimization.
Components are separated into:
1. Vision Encoder - Processes individual frames
2. Video Tracker - Tracks objects across frames with memory

Author: Claude Code
Date: 2025-12-26
"""

import torch
from pathlib import Path
from transformers import Sam3VideoModel, Sam3VideoProcessor
from PIL import Image
import requests
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# Device configuration
device = "cpu"  # Use CPU for export compatibility
print(f"Using device: {device}")

# Output directory
output_dir = Path("onnx_weights")
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("SAM3 Video ONNX Export")
print("=" * 60)

# Load model and processor
print("\n[1/5] Loading SAM3 Video model...")
try:
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    model.eval()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nNOTE: If SAM3VideoModel is not available in transformers,")
    print("please install the latest version or use the sam3 package directly:")
    print("  pip install git+https://github.com/huggingface/transformers.git")
    print("  or")
    print("  pip install sam3")
    raise

# Prepare sample data
print("\n[2/5] Preparing sample data...")
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
print(f"✓ Sample image loaded: {image.size}")


# ============================================================================
# Component 1: Vision Encoder Wrapper
# ============================================================================

class Sam3VideoEncoderWrapper(torch.nn.Module):
    """
    Wraps the SAM3 vision encoder for ONNX export.

    This component extracts features from individual video frames.
    """

    def __init__(self, sam3_video_model):
        super().__init__()
        # Extract the vision encoder from the SAM3 video model
        self.encoder = sam3_video_model.perception_encoder

    def forward(self, pixel_values):
        """
        Extract features from input frames.

        Args:
            pixel_values: [B, C, H, W] - Input frames

        Returns:
            features: [B, D, H', W'] - Encoded features
        """
        # Get image embeddings
        outputs = self.encoder(pixel_values)

        # Return the last hidden state (features)
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            return outputs


# ============================================================================
# Component 2: Video Tracker Wrapper
# ============================================================================

class Sam3VideoTrackerWrapper(torch.nn.Module):
    """
    Wraps the SAM3 video tracker for ONNX export.

    This component performs object tracking using current frame features
    and memory from previous frames.
    """

    def __init__(self, sam3_video_model):
        super().__init__()
        # Extract the tracker from the SAM3 video model
        self.tracker = sam3_video_model.tracker
        self.mask_decoder = sam3_video_model.mask_decoder

    def forward(self, image_embeddings, text_embeddings):
        """
        Perform tracking with text prompts.

        Args:
            image_embeddings: [B, D, H', W'] - Current frame features
            text_embeddings: [B, N, D] - Text prompt embeddings

        Returns:
            pred_masks: [B, Q, H, W] - Predicted segmentation masks
        """
        # Combine image and text embeddings for tracking
        # This is a simplified version - actual implementation may differ

        # Generate masks from embeddings
        outputs = self.mask_decoder(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings
        )

        # Return predicted masks
        if hasattr(outputs, 'pred_masks'):
            return outputs.pred_masks
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            return outputs


# ============================================================================
# Export Component 1: Vision Encoder
# ============================================================================

print("\n[3/5] Exporting Vision Encoder...")

try:
    encoder_wrapper = Sam3VideoEncoderWrapper(model).to(device).eval()

    # Create dummy input
    # Standard SAM3 input size is 1024x1024
    dummy_frame = torch.randn(1, 3, 1024, 1024).to(device)

    # Test forward pass
    with torch.no_grad():
        encoder_output = encoder_wrapper(dummy_frame)
        print(f"  Encoder output shape: {encoder_output.shape}")

    # Export to ONNX
    encoder_path = output_dir / "sam3_video_encoder.onnx"

    torch.onnx.export(
        encoder_wrapper,
        (dummy_frame,),
        encoder_path,
        input_names=["pixel_values"],
        output_names=["features"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "features": {0: "batch"}
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False
    )

    print(f"✓ Encoder exported to: {encoder_path}")
    print(f"  Size: {encoder_path.stat().st_size / (1024**2):.2f} MB")

except Exception as e:
    print(f"✗ Error exporting encoder: {e}")
    print("\nAttempting alternative approach...")

    # Alternative: Export the full model's encoder directly
    try:
        class SimpleEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values):
                # Get image embeddings using the model's encoder
                if hasattr(self.model, 'get_image_embeddings'):
                    return self.model.get_image_embeddings(pixel_values)
                elif hasattr(self.model, 'vision_encoder'):
                    return self.model.vision_encoder(pixel_values)
                else:
                    # Fallback: use the full model but only return embeddings
                    outputs = self.model(pixel_values=pixel_values)
                    return outputs.image_embeddings if hasattr(outputs, 'image_embeddings') else outputs[0]

        encoder_wrapper = SimpleEncoderWrapper(model).to(device).eval()
        dummy_frame = torch.randn(1, 3, 1024, 1024).to(device)

        with torch.no_grad():
            encoder_output = encoder_wrapper(dummy_frame)
            print(f"  Alternative encoder output shape: {encoder_output.shape}")

        encoder_path = output_dir / "sam3_video_encoder.onnx"
        torch.onnx.export(
            encoder_wrapper,
            (dummy_frame,),
            encoder_path,
            input_names=["pixel_values"],
            output_names=["features"],
            dynamic_axes={
                "pixel_values": {0: "batch", 2: "height", 3: "width"},
                "features": {0: "batch"}
            },
            opset_version=17,
            do_constant_folding=True,
            verbose=False
        )

        print(f"✓ Alternative encoder exported to: {encoder_path}")

    except Exception as e2:
        print(f"✗ Alternative export also failed: {e2}")
        print("\nPlease check the SAM3VideoModel architecture.")
        raise


# ============================================================================
# Export Component 2: Tracker (Simplified)
# ============================================================================

print("\n[4/5] Exporting Video Tracker...")

print("⚠ NOTE: Full tracker export with memory management is complex.")
print("  For Phase 1, we will manage memory in Python and export only")
print("  the mask prediction component.")

try:
    # For now, we'll export a simplified mask decoder
    # that takes image embeddings and text embeddings

    class SimplifiedTrackerWrapper(torch.nn.Module):
        def __init__(self, sam3_model):
            super().__init__()
            self.model = sam3_model

        def forward(self, image_embeddings, input_ids, attention_mask):
            """
            Simplified tracker that predicts masks from embeddings.

            Args:
                image_embeddings: [B, D, H', W'] - Image features
                input_ids: [B, N] - Text token IDs
                attention_mask: [B, N] - Attention mask

            Returns:
                pred_masks: [B, Q, H, W] - Predicted masks
            """
            outputs = self.model(
                image_embeddings=image_embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Return masks
            if hasattr(outputs, 'pred_masks'):
                masks = torch.sigmoid(outputs.pred_masks)
            else:
                masks = outputs[0]

            return masks

    tracker_wrapper = SimplifiedTrackerWrapper(model).to(device).eval()

    # Create dummy inputs
    # These shapes are estimates and may need adjustment
    dummy_embeddings = torch.randn(1, 256, 64, 64).to(device)  # [B, D, H', W']
    dummy_input_ids = torch.randint(0, 1000, (1, 77)).to(device)  # [B, N]
    dummy_attention_mask = torch.ones(1, 77).to(device)  # [B, N]

    # Test forward pass
    with torch.no_grad():
        tracker_output = tracker_wrapper(dummy_embeddings, dummy_input_ids, dummy_attention_mask)
        print(f"  Tracker output shape: {tracker_output.shape}")

    tracker_path = output_dir / "sam3_video_tracker.onnx"

    torch.onnx.export(
        tracker_wrapper,
        (dummy_embeddings, dummy_input_ids, dummy_attention_mask),
        tracker_path,
        input_names=["image_embeddings", "input_ids", "attention_mask"],
        output_names=["pred_masks"],
        dynamic_axes={
            "image_embeddings": {0: "batch"},
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "pred_masks": {0: "batch", 1: "num_objects"}
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False
    )

    print(f"✓ Tracker exported to: {tracker_path}")
    print(f"  Size: {tracker_path.stat().st_size / (1024**2):.2f} MB")

except Exception as e:
    print(f"⚠ Tracker export encountered issues: {e}")
    print("\nThis is expected for Phase 1 implementation.")
    print("We will use the full model with Python-based memory management.")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("[5/5] Export Summary")
print("=" * 60)

exported_files = list(output_dir.glob("*.onnx"))
if exported_files:
    print(f"\n✓ Successfully exported {len(exported_files)} component(s):")
    for file in exported_files:
        size_mb = file.stat().st_size / (1024**2)
        print(f"  • {file.name} ({size_mb:.2f} MB)")

    print("\nNext steps:")
    print("1. Build TensorRT engines:")
    print("   bash scripts/build_video_engines.sh")
    print("\n2. Implement Python Session Manager with memory management")
    print("\n3. Run video tracking demo:")
    print("   python examples/video_tracking_demo.py")
else:
    print("\n⚠ No ONNX files were exported successfully.")
    print("\nFor Phase 1, we will proceed with a hybrid approach:")
    print("1. Use PyTorch for the full model")
    print("2. Implement memory management in Python")
    print("3. Optimize critical paths with TensorRT in Phase 2")

print("\n" + "=" * 60)
print("Export process completed!")
print("=" * 60)
