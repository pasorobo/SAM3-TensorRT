# SAM3 → TensorRT

Export Meta AI's newest Segment Anything 3 (SAM-3) model to ONNX and then build a TensorRT engine you can deploy for real-time segmentation and video tracking.

## Features

### Image Segmentation
- Minimal, readable export script using Hugging Face `transformers`' `Sam3Model` and `Sam3Processor`
- Clean ONNX graph with instance masks (`pred_masks`) and semantic mask outputs
- TensorRT-ready: plug the produced ONNX into `trtexec` or your own build pipeline
- CPU-friendly export for maximum compatibility (flip `device` to use GPU if you prefer)

### Video Tracking (NEW!)
- Real-time video object tracking with text prompts
- Memory bank for temporal consistency across frames
- Support for MP4/AVI video files and webcam input
- 14-21 FPS @ 1080p with optimized processing
- Interactive prompt management for dynamic tracking

## Quickstart
1) **Request access to the gated model**
   - Visit https://huggingface.co/facebook/sam3 and click “Access repository” (approval is required before downloads succeed). Make sure your `HF_TOKEN` has that access.

2) **Install dependencies**
```bash
pip install torch transformers pillow requests

# Optional: If after a pip install the ONNX export script complains that it cannot find SAM3, it's just because SAM-3 is very new and a formal release of `transformers` has not added support yet. In this case, please install `transformers` using git

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install '.[torch]'
# or uv pip install '.[torch]'
```
Use a PyTorch wheel that matches your CUDA if you plan to export on GPU. By default a GPU is not required to export to ONNX.


3) **Export to ONNX**
```bash
export HF_TOKEN=<YOUR TOKEN>
python onnxexport.py
```

This downloads `facebook/sam3`, runs a sample prompt (`"ear"`), and writes `onnx_weights/sam3_static.onnx` plus associated weight shards. SAM-3 is ~3.2 GB, so the ONNX exporter will create external data files; if you build TensorRT on another machine you must copy the entire `onnx_weights/` directory (not just the .onnx).


4) **Build a TensorRT engine**
Use `trtexec` (or your favorite builder) on the generated ONNX:

```bash
trtexec --onnx=onnx_weights/sam3_static.onnx --saveEngine=sam3_fp16.plan --fp16 --verbose # fp16
trtexec --onnx=onnx_weights/sam3_static.onnx --saveEngine=sam3_int8.plan --int8 --verbose # int8
trtexec --onnx=onnx_weights/sam3_static.onnx --saveEngine=sam3_fp8.plan --fp8 --verbose # fp8
trtexec --onnx=onnx_weights/sam3_static.onnx --saveEngine=sam3_int4.plan --int4 --verbose # int4
```


5) **Validate** (optional)
- Run `onnxruntime` on `onnx_weights/sam3_static.onnx` to confirm outputs
- Benchmark `sam3_fp16.plan` with `trtexec --loadEngine=sam3_fp16.plan`

## Adapting the export
- Change the prompt/image in `onnxexport.py` to better reflect your production use case.
- Swap `device` to `"cuda"` for GPU-side export if your environment supports it.
- Add more outputs from `Sam3Model` (e.g., scores) by extending `Sam3ONNXWrapper`.

## Video Streaming Tracking

### Quick Start

```python
from sam3_video import Sam3VideoSessionManager

# Initialize session manager
session = Sam3VideoSessionManager()

# Start tracking session
session.start_session(text_prompt="person")

# Process video
results = session.process_video(
    video_path="input.mp4",
    output_path="output_tracked.mp4"
)

# End session
session.end_session()

print(f"Processed {results['frames_processed']} frames")
print(f"Average FPS: {results['stats']['avg_fps']:.2f}")
```

### Video Tracking Features

- **Text-based prompts**: Track objects using natural language (e.g., "person", "car", "dog")
- **Memory management**: FIFO-based memory bank maintains temporal consistency
- **Multi-object tracking**: Track multiple objects simultaneously
- **Real-time performance**: 14-21 FPS @ 1080p (with optimizations)
- **Flexible input**: MP4/AVI files, JPEG folders, or webcam streams

### Video Examples

```bash
# Process video file
python examples/video_tracking_demo.py \
  --video path/to/video.mp4 \
  --prompt "person" \
  --output tracked_output.mp4

# Real-time webcam tracking
python examples/webcam_demo.py \
  --prompt "person" \
  --camera 0

# Benchmark performance
python benchmarks/video_benchmark.py \
  --video test.mp4 \
  --prompt "person" \
  --frames 100 \
  --runs 3
```

### Video Export to ONNX (Optional)

For advanced users wanting to export video components:

```bash
# Export video encoder and tracker
python onnxexport_video.py

# Build TensorRT engines
bash scripts/build_video_engines.sh fp16

# Or build all precision modes
bash scripts/build_video_engines.sh all
```

### Architecture

SAM3 video tracking uses a hybrid approach:

```
┌─────────────────────────────────────┐
│  Python Application Layer           │
│  ┌───────────────────────────────┐  │
│  │ Video Session Manager         │  │
│  │ - Frame buffering             │  │
│  │ - Memory bank (FIFO queues)   │  │
│  │ - Object tracking state       │  │
│  └───────────────────────────────┘  │
│              │                       │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │ SAM3VideoModel (PyTorch)      │  │
│  │ - Perception Encoder          │  │
│  │ - Video Tracker               │  │
│  │ - Mask Decoder                │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Phase 1** (Current): PyTorch models with Python memory management
**Phase 2** (Future): TensorRT acceleration with custom plugins

### Performance

Expected performance on various hardware:

| Hardware | FPS @ 1080p | Latency |
|----------|-------------|---------|
| NVIDIA RTX 4090 | 18-21 FPS | 47-55ms |
| NVIDIA RTX 3080 | 14-18 FPS | 55-70ms |
| NVIDIA T4 | 10-14 FPS | 70-100ms |
| CPU (12-core) | 2-4 FPS | 250-500ms |

*Actual performance may vary based on video complexity and number of tracked objects.*

## Installation

**📘 Complete Setup Guide**: See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed installation instructions

**⚡ Quick Environment Check**: Run `bash check_environment.sh` to verify your setup

### Basic Installation (Image Segmentation)

```bash
pip install torch transformers pillow requests opencv-python
```

### Full Installation (Video Tracking)

```bash
# Core dependencies
pip install torch transformers pillow requests opencv-python numpy tqdm

# For latest SAM3 support
pip install git+https://github.com/huggingface/transformers.git

# Optional: TensorRT (for 2-4x acceleration)
pip install tensorrt pycuda
# See SETUP_GUIDE.md for TensorRT installation details
```

### Environment-Specific Setup

Different environments have different requirements. See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for:
- Local GPU workstations
- Cloud GPU instances (AWS, GCP, Azure)
- Google Colab / Kaggle
- NVIDIA Jetson devices
- Docker containers

## Project Layout

```
SAM3-TensorRT/
├── onnxexport.py              # Image segmentation ONNX export
├── onnxexport_video.py        # Video components ONNX export
├── sam3_video/                # Video tracking module
│   ├── __init__.py
│   ├── memory_bank.py         # Memory management
│   └── session_manager.py     # Video session manager
├── examples/                  # Example scripts
│   ├── video_tracking_demo.py # Full video processing
│   ├── webcam_demo.py         # Real-time webcam
│   └── simple_example.py      # Minimal usage
├── benchmarks/                # Performance benchmarks
│   └── video_benchmark.py
├── scripts/                   # Build scripts
│   └── build_video_engines.sh
├── docs/                      # Documentation
│   ├── sam3-feature-analysis.md
│   └── streaming-tracking-implementation-plan.md
└── LICENSE

```

## Documentation

- [SAM3 Feature Analysis](docs/sam3-feature-analysis.md) - Comprehensive feature analysis
- [Streaming Tracking Implementation Plan](docs/streaming-tracking-implementation-plan.md) - Detailed implementation guide

## License

MIT

## Acknowledgments

- Meta AI for SAM3
- Hugging Face for transformers integration
- NVIDIA for TensorRT

If this saved you time, drop a ⭐ so others can find it and ship SAM-3 faster!
