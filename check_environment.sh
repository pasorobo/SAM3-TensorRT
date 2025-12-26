#!/bin/bash
echo "======================================================================"
echo "SAM3-TensorRT Environment Check"
echo "======================================================================"
echo ""

# GPU確認
echo -n "NVIDIA GPU:           "
nvidia-smi > /dev/null 2>&1 && echo "✓ Available" || echo "✗ Not found"

# CUDA確認
echo -n "CUDA Toolkit:         "
if nvcc --version > /dev/null 2>&1; then
    version=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
    echo "✓ $version"
else
    echo "✗ Not installed"
fi

# TensorRT確認
echo -n "TensorRT (trtexec):   "
trtexec --version > /dev/null 2>&1 && echo "✓ Available" || echo "✗ Not found"

echo ""
echo "Python Dependencies:"
echo "----------------------------------------------------------------------"

# Python依存関係
echo -n "PyTorch:              "
python3 -c "import torch; print(f'✓ {torch.__version__}')" 2>/dev/null || echo "✗ Not installed"

echo -n "Transformers:         "
python3 -c "import transformers; print(f'✓ {transformers.__version__}')" 2>/dev/null || echo "✗ Not installed"

echo -n "TensorRT Python:      "
python3 -c "import tensorrt; print(f'✓ {tensorrt.__version__}')" 2>/dev/null || echo "✗ Not installed"

echo -n "PyCUDA:               "
python3 -c "import pycuda; print(f'✓ Installed')" 2>/dev/null || echo "✗ Not installed"

echo -n "OpenCV:               "
python3 -c "import cv2; print(f'✓ {cv2.__version__}')" 2>/dev/null || echo "✗ Not installed"

echo -n "NumPy:                "
python3 -c "import numpy; print(f'✓ {numpy.__version__}')" 2>/dev/null || echo "✗ Not installed"

echo ""
echo "SAM3-TensorRT Module:"
echo "----------------------------------------------------------------------"

# SAM3-TensorRTモジュール
echo -n "Module Status:        "
python3 -c "from sam3_video import get_version_info; info=get_version_info(); print(f'✓ v{info[\"version\"]} (Phase {info[\"phase\"]})')" 2>/dev/null || echo "✗ Import failed"

echo -n "TensorRT Support:     "
python3 -c "from sam3_video import get_version_info; info=get_version_info(); print('✓ Enabled' if info['tensorrt_support'] else '✗ Disabled')" 2>/dev/null || echo "✗ Unknown"

echo ""
echo "======================================================================"
echo "Summary"
echo "======================================================================"

# サマリー判定
gpu_ok=$(nvidia-smi > /dev/null 2>&1 && echo "yes" || echo "no")
cuda_ok=$(nvcc --version > /dev/null 2>&1 && echo "yes" || echo "no")
trt_ok=$(trtexec --version > /dev/null 2>&1 && echo "yes" || echo "no")
torch_ok=$(python3 -c "import torch" 2>/dev/null && echo "yes" || echo "no")
module_ok=$(python3 -c "from sam3_video import get_version_info" 2>/dev/null && echo "yes" || echo "no")

if [ "$module_ok" = "yes" ]; then
    echo "✓ SAM3-TensorRT module is ready"
    
    if [ "$gpu_ok" = "yes" ] && [ "$cuda_ok" = "yes" ] && [ "$trt_ok" = "yes" ] && [ "$torch_ok" = "yes" ]; then
        echo "✓ Full TensorRT acceleration available"
        echo ""
        echo "Next steps:"
        echo "  1. python onnxexport_video.py"
        echo "  2. bash scripts/build_video_engines.sh fp16"
        echo "  3. python examples/video_tracking_demo_trt.py --video test.mp4 --prompt 'person'"
    elif [ "$torch_ok" = "yes" ]; then
        echo "⚠ PyTorch mode only (TensorRT not available)"
        echo ""
        echo "Available features:"
        echo "  - Video tracking with PyTorch (6-10 FPS)"
        echo ""
        echo "To enable TensorRT (2-4x speedup):"
        echo "  1. Install NVIDIA GPU drivers and CUDA"
        echo "  2. Install TensorRT"
        echo "  3. See SETUP_GUIDE.md for details"
    else
        echo "⚠ Dependencies missing"
        echo ""
        echo "Install dependencies:"
        echo "  pip install -r requirements.txt"
    fi
else
    echo "✗ SAM3-TensorRT module import failed"
    echo ""
    echo "Check installation:"
    echo "  cd SAM3-TensorRT"
    echo "  python3 -c 'from sam3_video import get_version_info; print(get_version_info())'"
fi

echo "======================================================================"
