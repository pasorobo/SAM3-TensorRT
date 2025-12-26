#!/bin/bash
#
# SAM3 Video TensorRT Engine Builder
#
# Builds TensorRT engines from ONNX models with various precision modes.
#
# Usage:
#   bash scripts/build_video_engines.sh [fp16|int8|fp8|int4|all]
#
# Author: Claude Code
# Date: 2025-12-26

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ONNX_DIR="onnx_weights"
ENGINE_DIR="trt_engines"
ENCODER_ONNX="${ONNX_DIR}/sam3_video_encoder.onnx"
TRACKER_ONNX="${ONNX_DIR}/sam3_video_tracker.onnx"

# Create engine directory
mkdir -p "${ENGINE_DIR}"

echo "======================================================================"
echo "SAM3 Video TensorRT Engine Builder"
echo "======================================================================"
echo ""

# Check if ONNX files exist
if [ ! -f "${ENCODER_ONNX}" ]; then
    echo -e "${RED}✗ Error: Encoder ONNX not found at ${ENCODER_ONNX}${NC}"
    echo "Please run: python onnxexport_video.py"
    exit 1
fi

echo -e "${GREEN}✓ Found encoder ONNX: ${ENCODER_ONNX}${NC}"

if [ -f "${TRACKER_ONNX}" ]; then
    echo -e "${GREEN}✓ Found tracker ONNX: ${TRACKER_ONNX}${NC}"
    HAS_TRACKER=true
else
    echo -e "${YELLOW}⚠ Tracker ONNX not found (optional for Phase 1)${NC}"
    HAS_TRACKER=false
fi

echo ""

# Parse arguments
MODE="${1:-fp16}"

build_encoder() {
    local precision=$1
    local engine_name="sam3_video_encoder_${precision}.plan"
    local engine_path="${ENGINE_DIR}/${engine_name}"

    echo "----------------------------------------------------------------------"
    echo "Building Encoder Engine: ${precision}"
    echo "----------------------------------------------------------------------"

    # Common arguments
    local common_args=(
        --onnx="${ENCODER_ONNX}"
        --saveEngine="${engine_path}"
        --verbose
        --workspace=4096
    )

    # Precision-specific arguments
    case $precision in
        fp16)
            common_args+=(--fp16)
            ;;
        int8)
            common_args+=(--int8)
            echo -e "${YELLOW}Note: INT8 requires calibration for best results${NC}"
            ;;
        fp8)
            common_args+=(--fp8)
            ;;
        int4)
            common_args+=(--int4)
            ;;
        *)
            echo -e "${RED}Unknown precision: ${precision}${NC}"
            return 1
            ;;
    esac

    # Dynamic shape configuration
    # Adjust these based on your use case
    common_args+=(
        --minShapes=pixel_values:1x3x512x512
        --optShapes=pixel_values:1x3x1024x1024
        --maxShapes=pixel_values:1x3x1920x1080
    )

    # Build engine
    echo "Running trtexec..."
    if trtexec "${common_args[@]}"; then
        echo -e "${GREEN}✓ Successfully built: ${engine_path}${NC}"

        # Display file size
        if [ -f "${engine_path}" ]; then
            local size=$(du -h "${engine_path}" | cut -f1)
            echo "  Size: ${size}"
        fi
    else
        echo -e "${RED}✗ Failed to build ${engine_name}${NC}"
        return 1
    fi

    echo ""
}

build_tracker() {
    local precision=$1
    local engine_name="sam3_video_tracker_${precision}.plan"
    local engine_path="${ENGINE_DIR}/${engine_name}"

    if [ "${HAS_TRACKER}" = false ]; then
        echo -e "${YELLOW}Skipping tracker (ONNX not available)${NC}"
        return 0
    fi

    echo "----------------------------------------------------------------------"
    echo "Building Tracker Engine: ${precision}"
    echo "----------------------------------------------------------------------"

    local common_args=(
        --onnx="${TRACKER_ONNX}"
        --saveEngine="${engine_path}"
        --verbose
        --workspace=4096
    )

    case $precision in
        fp16)
            common_args+=(--fp16)
            ;;
        int8)
            common_args+=(--int8)
            ;;
        fp8)
            common_args+=(--fp8)
            ;;
        int4)
            common_args+=(--int4)
            ;;
    esac

    # Dynamic shapes for tracker (adjust as needed)
    common_args+=(
        --minShapes=image_embeddings:1x256x32x32,input_ids:1x32,attention_mask:1x32
        --optShapes=image_embeddings:1x256x64x64,input_ids:1x77,attention_mask:1x77
        --maxShapes=image_embeddings:1x256x128x128,input_ids:1x128,attention_mask:1x128
    )

    echo "Running trtexec..."
    if trtexec "${common_args[@]}"; then
        echo -e "${GREEN}✓ Successfully built: ${engine_path}${NC}"

        if [ -f "${engine_path}" ]; then
            local size=$(du -h "${engine_path}" | cut -f1)
            echo "  Size: ${size}"
        fi
    else
        echo -e "${RED}✗ Failed to build ${engine_name}${NC}"
        return 1
    fi

    echo ""
}

# Build based on mode
case $MODE in
    fp16)
        build_encoder fp16
        build_tracker fp16
        ;;
    int8)
        build_encoder int8
        build_tracker int8
        ;;
    fp8)
        build_encoder fp8
        build_tracker fp8
        ;;
    int4)
        build_encoder int4
        build_tracker int4
        ;;
    all)
        echo "Building all precision modes..."
        echo ""
        build_encoder fp16
        build_tracker fp16
        build_encoder int8
        build_tracker int8
        build_encoder fp8
        build_tracker fp8
        ;;
    *)
        echo -e "${RED}Unknown mode: ${MODE}${NC}"
        echo "Usage: $0 [fp16|int8|fp8|int4|all]"
        exit 1
        ;;
esac

echo "======================================================================"
echo "Build Summary"
echo "======================================================================"
echo ""
echo "TensorRT engines built in: ${ENGINE_DIR}/"
ls -lh "${ENGINE_DIR}"/*.plan 2>/dev/null || echo "No engines found"
echo ""
echo "======================================================================"
echo "Next Steps:"
echo "======================================================================"
echo "1. Test the engines:"
echo "   python examples/video_tracking_demo.py --video test.mp4 --prompt 'person'"
echo ""
echo "2. Benchmark performance:"
echo "   python benchmarks/video_benchmark.py"
echo ""
echo "======================================================================"
