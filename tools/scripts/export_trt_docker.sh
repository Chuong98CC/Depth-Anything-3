#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX checkpoint inside the NVIDIA TensorRT container.
#
# Usage:
#   ./tools/scripts/export_trt_docker.sh <onnx_ckpt_path> [precision]
#
# Arguments:
#   onnx_ckpt_path    Absolute path to the ONNX file on the host,
#                     e.g. /home/chuong/workspace/depth_models/Depth-Anything-3/weights/da3_metric_644x490_large.onnx
#   precision         Optional TRT precision: fp16 (default), tf32, or fp32
#
# Example:
#   ./tools/scripts/export_trt_docker.sh \
#       /home/chuong/workspace/depth_models/Depth-Anything-3/weights/da3_metric_644x490_large.onnx fp16
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <onnx_ckpt_path> [precision]" >&2
    exit 1
fi

ONNX_CKPT_PATH="$1"
PRECISION="${2:-fp16}"

if [ ! -f "$ONNX_CKPT_PATH" ]; then
    echo "ONNX file not found: $ONNX_CKPT_PATH" >&2
    exit 1
fi

HOST_WEIGHTS_DIR="$(cd "$(dirname "$ONNX_CKPT_PATH")" && pwd)"
ONNX_NAME="$(basename "$ONNX_CKPT_PATH")"


# install tensorrt runtime in your env with: pip install tensorrt-cu12==10.16.1.11
DOCKER_IMG="nvcr.io/nvidia/tensorrt:26.05-py3"
DOCKER_WORKDIR="/workspace"

# Host tools dir is the parent of this script's directory (tools/scripts/ -> tools/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_TOOLS_DIR="$(dirname "$SCRIPT_DIR")"

ONNX_PATH="$DOCKER_WORKDIR/weights/$ONNX_NAME"
TRT_PATH="$DOCKER_WORKDIR/weights/${ONNX_NAME%.onnx}.engine"

docker run --gpus all -it --rm \
    -v "$HOST_WEIGHTS_DIR":$DOCKER_WORKDIR/weights \
    -v "$HOST_TOOLS_DIR":$DOCKER_WORKDIR \
    -w $DOCKER_WORKDIR $DOCKER_IMG \
    bash -c "python export_trt.py $ONNX_PATH --trt_path $TRT_PATH --precision $PRECISION"
