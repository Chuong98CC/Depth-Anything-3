#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX checkpoint inside the NVIDIA TensorRT container.
#
# Usage:
#   ./tools/scripts/export_trt_docker.sh <onnx_ckpt_name> <host_weights_dir> [precision]
#
# Arguments:
#   onnx_ckpt_name    Name of the ONNX file inside the weights dir, e.g. da3_metric_644x490_large.onnx
#   host_weights_dir  Absolute path to the weights folder on the host,
#                     e.g. /home/chuong/workspace/depth_models/Depth-Anything-3/weights
#   precision         Optional TRT precision: fp16 (default), tf32, or fp32
#
# Example:
#   ./tools/scripts/export_trt_docker.sh da3_metric_644x490_large.onnx \
#       /home/chuong/workspace/depth_models/Depth-Anything-3/weights fp16
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <onnx_ckpt_name> <host_weights_dir> [precision]" >&2
    exit 1
fi

ONNX_NAME="$1"
HOST_WEIGHTS_DIR="$2"


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
    bash -c "python export_trt.py $ONNX_PATH --trt_path $TRT_PATH --precision fp16"
