install tensorrt runtime in your env with: pip install tensorrt-cu12==10.16.1.11
DOCKER_IMG="nvcr.io/nvidia/tensorrt:26.05-py3"
DOCKER_WORKDIR="/workspace"
ONNX_PATH=$DOCKER_WORKDIR"/weights/da3_metric_644x490_large.onnx"
TRT_PATH=$DOCKER_WORKDIR"/weights/da3_metric_644x490_large.engine"
docker run --gpus all -it --rm \
    -v /home/chuong/workspace/depth_models/Depth-Anything-3/weights:$DOCKER_WORKDIR/weights \
    -v /home/chuong/workspace/depth_models/Depth-Anything-3/tools:$DOCKER_WORKDIR \
    -w $DOCKER_WORKDIR $DOCKER_IMG bash -c "python export_trt.py $ONNX_PATH --trt_path $TRT_PATH --precision fp16"