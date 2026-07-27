# install tensorrt runtime in your env with: pip install tensorrt-cu12==10.16.1.11
DOCKER_IMG="nvcr.io/nvidia/tensorrt:26.05-py3"
DOCKER_WORKDIR="/workspace"
ONNX_PATH=$DOCKER_WORKDIR"/weights/da3metric_large_644x490.onnx"
TRT_PATH=$DOCKER_WORKDIR"/weights/da3metric_large_644x490.engine"
docker run --gpus all -it --rm \
    -v /home/chuong/workspace/point_models/molmo-motion/checkpoints/DA3:$DOCKER_WORKDIR/weights \
    -v /home/chuong/workspace/point_models/molmo-motion/data_generation/depth_model:$DOCKER_WORKDIR \
    -w $DOCKER_WORKDIR $DOCKER_IMG bash -c "python export_trt.py $ONNX_PATH --trt_path $TRT_PATH --precision fp16"