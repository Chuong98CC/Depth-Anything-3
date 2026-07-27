ROOT_DIR=/home/chuong/workspace/point_models
MODEL_ENGINE=$ROOT_DIR/molmo-motion/checkpoints/DA3/da3metric_large_644x490_fp16.engine
CAMERA_CALIB=$ROOT_DIR/molmo-motion/data_generation/astribot_camera_calib_params/astribot_calib_head_rgbd.json
VIDEO_PATH=~/workspace/demo_data/astribot_demo.mp4
OUTPUT_DIR=$ROOT_DIR/molmo-motion/output/astribot_demo_depth
cd $ROOT_DIR/molmo-motion/data_generation/depth_model
python infer_depth.py \
    --trt_engine $MODEL_ENGINE \
    --calib $CAMERA_CALIB \
    --video $VIDEO_PATH \
    --camera head_rgbd \
    --start_frame 0 --end_frame 100 --frame_interval 20 \
    --output $OUTPUT_DIR \
    --save_depth \
    --save_viz