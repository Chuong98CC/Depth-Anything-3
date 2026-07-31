
NUM_CAMS=3
HEIGHT=490
WIDTH=644
CKPT=depth-anything/DA3NESTED-GIANT-LARGE-1.1

# CKPT_SMALL=depth-anything/DA3-SMALL
#
# python tools/export_onnx.py \
#     --model-dir $CKPT_SMALL \
#     --wrapper anyview --num-views $NUM_CAMS --height $HEIGHT --width $WIDTH \
#     --onnx-path weights/da3_anyview_n${NUM_CAMS}_${WIDTH}x${HEIGHT}_small.onnx \
#     --check-accuracy

# Any-view, plain: input is `image` only; the model predicts its own camera poses.
python tools/export_onnx.py \
    --model-dir $CKPT \
    --wrapper anyview --num-views $NUM_CAMS --height $HEIGHT --width $WIDTH \
    --onnx-path weights/da3_anyview_n${NUM_CAMS}_${WIDTH}x${HEIGHT}_giant-large-1.1.onnx \
    --check-accuracy

# Any-view, with camera pose: --use-extrinsics adds `extrinsics`/`intrinsics`
# inputs (fed to the camera encoder as priors). Note the "-with-camera-pose" suffix
# on the output path — the inference/comparison scripts select this file when you
# pass their own --use-extrinsics flag.
# python tools/export_onnx.py \
#     --model-dir $CKPT \
#     --wrapper anyview --num-views $NUM_CAMS --height $HEIGHT --width $WIDTH --use-extrinsics \
#     --onnx-path weights/da3_anyview_n${NUM_CAMS}_${WIDTH}x${HEIGHT}_giant-large-1.1-with-camera-pose.onnx \
#     --check-accuracy

# python tools/export_onnx.py \
#     --model-dir $CKPT \
#     --wrapper metric --height $HEIGHT --width $WIDTH \
#     --onnx-path weights/da3_metric_${WIDTH}x${HEIGHT}_giant-large-1.1.onnx \
#     --check-accuracy