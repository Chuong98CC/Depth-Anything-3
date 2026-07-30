
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

# python tools/export_onnx.py \
#     --model-dir $CKPT \
#     --wrapper anyview --num-views $NUM_CAMS --height $HEIGHT --width $WIDTH \
#     --onnx-path weights/da3_anyview_n${NUM_CAMS}_${WIDTH}x${HEIGHT}_giant-large-1.1.onnx \
#     --check-accuracy

python tools/export_onnx.py \
    --model-dir $CKPT \
    --wrapper metric --height $HEIGHT --width $WIDTH \
    --onnx-path weights/da3_metric_${WIDTH}x${HEIGHT}_giant-large-1.1.onnx \
    --check-accuracy