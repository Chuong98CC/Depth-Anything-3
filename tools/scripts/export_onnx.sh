
python tools/export_onnx.py \
    --model-dir depth-anything/DA3-SMALL \
    --wrapper anyview --num-views 3 --height 490 --width 644 \
    --onnx-path weights/da3_anyview_n3_644x490_small.onnx \
    --check-accuracy

# python tools/export_onnx.py \
#     --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1\
#     --wrapper anyview --num-views 3 --height 490 --width 644 \
#     --onnx-path weights/da3_anyview_n3_644x490_giant-large-1.1.onnx \
#     --check-accuracy

# python tools/export_onnx.py \
#     --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
#     --wrapper metric --height 490 --width 644 \
#     --onnx-path weights/da3_metric_644x490_giant-large-1.1.onnx \
#     --check-accuracy