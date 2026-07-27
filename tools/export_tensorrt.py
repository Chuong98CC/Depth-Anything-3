import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_path', type=str)
    parser.add_argument('--trt_path', type=str, default=None)
    parser.add_argument('--precision', type=str, choices=['fp16', 'tf32', 'fp32'], default='fp16')
    return parser

def main(args):
    weight_folder = os.path.dirname(args.onnx_path)
    base_name = os.path.basename(args.onnx_path)
    trt_path = args.trt_path if args.trt_path else os.path.join(weight_folder, f'trt_save')
    os.makedirs(trt_path, exist_ok=True)
    trt_file_path = os.path.join(trt_path, f'{os.path.splitext(base_name)[0]}_{args.precision}.engine')

    command = f'trtexec --onnx={args.onnx_path} --saveEngine={trt_file_path}'

    if args.precision == 'fp16':
        fp_16_options = '--fp16 --precisionConstraints=obey --layerPrecisions=node_linalg_vector_norm_2:fp32'
        command += f' {fp_16_options}'
    elif args.precision == 'fp32':
        command +=' --noTF32'

    os.system(command)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)