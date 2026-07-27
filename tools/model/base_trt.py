import numpy as np
import cv2
import tensorrt as trt
import torch
from abc import ABC, abstractmethod

# Convert numpy dtype name to torch dtype
def trt_to_torch_dtype(trt_dtype):
    dtype_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.UINT8: torch.uint8,
    }
    return dtype_map.get(trt_dtype, torch.float32)

class TRTModel(ABC):
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.context = None
        self.engine = None

        # Load the engine
        print(f"Loading engine from: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(
                f"Failed to load TensorRT engine from {engine_path}.\n"
                f"The engine is incompatible with TensorRT {trt.__version__}.\n"
                f"You need to rebuild the engine file with your current TensorRT version:\n"
                f"  python demo/export_tensorrt.py --model_type S --img_width 640 --img_height 480 --precision fp16"
            )

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Store tensor info for TensorRT 10.3+ (no manual memory allocation needed)
        self.inputs = []
        self.outputs = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = self.engine.get_tensor_dtype(tensor_name)

            tensor_info = {
                'name': tensor_name,
                'shape': shape,
                'dtype': dtype
            }

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor_info)
                print(f"Input: {tensor_name}, shape: {shape}, dtype: {dtype}")
            else:
                self.outputs.append(tensor_info)
                print(f"Output: {tensor_name}, shape: {shape}, dtype: {dtype}")

        self.target_w = self.inputs[0]['shape'][3]  # Assuming NCHW
        self.target_h = self.inputs[0]['shape'][2]

    @property
    def input_width(self):
        return self.target_w

    @property
    def input_height(self):
        return self.target_h

    def resize_img(self, img: np.ndarray):
        # Calculate uniform scale factor to maintain aspect ratio
        orig_h, orig_w = img.shape[:2]
        scale_w = self.target_w / orig_w
        scale_h = self.target_h / orig_h
        # Keep 2-decimal precision by truncating down so leftover fit is handled by padding.
        raw_scale = min(scale_w, scale_h)  # Use minimum to ensure image fits
        scale_factor = np.floor(raw_scale * 100.0) / 100.0
        if scale_factor <= 0:
            # Guard against degenerate tiny scales after truncation.
            scale_factor = raw_scale
        
        # Resize with uniform scale factor
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        img_resized = cv2.resize(img, (new_w, new_h))

        # Pad to target size (center the image)
        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "scale_factor": float(scale_factor),
            "tile_h": new_h,
            "tile_w": new_w,
            "pad_top": int(pad_top),
            "pad_left": int(pad_left)
        }
        return img_padded, meta

    def img2tensor(self, img: np.ndarray):
        # Get expected input dtype
        input_dtype = trt_to_torch_dtype(self.inputs[0]['dtype'])

        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).contiguous()

        # Move to GPU and convert to expected dtype
        tensor = tensor.cuda().to(input_dtype)
        return tensor

    def output2numpy(self, output_tensors: dict):
        """Convert TensorRT output tensors to numpy arrays."""
        # Convert outputs to numpy
        output_data = {}
        for name, tensor in output_tensors.items():
            output_data[name] = tensor.cpu().numpy()

        return output_data

    def __del__(self):
        """Clean up TensorRT resources"""
        if hasattr(self, 'context') and self.context is not None:
            del self.context
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine

    @abstractmethod
    def _infer(self, *args, **kwargs):
        """Run inference on a pair of stereo images. Must be implemented by subclasses. """
        pass

    def parse_outputs(self, *args, **kwargs):
        """Extract, crop and mask model outputs for a single frame. Must be implemented by subclasses."""
        pass

class MonoDepthTRT(TRTModel):
    def _infer(self, img: np.ndarray, np_output: bool = True):
        """
        TensorRT 10.3+ API: Direct tensor transfer without manual memory management
        Input: uint8 numpy array (0-255) in RGB format
        """
        # Ensure input image is uint8
        input_tensor = self.img2tensor(img)

        # Set input tensor directly (TensorRT 10.3+ handles memory internally)
        self.context.set_input_shape(self.inputs[0]['name'], tuple(input_tensor.shape))
        self.context.set_tensor_address(self.inputs[0]['name'], input_tensor.data_ptr())

        # Allocate output tensors
        output_tensors = {}
        for out in self.outputs:
            output_shape = self.context.get_tensor_shape(out['name'])
            output_dtype = trt_to_torch_dtype(out['dtype'])
            output_tensor = torch.empty(tuple(output_shape), dtype=output_dtype, device='cuda').contiguous()
            output_tensors[out['name']] = output_tensor
            self.context.set_tensor_address(out['name'], output_tensor.data_ptr())

        # Execute inference
        stream = torch.cuda.current_stream()
        success = self.context.execute_async_v3(stream.cuda_stream)
        if not success:
            raise RuntimeError("TensorRT inference execution failed")
        torch.cuda.synchronize()

        if np_output:
            return self.output2numpy(output_tensors)
        else:
            return output_tensors

class StereoDepthTRT(TRTModel):
    def _infer(self, left_img: np.ndarray, right_img: np.ndarray, np_output: bool = True):
        """
        TensorRT 10.3+ API: Direct tensor transfer without manual memory management
        Input: uint8 numpy arrays (0-255) in RGB format
        """

        # Ensure input images are uint8
        left_tensor = self.img2tensor(left_img)
        right_tensor = self.img2tensor(right_img)

        # Set input tensors directly (TensorRT 10.3+ handles memory internally)
        if len(self.inputs) == 2:
            # Model has separate left and right inputs
            self.context.set_input_shape(self.inputs[0]['name'], tuple(left_tensor.shape))
            self.context.set_input_shape(self.inputs[1]['name'], tuple(right_tensor.shape))
            self.context.set_tensor_address(self.inputs[0]['name'], left_tensor.data_ptr())
            self.context.set_tensor_address(self.inputs[1]['name'], right_tensor.data_ptr())
        else:
            # Model expects concatenated input (6 channels)
            input_tensor = torch.cat([left_tensor, right_tensor], dim=1).contiguous()
            self.context.set_input_shape(self.inputs[0]['name'], tuple(input_tensor.shape))
            self.context.set_tensor_address(self.inputs[0]['name'], input_tensor.data_ptr())

        # Allocate output tensors
        output_tensors = {}
        for out in self.outputs:
            output_shape = self.context.get_tensor_shape(out['name'])
            output_dtype = trt_to_torch_dtype(out['dtype'])
            output_tensor = torch.empty(tuple(output_shape), dtype=output_dtype, device='cuda').contiguous()
            output_tensors[out['name']] = output_tensor
            self.context.set_tensor_address(out['name'], output_tensor.data_ptr())

        # Execute inference
        stream = torch.cuda.current_stream()
        success = self.context.execute_async_v3(stream.cuda_stream)
        if not success:
            raise RuntimeError("TensorRT inference execution failed")
        torch.cuda.synchronize()

        if np_output:
            return self.output2numpy(output_tensors)
        else:
            return output_tensors

    def preprocess(self, left_img, right_img):
        left_resized, meta_info = self.resize_img(left_img)
        right_resized, _ = self.resize_img(right_img)
        return left_resized, right_resized, meta_info



