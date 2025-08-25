# wrapper.py
"""
Universal inference wrapper for FaceFusion:
supports model formats: pt, torchscript, onnx, trt (TensorRT engine).

Usage:
    from wrapper import ModelWrapper
    m = ModelWrapper(path="model.pt", fmt="pt", device="cuda", fp16=True)
    infer = m.get_infer_fn()
    out = infer(input_numpy)  # returns numpy output

Run as CLI to check missing deps and do simple random run:
    python3 wrapper.py --model model.pt --format pt --device cuda
"""
from __future__ import annotations
import sys
import os
import time
import argparse
import numpy as np

# dependency availability flags
_HAS_TORCH = False
_HAS_ONNXRUNTIME = False
_HAS_TRT = False
_HAS_PYCUDA = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import onnxruntime as ort
    _HAS_ONNXRUNTIME = True
except Exception:
    _HAS_ONNXRUNTIME = False

try:
    import tensorrt as trt  # type: ignore
    _HAS_TRT = True
except Exception:
    _HAS_TRT = False

try:
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # noqa: F401
    _HAS_PYCUDA = True
except Exception:
    _HAS_PYCUDA = False

# helpful suggestion strings
_SUGGEST = {
    "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA index as needed",
    "onnxruntime-gpu": "pip install onnxruntime-gpu",
    "tensorrt": "Install TensorRT (libnvinfer) and Python bindings from NVIDIA; wheel or apt — see NVIDIA docs",
    "pycuda": "pip install pycuda",
    "torch-tensorrt": "pip install torch-tensorrt  # optional alternative to generate TRT modules from PyTorch",
}

def check_missing_deps(formats: list[str]) -> dict[str, str]:
    """Return mapping of missing packages -> suggestion string."""
    miss = {}
    for f in formats:
        f = f.lower()
        if f in ("pt", "torchscript"):
            if not _HAS_TORCH:
                miss["torch"] = _SUGGEST["torch"]
        elif f == "onnx":
            if not _HAS_ONNXRUNTIME:
                miss["onnxruntime-gpu"] = _SUGGEST["onnxruntime-gpu"]
        elif f == "trt":
            if not _HAS_TRT:
                miss["tensorrt"] = _SUGGEST["tensorrt"]
            if not _HAS_PYCUDA:
                miss["pycuda"] = _SUGGEST["pycuda"]
    return miss

# ---------------------------
# PyTorch loader / infer
# ---------------------------
class _PTModel:
    def __init__(self, path: str, device: str = "cuda", fp16: bool = False):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not available. " + _SUGGEST["torch"])
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.fp16 = fp16
        # try torch.jit.load first (scripted), fallback to torch.load
        try:
            self.model = torch.jit.load(path, map_location=self.device)
        except Exception:
            # try state-dict or pickled module
            try:
                loaded = torch.load(path, map_location=self.device)
                if isinstance(loaded, torch.nn.Module):
                    self.model = loaded.to(self.device)
                else:
                    # maybe state_dict or plain dict -> require explicit model class (can't handle)
                    raise RuntimeError("Loaded object not nn.Module; please provide a scripted or saved Module.")
            except Exception as e:
                raise RuntimeError(f"Failed to load PyTorch model ({e})")
        self.model.eval()
        if self.fp16 and self.device.type == "cuda":
            self.model.half()

    def infer(self, x: np.ndarray) -> np.ndarray:
        # expect numpy BCHW float32/uint8
        tensor = torch.from_numpy(x)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        if self.fp16 and self.model is not None and self.device.type == "cuda":
            tensor = tensor.to(self.device).half()
        else:
            tensor = tensor.to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out_np = out.detach().cpu().numpy()
        return out_np

# ---------------------------
# ONNXRuntime loader / infer
# ---------------------------
class _ONNXModel:
    def __init__(self, path: str, device: str = "cuda"):
        if not _HAS_ONNXRUNTIME:
            raise RuntimeError("onnxruntime not available. " + _SUGGEST["onnxruntime-gpu"])
        providers = []
        # prefer Tensorrt EP then CUDA EP then CPU
        ava = ort.get_available_providers()
        if "TensorrtExecutionProvider" in ava:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in ava:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.sess = ort.InferenceSession(path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def infer(self, x: np.ndarray) -> np.ndarray:
        # if input is float in [0,255], convert
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        res = self.sess.run(None, {self.input_name: x})
        return res[0] if isinstance(res, (list, tuple)) else res

# ---------------------------
# TensorRT engine loader / infer
# ---------------------------
class _TRTModel:
    def __init__(self, engine_path: str, device: str = "cuda", fp16: bool = False):
        if not _HAS_TRT or not _HAS_PYCUDA:
            raise RuntimeError("TensorRT or PyCUDA not available. Install TensorRT + pycuda.")
        self.engine_path = engine_path
        self.fp16 = fp16
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            buf = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(buf)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TRT engine. Check engine validity and TRT version.")
        self.context = self.engine.create_execution_context()
        # create bindings and buffers
        self.bindings = []
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.stream = cuda.Stream()
        self.binding_idxs = {}
        for b in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(b)
            self.binding_idxs[name] = b
            if self.engine.binding_is_input(b):
                shape = self.engine.get_binding_shape(b)
                dtype = trt.nptype(self.engine.get_binding_dtype(b))
                # allow dynamic shapes: set binding shape later
                self.host_inputs.append(None)
                self.cuda_inputs.append(None)
            else:
                self.host_outputs.append(None)
                self.cuda_outputs.append(None)

    def _allocate_for_input(self, input_numpy: np.ndarray):
        # configure binding shapes if dynamic
        # find first input binding index
        for b in range(self.engine.num_bindings):
            if self.engine.binding_is_input(b):
                if -1 in self.engine.get_binding_shape(b):
                    # dynamic: set binding shape from numpy
                    self.context.set_binding_shape(b, tuple(input_numpy.shape))
                break
        # now allocate buffers according to binding sizes
        self.bindings = [None] * self.engine.num_bindings
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        for idx in range(self.engine.num_bindings):
            shape = self.context.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            size = int(np.prod(shape))
            host_mem = np.empty(shape, dtype=dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            if self.engine.binding_is_input(idx):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
            self.bindings[idx] = int(cuda_mem)

    def infer(self, x: np.ndarray) -> np.ndarray:
        # x is numpy array
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        # ensure proper shape ordering (TRT expects NCHW usually)
        if self.bindings is None or len(self.bindings) == 0:
            self._allocate_for_input(x)
        # copy input to host buffer (first host_inputs slot)
        self.host_inputs[0].flat[:] = x.ravel()
        # copy to device
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0].ravel().view(np.uint8), self.stream)
        # execute
        self.context.execute_async_v2(bindings=[int(b) for b in self.bindings], stream_handle=self.stream.handle)
        # copy outputs back
        out = []
        for i, host_out in enumerate(self.host_outputs):
            cuda.memcpy_dtoh_async(host_out, self.cuda_outputs[i], self.stream)
            out.append(host_out.copy())
        self.stream.synchronize()
        # if single output, return it
        return out[0] if len(out) == 1 else out

# ---------------------------
# Wrapper class
# ---------------------------
class ModelWrapper:
    def __init__(self, path: str, fmt: str = "pt", device: str = "cuda", fp16: bool = False):
        """
        path: path to model file (.pt/.pth/.onnx/.trt)
        fmt: one of 'pt', 'torchscript', 'onnx', 'trt'
        device: 'cuda' or 'cpu'
        fp16: request FP16 where supported (PyTorch/torch-tensorrt/trt)
        """
        self.path = path
        self.fmt = fmt.lower()
        self.device = device
        self.fp16 = fp16
        self._impl = None

        miss = check_missing_deps([self.fmt])
        if miss:
            lines = ["Missing dependencies for requested model format:"]
            for k, v in miss.items():
                lines.append(f" - {k}: {v}")
            raise RuntimeError("\n".join(lines))

        if self.fmt in ("pt", "torchscript"):
            self._impl = _PTModel(path, device=self.device, fp16=self.fp16)
        elif self.fmt == "onnx":
            self._impl = _ONNXModel(path, device=self.device)
        elif self.fmt == "trt":
            self._impl = _TRTModel(path, device=self.device, fp16=self.fp16)
        else:
            raise ValueError("Unsupported format: " + fmt)

    def get_infer_fn(self):
        return self._impl.infer

# ---------------------------
# CLI / quick test
# ---------------------------
def _cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", required=True, help="path to model file (.pt/.onnx/.trt)")
    p.add_argument("--format", "-f", default="pt", help="pt|torchscript|onnx|trt")
    p.add_argument("--device", "-d", default="cuda", help="cuda|cpu")
    p.add_argument("--fp16", action="store_true", help="use FP16 where possible")
    p.add_argument("--shape", default="1,3,256,256", help="input shape as N,C,H,W (comma)")
    args = p.parse_args()

    formats = [args.format]
    miss = check_missing_deps(formats)
    if miss:
        print("Missing packages for requested format:")
        for k, v in miss.items():
            print(f" * {k}: {v}")
        print("Install required packages and rerun.")
        sys.exit(2)

    shape = tuple(int(x) for x in args.shape.split(","))
    rng = np.random.RandomState(0)
    sample = (rng.rand(*shape).astype(np.float32) * 255.0).astype(np.uint8)

    print(f"Loading model {args.model} as {args.format} on device {args.device} fp16={args.fp16} ...")
    try:
        wrapper = ModelWrapper(args.model, fmt=args.format, device=args.device, fp16=args.fp16)
    except Exception as e:
        print("Failed to init model wrapper:", e)
        sys.exit(3)

    infer = wrapper.get_infer_fn()
    print("Warmup + timing (5 runs)...")
    # warmup
    for _ in range(2):
        _ = infer(sample)
    ts = time.time()
    for _ in range(5):
        out = infer(sample)
    t = (time.time() - ts) / 5.0
    print("Done. avg latency (s):", t)
    print("Output shape:", out.shape if hasattr(out, "shape") else "list")
    print("OK")

if __name__ == "__main__":
    _cli_main()
