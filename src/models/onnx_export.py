"""
ONNX Export Pipeline
=====================
Exports trained YOLOv8 model to ONNX format for deployment.

Design:
    ONNX enables framework-agnostic deployment — the exported model runs on:
    - NVIDIA Triton Inference Server
    - ONNXRuntime (CPU/GPU/edge devices)
    - TensorRT (for NVIDIA GPU production deployment)
    - OpenVINO (for Intel hardware)

    For aerospace/defense edge deployment (drone processors, embedded systems),
    ONNX → TensorRT quantization can achieve 3-5× additional speedup.
"""

import argparse
import importlib
from pathlib import Path

from loguru import logger


def export_to_onnx(
    weights: str,
    output_dir: str = "models/onnx",
    imgsz: int = 640,
    batch: int = 1,
    simplify: bool = True,
    dynamic: bool = False,
    opset: int = 17,
    half: bool = False,
    device: str = "cpu",
) -> str:
    """
    Export YOLOv8 PyTorch weights to ONNX.

    Args:
        weights: Path to .pt weights file
        output_dir: Directory to save ONNX model
        imgsz: Input image size
        batch: Static batch size (use dynamic=True for variable)
        simplify: Run onnxsim to simplify graph
        dynamic: Enable dynamic batch/size axes
        opset: ONNX opset version (17 recommended)
        half: Export FP16 (requires CUDA device)
        device: Export device

    Returns:
        Path to exported ONNX file
    """
    from ultralytics import YOLO

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weights_path = Path(weights)
    logger.info(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    logger.info(f"Exporting to ONNX (opset={opset}, imgsz={imgsz}, batch={batch})")

    export_args = dict(
        format="onnx",
        imgsz=imgsz,
        batch=batch,
        simplify=simplify,
        dynamic=dynamic,
        opset=opset,
        half=half,
        device=device,
        verbose=False,
    )

    exported_path = model.export(**export_args)
    logger.info(f"ONNX export complete: {exported_path}")

    # Move to output dir if not already there
    exported = Path(exported_path)
    final_path = output_path / exported.name
    if exported.resolve() != final_path.resolve():
        import shutil
        shutil.copy2(exported, final_path)

    # Validate exported model
    _validate_onnx(str(final_path))

    # Print model info
    file_size_mb = final_path.stat().st_size / 1e6
    logger.success(f"✅ ONNX model ready: {final_path} ({file_size_mb:.1f} MB)")

    return str(final_path)


def _validate_onnx(onnx_path: str) -> None:
    """Validate ONNX model graph integrity."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info("ONNX model validation passed ✓")

        # Print input/output info
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            logger.info(f"  Input:  {inp.name} {shape}")
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            logger.info(f"  Output: {out.name} {shape}")
    except ImportError:
        logger.warning("onnx package not installed. Skipping validation.")
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")


def run_onnx_inference(onnx_path: str, image_path: str, conf: float = 0.25) -> list:
    """
    Demo inference using ONNXRuntime directly.
    Useful for testing exported model without Ultralytics dependency.
    """
    import cv2
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("Install onnxruntime: pip install onnxruntime-gpu")

    # Load model
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    # Preprocess
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis]  # BCHW

    # Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})
    first_output = np.asarray(outputs[0])

    logger.info(f"ONNXRuntime inference complete. Output shape: {first_output.shape}")
    return outputs


def export_to_tensorrt(onnx_path: str, engine_path: str, fp16: bool = True) -> str:
    """
    Convert ONNX to TensorRT engine for maximum GPU throughput.

    Requires: tensorrt, pycuda
    Note: Engine is device-specific — must be built on target hardware.
    """
    try:
        trt = importlib.import_module("tensorrt")
    except ImportError:
        raise ImportError(
            "TensorRT not available. Install with: "
            "pip install tensorrt pycuda"
        )

    logger.info(f"Converting to TensorRT (FP16={fp16})...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled")

    engine = builder.build_engine(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    logger.success(f"TensorRT engine saved: {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Export SatDet model to ONNX")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--output", default="models/onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    export_to_onnx(
        weights=args.weights,
        output_dir=args.output,
        imgsz=args.imgsz,
        batch=args.batch,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
        opset=args.opset,
        half=args.half,
        device=args.device,
    )


if __name__ == "__main__":
    main()
