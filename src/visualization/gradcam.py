"""
Grad-CAM Attention Visualization
===================================
Generates class activation maps to visualize what regions the model
attends to when making detections.

Design Note:
    For YOLO-style models, we hook into the C2f blocks in the neck
    (feature pyramid) rather than the final detection head.
    The C2f layers at stride 8 (small object scale) are most informative
    for satellite imagery analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Tuple

import torch
import torch.nn.functional as F
from loguru import logger


class YOLOGradCAM:
    """
    Grad-CAM implementation for YOLOv8 architecture.

    Hooks into specified backbone/neck layers to extract gradients
    and feature maps for attention visualization.
    """

    def __init__(
        self,
        model,
        target_layer_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            model: Ultralytics YOLO model
            target_layer_indices: Layer indices to hook (None = auto-detect C2f layers)
        """
        self.model = model
        self.device, self.dtype = self._model_device_dtype()
        self.gradients = {}
        self.activations = {}
        self.hooks = []

        # YOLOv8 C2f layers in the neck (strides 8, 16, 32)
        # Layer 9 = stride-8 (best for small objects in satellite imagery)
        self.target_layers = self._get_target_layers(target_layer_indices)
        self._register_hooks()

    def _get_target_layers(self, indices: Optional[List[int]]) -> list:
        """Auto-detect C2f layers in neck if indices not specified."""
        if indices is not None:
            return [list(self.model.model.model.children())[i] for i in indices]

        # Find C2f layers
        target = []
        for name, module in self.model.model.model.named_modules():
            if "C2f" in type(module).__name__:
                target.append(module)
        # Use the last 3 C2f layers (neck feature scales)
        return target[-3:] if len(target) >= 3 else target

    def _register_hooks(self):
        """Register forward/backward hooks on target layers."""
        for idx, layer in enumerate(self.target_layers):
            self.hooks.append(
                layer.register_forward_hook(self._forward_hook(idx))
            )
            self.hooks.append(
                layer.register_full_backward_hook(self._backward_hook(idx))
            )

    def _forward_hook(self, idx: int):
        def hook(module, input, output):
            self.activations[idx] = output.detach()

        return hook

    def _backward_hook(self, idx: int):
        def hook(module, grad_input, grad_output):
            self.gradients[idx] = grad_output[0].detach()

        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def _model_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        """Return the device and dtype used by the underlying YOLO module."""
        try:
            param = next(self.model.model.parameters())
            dtype = param.dtype if param.is_floating_point() else torch.float32
            return param.device, dtype
        except StopIteration:
            return torch.device("cpu"), torch.float32

    def generate(
        self,
        image: Union[str, np.ndarray],
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            image: Input image path or numpy array
            target_class: Target class ID (None = use highest confidence detection)

        Returns:
            (original_image, heatmap_overlay) both as BGR numpy arrays
        """
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
        else:
            img_bgr = image.copy()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_bgr.shape[:2]

        # Preprocess
        img_tensor = self._preprocess(img_rgb).to(device=self.device, dtype=self.dtype)
        img_tensor.requires_grad_(True)

        # Clear stored grads/activations
        self.gradients = {}
        self.activations = {}

        # Forward pass through YOLO internals
        with torch.enable_grad():
            # Use model's predict with grad
            self.model.model.train(False)
            pred = self.model.model(img_tensor)

            # Build scalar objective: sum of objectness scores
            # (or target class scores if specified)
            if isinstance(pred, (list, tuple)):
                # Get raw predictions before NMS
                raw_pred = pred[0] if isinstance(pred[0], torch.Tensor) else pred
            else:
                raw_pred = pred

            if raw_pred.dim() == 3:
                # [B, anchors, (4 + nc)]
                scores = raw_pred[0, :, 4:]  # objectness + class scores
                if target_class is not None and target_class < scores.shape[-1]:
                    obj_score = scores[:, target_class].sum()
                else:
                    obj_score = scores.max(dim=-1).values.sum()
            else:
                obj_score = raw_pred.sum()

        # Backward pass
        self.model.model.zero_grad()
        obj_score.backward(retain_graph=True)

        # Generate CAM from last hooked layer
        layer_idx = next(
            (idx for idx in reversed(range(len(self.target_layers)))
             if idx in self.activations and idx in self.gradients),
            None,
        )
        if layer_idx is None:
            logger.warning("No activations/gradients captured. Using blank heatmap.")
            cam = np.zeros((img_h, img_w), dtype=np.float32)
        else:
            activation = self.activations[layer_idx].squeeze(0)  # [C, H, W]
            gradient = self.gradients[layer_idx].squeeze(0)      # [C, H, W]

            # Global average pool gradients
            weights = gradient.mean(dim=(1, 2))           # [C]

            # Weighted sum of activations
            cam = (weights[:, None, None] * activation).sum(dim=0)  # [H, W]
            cam = F.relu(cam).cpu().numpy()

            # Normalize
            if cam.max() > 0:
                cam = cam / cam.max()

            # Resize to original image
            cam = cv2.resize(cam, (img_w, img_h))

        # Create colored overlay
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

        return img_bgr, overlay

    @staticmethod
    def _preprocess(img_rgb: np.ndarray, size: int = 640) -> torch.Tensor:
        """Preprocess image to model input tensor."""
        img_resized = cv2.resize(img_rgb, (size, size))
        img_float = img_resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        return tensor


def generate_gradcam(
    weights: str,
    image_path: str,
    output_path: str,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience wrapper: load model, generate Grad-CAM, save result.

    Args:
        weights: Path to .pt weights
        image_path: Input image path
        output_path: Where to save the heatmap visualization
        target_class: Target class ID (None = auto)

    Returns:
        Heatmap overlay image as numpy array
    """
    from ultralytics import YOLO
    model = YOLO(weights)

    gcam = YOLOGradCAM(model)
    try:
        original, overlay = gcam.generate(image_path, target_class)
        cv2.imwrite(output_path, overlay)
        logger.success(f"Grad-CAM saved: {output_path}")
        return overlay
    finally:
        gcam.remove_hooks()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="gradcam_output.jpg")
    parser.add_argument("--class-id", type=int, default=None)
    args = parser.parse_args()

    generate_gradcam(args.weights, args.image, args.output, args.class_id)
