"""Explainability helpers: Grad-CAM, failure case selection, manual inspection.

Grad-CAM is computed on the predicted top class to visualise which spatial
regions most influence the model's prediction. Comparing low-disagreement
and high-disagreement images shows whether the model attends to the whole
object (clear case) or to multiple plausible objects / ambiguous regions
(ambiguous case).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """Class activation map via gradients of a target class wrt activations
    of a chosen feature layer.

    Use the last conv block of the backbone (e.g. model.backbone.layer4) as
    the target layer for ResNet-style backbones.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Return a (H, W) heatmap normalised to [0, 1] for a single image."""
        assert x.ndim == 4 and x.size(0) == 1, "Pass a single image with shape (1, C, H, W)"
        self.model.eval()
        x.requires_grad_(False)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=-1).item())
        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        acts = self.activations[0]                    # (C, h, w)
        grads = self.gradients[0]                     # (C, h, w)
        weights = grads.mean(dim=(1, 2))              # (C,)
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def select_failure_cases(artifacts: dict, n: int = 8) -> np.ndarray:
    """Pick n test images with the largest |H(p) - H(q)|.

    These are images where the model badly misjudges the *amount* of
    disagreement, which is the failure mode that matters for this task.
    """
    err = np.abs(artifacts["true_entropy"] - artifacts["pred_entropy"])
    order = np.argsort(-err)
    return order[:n]


def select_extreme_entropy(artifacts: dict, n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the n lowest- and n highest- TRUE entropy images.

    Returned indices are positions in the test loader output (use them to
    index artifacts["indices"] for the original CIFAR-10H row id).
    """
    H = artifacts["true_entropy"]
    order_low = np.argsort(H)[:n]
    order_high = np.argsort(-H)[:n]
    return order_low, order_high
