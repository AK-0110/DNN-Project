"""Robustness checks.

  - resample_annotators : approximate annotator subsampling by sampling
    from the empirical p(y|x) and re-normalising. Tests whether predictions
    are stable when the soft-label "ground truth" is noisier.

  - corrupt_image       : applies one of {gaussian_noise, gaussian_blur,
    contrast} at a chosen severity level (1-5).

  - class_conditional_metrics : evaluation broken down by hard-label class.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Annotator subsampling.
# ---------------------------------------------------------------------------

def resample_annotators(probs: np.ndarray, fraction: float, total_annotators: int = 50,
                         seed: int = 0) -> np.ndarray:
    """Approximate fewer annotators per image by re-sampling labels.

    For each image we sample n = round(fraction * total_annotators) labels
    from the multinomial defined by the original distribution, then
    re-normalise to a probability vector. With fewer samples, distributions
    become noisier (more 0s, sharper).
    """
    n_per_image = max(1, int(round(fraction * total_annotators)))
    rng = np.random.default_rng(seed)
    out = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        counts = rng.multinomial(n_per_image, probs[i])
        out[i] = counts / counts.sum()
    return out


# ---------------------------------------------------------------------------
# OOD corruptions.
# ---------------------------------------------------------------------------

# Severity tables: indexed 1..5. Values picked to roughly match CIFAR-10-C scales.
_NOISE_STD = {1: 0.04, 2: 0.06, 3: 0.08, 4: 0.10, 5: 0.12}
_BLUR_SIGMA = {1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.2}
_CONTRAST_C = {1: 0.75, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.15}


def _gaussian_kernel_1d(sigma: float, ksize: int) -> torch.Tensor:
    x = torch.arange(ksize) - (ksize - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on a (B, 3, H, W) tensor in [0, 1]."""
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    k1d = _gaussian_kernel_1d(sigma, ksize).to(images.device, images.dtype)
    kx = k1d.view(1, 1, 1, ksize).expand(3, 1, 1, ksize)
    ky = k1d.view(1, 1, ksize, 1).expand(3, 1, ksize, 1)
    pad = ksize // 2
    x = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, kx, groups=3)
    x = F.conv2d(x, ky, groups=3)
    return x


def corrupt_batch(images: torch.Tensor, kind: str, severity: int) -> torch.Tensor:
    """Apply a corruption to a (B, 3, H, W) tensor in the [0, 1] image space.

    These corruptions assume *unnormalised* images. If your loader produces
    normalised tensors, denormalise first, corrupt, then renormalise.
    """
    assert severity in (1, 2, 3, 4, 5)
    kind = kind.lower()
    if kind == "gaussian_noise":
        std = _NOISE_STD[severity]
        noise = torch.randn_like(images) * std
        return (images + noise).clamp(0.0, 1.0)
    if kind == "gaussian_blur":
        return _gaussian_blur(images, _BLUR_SIGMA[severity]).clamp(0.0, 1.0)
    if kind == "contrast":
        c = _CONTRAST_C[severity]
        means = images.mean(dim=(2, 3), keepdim=True)
        return ((images - means) * c + means).clamp(0.0, 1.0)
    raise ValueError(f"Unknown corruption: {kind}")


# ---------------------------------------------------------------------------
# Class-conditional evaluation.
# ---------------------------------------------------------------------------

def class_conditional_metrics(true_probs: np.ndarray, pred_probs: np.ndarray,
                               hard_labels: np.ndarray) -> dict:
    """Mean KL, JSD, and entropy correlation per hard-label class.

    Note: 'class' here means the majority/hard label of the image. Some
    classes are inherently more ambiguous than others (e.g. cat-vs-dog).
    """
    from .evaluate import kl_per_image, jsd_per_image
    from scipy.stats import pearsonr
    from .data import CIFAR10_CLASSES

    EPS = 1e-12
    Hp = -np.sum(true_probs * np.log2(np.clip(true_probs, EPS, 1.0)), axis=-1)
    Hq = -np.sum(pred_probs * np.log2(np.clip(pred_probs, EPS, 1.0)), axis=-1)

    out = {}
    for c in range(10):
        mask = hard_labels == c
        if mask.sum() < 5:
            continue
        kl = kl_per_image(true_probs[mask], pred_probs[mask])
        jsd = jsd_per_image(true_probs[mask], pred_probs[mask])
        try:
            r, _ = pearsonr(Hp[mask], Hq[mask])
        except Exception:
            r = float("nan")
        out[CIFAR10_CLASSES[c]] = {
            "n": int(mask.sum()),
            "KL_mean": float(kl.mean()),
            "JSD_mean": float(jsd.mean()),
            "pearson_H": float(r),
            "true_entropy_mean": float(Hp[mask].mean()),
            "pred_entropy_mean": float(Hq[mask].mean()),
        }
    return out
