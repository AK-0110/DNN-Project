"""Loss functions for soft-label distribution prediction.

All losses take (logits, target_probs) where:
  - logits         : (B, C) raw scores
  - target_probs   : (B, C) non-negative, rows sum to 1

They all return a scalar tensor (mean over batch).

Custom composite loss `CompositeDisagreementLoss`:
    L = KL(p || q) + lambda_H * |H(p) - H(q)| + gamma * focal_weighted_KL

Justification:
  - KL pulls q toward p in a mass-matching sense, but a model that always
    predicts a sharp distribution can still get reasonable mean KL because
    most CIFAR-10H images are low-entropy. We need explicit pressure to
    match the *shape* of disagreement.
  - |H(p) - H(q)| penalises mismatch in total disagreement.
  - The focal term up-weights the small fraction of high-entropy
    (high-disagreement) images, which are the ones we actually care about
    for this task.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12


# ---------------------------------------------------------------------------
# Standard losses.
# ---------------------------------------------------------------------------

class KLDivLoss(nn.Module):
    """KL(p || q) where p = target, q = softmax(logits).

    Plain English: how much information is lost when we use q to approximate p.
    For each image, KL is 0 when q exactly equals p. KL is asymmetric.
    """
    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        log_q = F.log_softmax(logits, dim=-1)
        # KL(p||q) = sum_y p(y) * (log p(y) - log q(y))
        log_p = torch.log(target_probs.clamp(min=EPS))
        kl = (target_probs * (log_p - log_q)).sum(dim=-1)
        return kl.mean()


class JSDLoss(nn.Module):
    """Jensen-Shannon divergence — symmetric, bounded in [0, 1] (using log2).

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), m = 0.5 * (p + q)

    Suitable here because it is symmetric (penalises both directions of
    mismatch equally) and bounded, so gradients do not blow up when q is
    near zero on a class with non-zero p.
    """
    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        q = F.softmax(logits, dim=-1)
        p = target_probs                        # keep raw probs for mixture midpoint
        m = 0.5 * (p + q)                      # mixture uses unshifted p and q
        # clamp only when computing logs to avoid log(0)
        log_m = torch.log(m.clamp(min=EPS))
        log_p = torch.log(p.clamp(min=EPS))
        log_q = torch.log(q.clamp(min=EPS))
        kl_pm = (p * (log_p - log_m)).sum(dim=-1)
        kl_qm = (q * (log_q - log_m)).sum(dim=-1)
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd.mean()


class CosineDistributionLoss(nn.Module):
    """1 - cosine_similarity(p, q).

    Measures angular mismatch between distribution vectors. Directionally
    sensitive but ignores magnitude, which is fine here since both vectors
    are constrained to sum to 1.
    """
    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        q = F.softmax(logits, dim=-1)
        sim = F.cosine_similarity(q, target_probs, dim=-1)
        return (1.0 - sim).mean()


class SoftCrossEntropyLoss(nn.Module):
    """Cross entropy with soft targets: -sum_y p(y) * log q(y).

    Equivalent to KL(p||q) + H(p) — and since H(p) is constant w.r.t. model
    parameters, optimising this is equivalent to optimising KL. Included as
    a sanity-check baseline.
    """
    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        log_q = F.log_softmax(logits, dim=-1)
        return -(target_probs * log_q).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Earth Mover's Distance (bonus).
# ---------------------------------------------------------------------------

class EMDLoss(nn.Module):
    """1-D EMD over the class index axis using the closed form for sorted
    distributions: EMD(p, q) = sum_i |CDF_p(i) - CDF_q(i)|.

    This treats class index as a 1-D ordering. Strictly speaking, CIFAR-10
    classes are categorical, not ordinal, so this is a rough proxy that
    nonetheless penalises mass that has to "travel" between classes.

    For a more principled EMD, swap in `pot.emd2` with a class-distance
    matrix derived from semantic similarity (e.g. cat-dog should be cheap
    to swap, cat-truck should be expensive).
    """
    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        q = F.softmax(logits, dim=-1)
        p = target_probs
        cdf_p = torch.cumsum(p, dim=-1)
        cdf_q = torch.cumsum(q, dim=-1)
        emd = (cdf_p - cdf_q).abs().sum(dim=-1)
        return emd.mean()


# ---------------------------------------------------------------------------
# Custom composite loss.
# ---------------------------------------------------------------------------

class CompositeDisagreementLoss(nn.Module):
    """KL + entropy-error penalty + focal weighting on high-disagreement images.

    Args:
        lambda_h: weight on |H(p) - H(q)| (matches the *amount* of disagreement)
        gamma:    focal exponent. With gamma > 0, images with higher true
                  entropy (more disagreement) are weighted more in the loss.
                  This corrects for class imbalance: most CIFAR-10H images
                  have very low entropy.
    """
    def __init__(self, lambda_h: float = 0.5, gamma: float = 1.0,
                 num_classes: int = 10):
        super().__init__()
        self.lambda_h = lambda_h
        self.gamma = gamma
        # Precompute max entropy (log2 C) as a Python float — avoids a CPU
        # tensor allocation and device sync on every forward pass (BUG-4 fix).
        import math
        self.H_max: float = math.log2(num_classes)

    @staticmethod
    def _entropy_bits(probs: torch.Tensor) -> torch.Tensor:
        # Clamp once and use the same clamped tensor in both places so that
        # the 0·log(0)=0 convention is handled consistently (BUG-7 analogue).
        p_safe = probs.clamp(min=EPS)
        return -(p_safe * torch.log2(p_safe)).sum(dim=-1)

    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        # Use F.softmax directly — numerically identical to log_softmax.exp()
        # but avoids an extra fp16 rounding step under AMP (BUG-3 fix).
        q = F.softmax(logits, dim=-1)
        log_q = torch.log(q.clamp(min=EPS))
        p = target_probs

        # KL(p || q) per sample
        log_p = torch.log(p.clamp(min=EPS))
        kl_per = (p * (log_p - log_q)).sum(dim=-1)

        # entropy error (bits)
        H_p = self._entropy_bits(p)
        H_q = self._entropy_bits(q)
        entropy_err = (H_p - H_q).abs()

        # focal weight: emphasises images where true entropy is large.
        # weight in [1, 1 + gamma * (H_p / H_max)]
        focal_w = 1.0 + self.gamma * (H_p / self.H_max)

        loss_per = focal_w * (kl_per + self.lambda_h * entropy_err)
        return loss_per.mean()


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------

LOSS_REGISTRY = {
    "kl": KLDivLoss,
    "jsd": JSDLoss,
    "cosine": CosineDistributionLoss,
    "soft_ce": SoftCrossEntropyLoss,
    "emd": EMDLoss,
    "composite": CompositeDisagreementLoss,
}


def build_loss(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Choices: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)
