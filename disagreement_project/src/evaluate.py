"""Evaluation metrics for soft-label distribution prediction.

Metrics implemented:
  - per-image and mean: KL(p||q), JSD(p,q), cosine similarity
  - entropy correlation: Pearson and Spearman between H(p) and H(q)
  - Precision@K: do the top-K most-disagreement-predicted images really
    overlap with the top-K most-disagreement true images?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

EPS = 1e-12


# ---------------------------------------------------------------------------
# Per-image metric helpers (numpy).
# ---------------------------------------------------------------------------

def _entropy_bits(probs: np.ndarray) -> np.ndarray:
    """Per-row entropy in bits.

    Clamps once and uses the same array in both positions so that the
    0·log2(0)=0 convention is applied consistently (BUG-7 fix).
    """
    p = np.clip(probs, EPS, 1.0)   # clamp once
    return -np.sum(p * np.log2(p), axis=-1)  # both sides use clamped p


def kl_per_image(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_safe = np.clip(p, EPS, 1.0)
    q_safe = np.clip(q, EPS, 1.0)
    return np.sum(p * (np.log(p_safe) - np.log(q_safe)), axis=-1)


def jsd_per_image(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * kl_per_image(p, m) + 0.5 * kl_per_image(q, m)


def cosine_per_image(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    num = np.sum(p * q, axis=-1)
    den = np.linalg.norm(p, axis=-1) * np.linalg.norm(q, axis=-1)
    return num / np.clip(den, EPS, None)


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    kl_mean: float
    kl_std: float
    jsd_mean: float
    jsd_std: float
    cosine_mean: float
    cosine_std: float
    pearson_entropy: float
    spearman_entropy: float
    precision_at_k: dict       # {k: precision}
    n_images: int

    def to_row(self, name: str) -> dict:
        row = {
            "model": name,
            "n": self.n_images,
            "KL_mean": self.kl_mean, "KL_std": self.kl_std,
            "JSD_mean": self.jsd_mean, "JSD_std": self.jsd_std,
            "cos_mean": self.cosine_mean, "cos_std": self.cosine_std,
            "pearson_H": self.pearson_entropy,
            "spearman_H": self.spearman_entropy,
        }
        for k, v in self.precision_at_k.items():
            row[f"P@{k}"] = v
        return row


def precision_at_k(true_entropy: np.ndarray, pred_entropy: np.ndarray, k: int) -> float:
    """Fraction of overlap between the top-k most-uncertain images by true
    entropy and the top-k most-uncertain images by predicted entropy.
    """
    if k > len(true_entropy):
        k = len(true_entropy)
    top_true = set(np.argsort(-true_entropy)[:k].tolist())
    top_pred = set(np.argsort(-pred_entropy)[:k].tolist())
    return len(top_true & top_pred) / k


@torch.no_grad()
def collect_predictions(model, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (true_probs, pred_probs, hard_labels, dataset_indices) as numpy arrays."""
    model.eval()
    P, Q, Y, IDX = [], [], [], []
    for x, p, hard, idx in loader:
        x = x.to(device, non_blocking=True)
        q = F.softmax(model(x), dim=-1).cpu().numpy()
        P.append(p.numpy())
        Q.append(q)
        Y.append(np.asarray(hard))
        IDX.append(np.asarray(idx))
    return (
        np.concatenate(P, axis=0),
        np.concatenate(Q, axis=0),
        np.concatenate(Y, axis=0),
        np.concatenate(IDX, axis=0),
    )


def evaluate(
    model,
    loader: DataLoader,
    device,
    k_values: Optional[list[int]] = None,
) -> tuple[EvalResult, dict]:
    """Run the full evaluation suite.

    Returns:
        EvalResult: aggregate metrics
        artifacts: dict with raw arrays for plotting
            {"true_probs", "pred_probs", "hard_labels", "indices",
             "true_entropy", "pred_entropy",
             "kl", "jsd", "cosine"}
    """
    if k_values is None:
        k_values = [100, 200, 500]

    P, Q, Y, IDX = collect_predictions(model, loader, device)
    Hp = _entropy_bits(P)
    Hq = _entropy_bits(Q)

    kl = kl_per_image(P, Q)
    jsd = jsd_per_image(P, Q)
    cos = cosine_per_image(P, Q)

    pear, _ = pearsonr(Hp, Hq)
    spear, _ = spearmanr(Hp, Hq)

    p_at_k = {k: precision_at_k(Hp, Hq, k) for k in k_values}

    result = EvalResult(
        kl_mean=float(kl.mean()), kl_std=float(kl.std()),
        jsd_mean=float(jsd.mean()), jsd_std=float(jsd.std()),
        cosine_mean=float(cos.mean()), cosine_std=float(cos.std()),
        pearson_entropy=float(pear),
        spearman_entropy=float(spear),
        precision_at_k=p_at_k,
        n_images=int(P.shape[0]),
    )
    artifacts = {
        "true_probs": P, "pred_probs": Q, "hard_labels": Y, "indices": IDX,
        "true_entropy": Hp, "pred_entropy": Hq,
        "kl": kl, "jsd": jsd, "cosine": cos,
    }
    return result, artifacts
