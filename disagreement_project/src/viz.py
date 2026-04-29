"""Plotting helpers. All figures saved to outputs/figures/<name>.png."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .data import CIFAR10_CLASSES, CIFAR_MEAN, CIFAR_STD


def _save(fig, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data-stage plots.
# ---------------------------------------------------------------------------

def plot_entropy_histogram(entropies: np.ndarray, out_path: str, title: str = "True entropy histogram"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(entropies, bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Shannon entropy (bits)")
    ax.set_ylabel("# images")
    ax.set_title(title)
    _save(fig, out_path)


def plot_per_class_entropy(per_class: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(CIFAR10_CLASSES, per_class, color="indianred")
    ax.set_ylabel("Avg true entropy (bits)")
    ax.set_title("Per-class average annotator entropy")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out_path)


def plot_soft_confusion(matrix: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
                annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=True)
    ax.set_xlabel("Annotator class assignments")
    ax.set_ylabel("Hard label (majority)")
    ax.set_title("Avg annotator distribution per hard-label class")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Image grid (denormalises CIFAR-style normalisation).
# ---------------------------------------------------------------------------

def _denorm(x: np.ndarray) -> np.ndarray:
    """x of shape (3, H, W) normalised with CIFAR stats -> (H, W, 3) in [0, 1]."""
    mean = np.array(CIFAR_MEAN).reshape(3, 1, 1)
    std = np.array(CIFAR_STD).reshape(3, 1, 1)
    img = x * std + mean
    img = np.clip(img, 0, 1).transpose(1, 2, 0)
    return img


def plot_low_high_entropy_grid(low_imgs, low_probs, high_imgs, high_probs, out_path: str):
    """Each image gets a row showing the image plus its annotator distribution as a bar."""
    n = len(low_imgs)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.5))
    for ax_row, (imgs, probs, label) in zip(
        [axes[0], axes[1]], [(low_imgs, low_probs, "low entropy"), (high_imgs, high_probs, "high entropy")]
    ):
        for i in range(n):
            img = _denorm(imgs[i].numpy() if hasattr(imgs[i], "numpy") else imgs[i])
            ax_row[i].imshow(img)
            top = int(np.argmax(probs[i]))
            ax_row[i].set_title(f"{CIFAR10_CLASSES[top]}\np={probs[i][top]:.2f}", fontsize=8)
            ax_row[i].axis("off")
    fig.text(0.02, 0.75, "low\nentropy", ha="left", va="center")
    fig.text(0.02, 0.27, "high\nentropy", ha="left", va="center")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Training curves.
# ---------------------------------------------------------------------------

def plot_training_curves(train_loss, val_loss, val_metric, out_path: str, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(train_loss, label="train loss")
    axes[0].plot(val_loss, label="val loss (KL)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].set_title(f"{title} loss")
    axes[1].plot(val_metric, color="darkgreen")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val KL")
    axes[1].set_title(f"{title} val metric")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Evaluation plots.
# ---------------------------------------------------------------------------

def plot_pred_vs_true_entropy(true_h, pred_h, out_path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_h, pred_h, s=6, alpha=0.4)
    lim = max(float(true_h.max()), float(pred_h.max()))
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("True entropy (bits)")
    ax.set_ylabel("Predicted entropy (bits)")
    ax.set_title(f"Predicted vs true entropy {title}")
    _save(fig, out_path)


def plot_grouped_loss_comparison(table_df, metric: str, out_path: str):
    """table_df: a pandas dataframe with column 'model' and metric columns."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table_df["model"], table_df[metric], color="slateblue")
    ax.set_ylabel(metric); ax.set_title(f"{metric} across loss functions")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, out_path)


def plot_corruption_response(severities, mean_pred_entropy_by_kind: dict, out_path: str):
    """mean_pred_entropy_by_kind: {kind: [mean H_q at each severity]}"""
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, vals in mean_pred_entropy_by_kind.items():
        ax.plot(severities, vals, marker="o", label=kind)
    ax.set_xlabel("Severity"); ax.set_ylabel("Mean predicted entropy (bits)")
    ax.set_title("Predicted disagreement vs corruption severity")
    ax.legend()
    _save(fig, out_path)


def plot_per_class_bar(per_class: dict, metric: str, out_path: str):
    classes = list(per_class.keys())
    values = [per_class[c][metric] for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(classes, values, color="seagreen")
    ax.set_ylabel(metric); ax.set_title(f"Per-class {metric}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out_path)


def plot_failure_cases(images, true_probs, pred_probs, out_path: str):
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.2 * n))
    if n == 1:
        axes = axes[None, :]
    x = np.arange(10)
    for r in range(n):
        img = _denorm(images[r].numpy() if hasattr(images[r], "numpy") else images[r])
        axes[r, 0].imshow(img); axes[r, 0].axis("off")
        axes[r, 1].bar(x - 0.2, true_probs[r], width=0.4, label="true")
        axes[r, 1].bar(x + 0.2, pred_probs[r], width=0.4, label="pred")
        axes[r, 1].set_xticks(x); axes[r, 1].set_xticklabels(CIFAR10_CLASSES, rotation=45, fontsize=7)
        axes[r, 1].set_ylim(0, 1)
        if r == 0:
            axes[r, 1].legend(fontsize=7)
    fig.suptitle("Failure cases: largest |H(p) - H(q)|")
    _save(fig, out_path)


def plot_gradcam_panel(images, cams, titles, out_path: str):
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.5))
    for i in range(n):
        img = _denorm(images[i].numpy() if hasattr(images[i], "numpy") else images[i])
        axes[0, i].imshow(img); axes[0, i].axis("off"); axes[0, i].set_title(titles[i], fontsize=8)
        axes[1, i].imshow(img); axes[1, i].imshow(cams[i], cmap="jet", alpha=0.45)
        axes[1, i].axis("off")
    _save(fig, out_path)


def plot_architecture_diagram(out_path: str):
    """A simple labelled diagram of backbone + head."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    boxes = [
        (0.02, "Input\n(3, 32, 32)"),
        (0.18, "Stem\nConv3x3 + BN + ReLU"),
        (0.36, "ResNet stages\nlayer1..layer4"),
        (0.56, "Global avg pool\n-> 512-d"),
        (0.72, "Head\nLinear / MLP / Temp"),
        (0.88, "Softmax\n10-d distribution"),
    ]
    for x, txt in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.12, 0.3, fc="#cce5ff", ec="black"))
        ax.text(x + 0.06, 0.5, txt, ha="center", va="center", fontsize=8)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.12
        x2 = boxes[i + 1][0]
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5),
                    arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Disagreement model architecture")
    _save(fig, out_path)
