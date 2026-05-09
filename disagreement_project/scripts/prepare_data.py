"""Step 1: download CIFAR-10, align with CIFAR-10H, run sanity checks,
produce all data-stage plots required by the assignment.

Usage:
    python scripts/01_prepare_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow `from src.x import y` when run from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import data as data_mod
from src import viz
from src.utils import load_config, set_seed, ensure_dirs, save_json


def main():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    set_seed(cfg["seed"], deterministic=cfg["deterministic"])
    ensure_dirs(cfg)

    fig_dir = Path(cfg["paths"]["figures"])
    tab_dir = Path(cfg["paths"]["tables"])

    print("[1/5] Downloading CIFAR-10 train set (50,000 hard-label images)...")
    _ = data_mod.cifar10_train_dataset(cfg["paths"]["data_root"])

    print("[2/5] Loading CIFAR-10 test split + CIFAR-10H soft labels (10,000 images)...")
    full = data_mod.CIFAR10HSoft(
        data_root=cfg["paths"]["data_root"],
        cifar10h_probs_path=cfg["paths"]["cifar10h_probs"],
        indices=None,
        transform=data_mod.get_eval_transform(),
    )
    probs = full.soft_labels
    hard_labels = np.array([full.cifar.targets[i] for i in range(10000)])

    print("[3/5] Sanity checks on annotator distributions...")
    sc = data_mod.sanity_check_probs(probs)
    save_json(sc, tab_dir / "data_sanity_checks.json")
    print(f"      shape={sc['shape']}  rows-off-by->{sc['atol']}={sc['n_rows_off_by_more_than_atol']}")
    print(f"      min row sum={sc['min_row_sum']:.6f}  max row sum={sc['max_row_sum']:.6f}")
    print("      verifying alignment by checking hard-label majority match...")
    majority = probs.argmax(axis=1)
    agree = float((majority == hard_labels).mean())
    print(f"      hard-label / majority-vote agreement = {agree:.4f}")
    save_json({"hard_majority_agreement": agree}, tab_dir / "data_alignment_check.json")

    print("[4/5] Computing dataset statistics...")
    H = data_mod.shannon_entropy(probs)
    per_class_H = data_mod.per_class_average_entropy(probs, hard_labels)
    soft_conf = data_mod.soft_confusion_matrix(probs, hard_labels)
    np.save(tab_dir / "true_entropy.npy", H)
    np.save(tab_dir / "per_class_entropy.npy", per_class_H)
    np.save(tab_dir / "soft_confusion.npy", soft_conf)

    print("[5/5] Producing required data-stage plots...")
    viz.plot_entropy_histogram(H, str(fig_dir / "data_entropy_histogram.png"))
    viz.plot_per_class_entropy(per_class_H, str(fig_dir / "data_per_class_entropy.png"))
    viz.plot_soft_confusion(soft_conf, str(fig_dir / "data_soft_confusion.png"))

    # low/high entropy example grid
    n_show = 8
    order = np.argsort(H)
    low_idx = order[:n_show]
    high_idx = order[-n_show:][::-1]
    low_imgs, low_probs_list = [], []
    high_imgs, high_probs_list = [], []
    for i in low_idx:
        img, p, _, _ = full[int(i)]
        low_imgs.append(img); low_probs_list.append(p.numpy())
    for i in high_idx:
        img, p, _, _ = full[int(i)]
        high_imgs.append(img); high_probs_list.append(p.numpy())
    viz.plot_low_high_entropy_grid(
        low_imgs, low_probs_list, high_imgs, high_probs_list,
        str(fig_dir / "data_low_high_entropy_examples.png"),
    )

    # write the splits to disk so other scripts use the SAME indices
    train_idx, val_idx, test_idx = data_mod.make_splits(
        n_total=10000,
        n_train=cfg["split"]["n_train"],
        n_val=cfg["split"]["n_val"],
        n_test=cfg["split"]["n_test"],
        seed=cfg["seed"],
    )
    np.savez(tab_dir / "splits.npz",
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"      splits saved: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    print("done.")


if __name__ == "__main__":
    main()
