"""Step 4: core performance evaluation.

Loads each per-loss checkpoint from script 02, evaluates on the held-out
test set, and produces:
  - the summary comparison table across loss functions
  - the predicted-vs-true entropy scatter plots
  - the grouped comparison chart on the primary metric (KL_mean)
  - a qualitative example panel covering low/medium/high disagreement

Usage:
    python scripts/04_evaluate_all.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import data as data_mod
from src import models as models_mod
from src import evaluate as eval_mod
from src import viz
from src.utils import (load_config, ensure_dirs, get_device, load_checkpoint,
                       set_seed, save_json)


def make_test_loader(cfg):
    splits = np.load(Path(cfg["paths"]["tables"]) / "splits.npz")
    test_ds = data_mod.CIFAR10HSoft(
        cfg["paths"]["data_root"], cfg["paths"]["cifar10h_probs"],
        indices=splits["test_idx"], transform=data_mod.get_eval_transform(),
    )
    return DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                      shuffle=False, num_workers=2, pin_memory=True), test_ds


def qualitative_panel(test_ds, artifacts, out_path: str, n_each: int = 4):
    """Show n_each images at low / medium / high TRUE entropy along with
    side-by-side true vs predicted distributions.
    """
    H = artifacts["true_entropy"]
    order = np.argsort(H)
    low = order[:n_each]
    high = order[-n_each:][::-1]
    mid_start = len(order) // 2 - n_each // 2
    medium = order[mid_start:mid_start + n_each]
    picks = np.concatenate([low, medium, high])

    imgs = []
    for i in picks:
        img, _, _, _ = test_ds[int(i)]
        imgs.append(img)
    true_p = artifacts["true_probs"][picks]
    pred_p = artifacts["pred_probs"][picks]
    viz.plot_failure_cases(imgs, true_p, pred_p, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses", nargs="+", default=["kl", "jsd", "composite"])
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    set_seed(cfg["seed"], deterministic=cfg["deterministic"])
    ensure_dirs(cfg)
    device = get_device()
    print(f"device: {device}")

    test_loader, test_ds = make_test_loader(cfg)

    rows = []
    artifacts_by_loss = {}
    for loss_name in args.losses:
        ckpt_path = Path(cfg["paths"]["checkpoints"]) / f"best_{loss_name}.pt"
        if not ckpt_path.exists():
            print(f"missing {ckpt_path}, skipping {loss_name}")
            continue
        model = models_mod.build_model(head_kind=cfg["model"]["head"]).to(device)
        model.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"])
        res, art = eval_mod.evaluate(model, test_loader, device, cfg["eval"]["precision_at_k"])
        rows.append(res.to_row(loss_name))
        artifacts_by_loss[loss_name] = art

        # required: scatter of pred vs true entropy, per loss
        viz.plot_pred_vs_true_entropy(
            art["true_entropy"], art["pred_entropy"],
            str(Path(cfg["paths"]["figures"]) / f"entropy_scatter_{loss_name}.png"),
            title=f"({loss_name})",
        )

    df = pd.DataFrame(rows)
    out_csv = Path(cfg["paths"]["tables"]) / "loss_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsummary table -> {out_csv}")
    print(df.to_string(index=False))

    # required: grouped comparison chart across losses, primary metric
    if not df.empty:
        viz.plot_grouped_loss_comparison(
            df[["model", "KL_mean"]], "KL_mean",
            str(Path(cfg["paths"]["figures"]) / "loss_comparison_KL.png"),
        )
        viz.plot_grouped_loss_comparison(
            df[["model", "spearman_H"]], "spearman_H",
            str(Path(cfg["paths"]["figures"]) / "loss_comparison_spearman.png"),
        )

        # required: qualitative examples panel — use the best model (lowest KL_mean)
        best_loss = df.sort_values("KL_mean").iloc[0]["model"]
        qualitative_panel(
            test_ds,
            artifacts_by_loss[best_loss],
            str(Path(cfg["paths"]["figures"]) / f"qualitative_examples_{best_loss}.png"),
        )

    print("done.")


if __name__ == "__main__":
    main()
