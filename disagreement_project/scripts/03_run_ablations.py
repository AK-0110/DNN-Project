"""Step 3: ablation studies. Pick three of A/B/C/D from the assignment.

Default: runs A (backbone init), B (loss comparison — re-uses checkpoints
from script 02), and D (prediction head architecture). The 4th
ablation (C: training data strategy) is included for bonus credit and
can be turned on with --include-c.

Usage:
    python scripts/03_run_ablations.py
    python scripts/03_run_ablations.py --include-c        # also run the bonus ablation
"""
from __future__ import annotations

import argparse
import copy
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
from src import losses as losses_mod
from src import train as train_mod
from src import evaluate as eval_mod
from src import viz
from src.utils import (load_config, set_seed, ensure_dirs, get_device,
                       load_checkpoint, save_json)


def make_loaders(cfg):
    splits = np.load(Path(cfg["paths"]["tables"]) / "splits.npz")
    train_ds = data_mod.CIFAR10HSoft(
        cfg["paths"]["data_root"], cfg["paths"]["cifar10h_probs"],
        indices=splits["train_idx"],
        transform=data_mod.get_train_transform(
            cfg["augment"]["random_crop_padding"],
            cfg["augment"]["random_horizontal_flip"]),
    )
    val_ds = data_mod.CIFAR10HSoft(
        cfg["paths"]["data_root"], cfg["paths"]["cifar10h_probs"],
        indices=splits["val_idx"], transform=data_mod.get_eval_transform(),
    )
    test_ds = data_mod.CIFAR10HSoft(
        cfg["paths"]["data_root"], cfg["paths"]["cifar10h_probs"],
        indices=splits["test_idx"], transform=data_mod.get_eval_transform(),
    )
    bs = cfg["train"]["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True),
    )


def init_with(cfg_local, device, pretrain_ckpt):
    """Build a model whose head + init are taken from cfg_local['model']."""
    model = models_mod.build_model(head_kind=cfg_local["model"]["head"]).to(device)
    init = cfg_local["model"]["init"]
    if init == "imagenet":
        models_mod.init_from_imagenet_resnet18(model)
    elif init == "hard_pretrain":
        state = load_checkpoint(pretrain_ckpt, map_location=device)["model"]
        own = model.state_dict()
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        model.load_state_dict(own, strict=False)
    return model


def train_and_eval(cfg_local, tag, device, pretrain_ckpt, loss_name, train_loader, val_loader, test_loader):
    print(f"\n--- ablation run: {tag} ---")
    set_seed(cfg_local["seed"], deterministic=cfg_local["deterministic"])
    model = init_with(cfg_local, device, pretrain_ckpt)
    loss_fn = losses_mod.build_loss(loss_name)
    ckpt = Path(cfg_local["paths"]["checkpoints"]) / f"ablation_{tag}.pt"
    train_mod.train_soft(model, train_loader, val_loader, loss_fn, cfg_local, device,
                         str(ckpt), log_prefix=f"[abl/{tag}] ")
    state = load_checkpoint(ckpt, map_location=device)["model"]
    model.load_state_dict(state)
    res, _ = eval_mod.evaluate(model, test_loader, device, cfg_local["eval"]["precision_at_k"])
    return res


def ablation_a_init(cfg, device, pretrain_ckpt, train_loader, val_loader, test_loader, loss_name):
    rows = []
    for init in ["random", "imagenet", "hard_pretrain"]:
        c = copy.deepcopy(cfg)
        c["model"]["init"] = init
        # shorten epochs a bit for ablations
        c["train"]["epochs"] = min(cfg["train"]["epochs"], 30)
        res = train_and_eval(c, f"A_init_{init}", device, pretrain_ckpt,
                              loss_name, train_loader, val_loader, test_loader)
        rows.append({"ablation": "A_backbone_init", "variant": init, **res.to_row(f"init={init}")})
    return rows


def ablation_b_loss_from_cache(cfg, device, test_loader):
    """Re-uses the per-loss checkpoints saved in script 02 — no extra training."""
    rows = []
    for loss_name in ["kl", "jsd", "composite"]:
        ckpt_path = Path(cfg["paths"]["checkpoints"]) / f"best_{loss_name}.pt"
        if not ckpt_path.exists():
            print(f"[abl/B] missing {ckpt_path}, skipping")
            continue
        model = models_mod.build_model(head_kind=cfg["model"]["head"]).to(device)
        model.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"])
        res, _ = eval_mod.evaluate(model, test_loader, device, cfg["eval"]["precision_at_k"])
        rows.append({"ablation": "B_loss_function", "variant": loss_name, **res.to_row(f"loss={loss_name}")})
    return rows


def ablation_d_head(cfg, device, pretrain_ckpt, train_loader, val_loader, test_loader, loss_name):
    rows = []
    for head in ["linear", "mlp", "temperature"]:
        c = copy.deepcopy(cfg)
        c["model"]["head"] = head
        c["train"]["epochs"] = min(cfg["train"]["epochs"], 30)
        res = train_and_eval(c, f"D_head_{head}", device, pretrain_ckpt,
                              loss_name, train_loader, val_loader, test_loader)
        rows.append({"ablation": "D_head_arch", "variant": head, **res.to_row(f"head={head}")})
    return rows


def ablation_c_training_strategy(cfg, device, pretrain_ckpt, train_loader, val_loader, test_loader, loss_name):
    """Compare soft-only training vs hard-pretrain -> soft fine-tune."""
    rows = []
    for strategy in ["soft_only", "pretrain_then_finetune"]:
        c = copy.deepcopy(cfg)
        c["model"]["init"] = "random" if strategy == "soft_only" else "hard_pretrain"
        c["train"]["epochs"] = min(cfg["train"]["epochs"], 30)
        res = train_and_eval(c, f"C_strategy_{strategy}", device, pretrain_ckpt,
                              loss_name, train_loader, val_loader, test_loader)
        rows.append({"ablation": "C_training_strategy", "variant": strategy, **res.to_row(f"strategy={strategy}")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-c", action="store_true", help="Also run the bonus 4th ablation (training strategy).")
    ap.add_argument("--loss-for-other-ablations", default="composite",
                    help="Loss to use when comparing init/head/strategy.")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    set_seed(cfg["seed"], deterministic=cfg["deterministic"])
    ensure_dirs(cfg)
    device = get_device()
    print(f"device: {device}")

    pretrain_ckpt = Path(cfg["paths"]["checkpoints"]) / "pretrain_cifar10.pt"
    if not pretrain_ckpt.exists():
        raise FileNotFoundError(
            "Pretrain checkpoint missing. Run scripts/02_train_all_losses.py first."
        )

    train_loader, val_loader, test_loader = make_loaders(cfg)

    all_rows = []
    all_rows.extend(ablation_a_init(cfg, device, str(pretrain_ckpt),
                                     train_loader, val_loader, test_loader,
                                     args.loss_for_other_ablations))
    all_rows.extend(ablation_b_loss_from_cache(cfg, device, test_loader))
    all_rows.extend(ablation_d_head(cfg, device, str(pretrain_ckpt),
                                     train_loader, val_loader, test_loader,
                                     args.loss_for_other_ablations))
    if args.include_c:
        all_rows.extend(ablation_c_training_strategy(cfg, device, str(pretrain_ckpt),
                                                      train_loader, val_loader, test_loader,
                                                      args.loss_for_other_ablations))

    df = pd.DataFrame(all_rows)
    out_csv = Path(cfg["paths"]["tables"]) / "ablation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nablation results -> {out_csv}")
    print(df.to_string(index=False))

    # required visualization: one ablation summary chart on KL_mean
    fig_path = Path(cfg["paths"]["figures"]) / "ablation_summary_KL.png"
    df_plot = df.copy()
    df_plot["model"] = df_plot["ablation"] + " | " + df_plot["variant"]
    viz.plot_grouped_loss_comparison(df_plot[["model", "KL_mean"]], "KL_mean", str(fig_path))


if __name__ == "__main__":
    main()
