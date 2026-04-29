"""Step 2: train the model under each loss function with an identical
training protocol (same backbone, optimizer, schedule, batch size,
augmentation), so the loss comparison is fair.

Default losses trained: KL (mandatory baseline), JSD, custom composite.
Pass --include-emd to also train the EMD bonus loss.
Pass --include-cos to also train the cosine loss (useful for the loss-comparison ablation).

Usage:
    python scripts/02_train_all_losses.py
    python scripts/02_train_all_losses.py --include-emd --include-cos
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import data as data_mod
from src import models as models_mod
from src import losses as losses_mod
from src import train as train_mod
from src import viz
from src.utils import (load_config, set_seed, ensure_dirs, get_device,
                       count_parameters, save_json, load_checkpoint)


def build_loaders(cfg, indices_train, indices_val):
    train_ds = data_mod.CIFAR10HSoft(
        cfg["paths"]["data_root"], cfg["paths"]["cifar10h_probs"],
        indices=indices_train,
        transform=data_mod.get_train_transform(
            cfg["augment"]["random_crop_padding"],
            cfg["augment"]["random_horizontal_flip"],
        ),
    )
    val_ds = data_mod.CIFAR10HSoft(
        cfg["paths"]["data_root"], cfg["paths"]["cifar10h_probs"],
        indices=indices_val,
        transform=data_mod.get_eval_transform(),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def maybe_pretrain(cfg, device, ckpt_path):
    """Hard-label pretraining on CIFAR-10 (50,000 images). Skip if already done."""
    if Path(ckpt_path).exists():
        print(f"[pretrain] cached at {ckpt_path}")
        return
    print("[pretrain] running CIFAR-10 hard-label pretraining...")
    model = models_mod.build_model(head_kind="linear").to(device)
    train_ds = data_mod.cifar10_train_dataset(
        cfg["paths"]["data_root"],
        transform=data_mod.get_train_transform(
            cfg["augment"]["random_crop_padding"],
            cfg["augment"]["random_horizontal_flip"],
        ),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    train_mod.pretrain_hard(model, train_loader, cfg, device, ckpt_path)


def init_model(cfg, device, pretrain_ckpt):
    model = models_mod.build_model(head_kind=cfg["model"]["head"]).to(device)
    init = cfg["model"]["init"]
    if init == "random":
        pass
    elif init == "imagenet":
        models_mod.init_from_imagenet_resnet18(model)
    elif init == "hard_pretrain":
        if not Path(pretrain_ckpt).exists():
            raise FileNotFoundError(
                f"hard_pretrain init requested but {pretrain_ckpt} missing. "
                "Run with the default config first or set model.init=random."
            )
        state = load_checkpoint(pretrain_ckpt, map_location=device)["model"]
        # only the backbone needs to load; head is task-specific
        own = model.state_dict()
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        model.load_state_dict(own, strict=False)
    else:
        raise ValueError(f"Unknown init: {init}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-emd", action="store_true")
    ap.add_argument("--include-cos", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    set_seed(cfg["seed"], deterministic=cfg["deterministic"])
    ensure_dirs(cfg)
    device = get_device()
    print(f"device: {device}")

    splits = np.load(Path(cfg["paths"]["tables"]) / "splits.npz")
    train_loader, val_loader = build_loaders(cfg, splits["train_idx"], splits["val_idx"])

    # Pretrain backbone once on CIFAR-10 hard labels.
    pretrain_ckpt = Path(cfg["paths"]["checkpoints"]) / "pretrain_cifar10.pt"
    maybe_pretrain(cfg, device, str(pretrain_ckpt))

    # Architecture diagram + parameter count summary table.
    viz.plot_architecture_diagram(str(Path(cfg["paths"]["figures"]) / "architecture.png"))
    param_rows = []
    for head_kind in ["linear", "mlp", "temperature"]:
        m = models_mod.build_model(head_kind=head_kind)
        param_rows.append({
            "variant": f"resnet18_cifar + {head_kind}_head",
            "trainable_params": count_parameters(m),
        })
    import pandas as pd
    pd.DataFrame(param_rows).to_csv(
        Path(cfg["paths"]["tables"]) / "param_counts.csv", index=False)
    print(f"param counts written to {cfg['paths']['tables']}/param_counts.csv")

    # Loss list to train.
    loss_specs = [("kl", {}), ("jsd", {}), ("composite", {"lambda_h": 0.5, "gamma": 1.0})]
    if args.include_cos:
        loss_specs.append(("cosine", {}))
    if args.include_emd:
        loss_specs.append(("emd", {}))

    histories = {}
    for loss_name, loss_kwargs in loss_specs:
        print(f"\n=== Training with loss = {loss_name} ===")
        model = init_model(cfg, device, str(pretrain_ckpt))
        loss_fn = losses_mod.build_loss(loss_name, **loss_kwargs)
        ckpt_path = Path(cfg["paths"]["checkpoints"]) / f"best_{loss_name}.pt"
        hist = train_mod.train_soft(
            model, train_loader, val_loader, loss_fn, cfg, device,
            str(ckpt_path), log_prefix=f"[{loss_name}] ",
        )
        histories[loss_name] = {
            "train_loss": hist.train_loss,
            "val_loss": hist.val_loss,
            "best_epoch": hist.best_epoch,
            "best_val_loss": hist.best_val_loss,
        }
        viz.plot_training_curves(
            hist.train_loss, hist.val_loss, hist.val_metric,
            str(Path(cfg["paths"]["figures"]) / f"training_curves_{loss_name}.png"),
            title=loss_name,
        )

    save_json(histories, Path(cfg["paths"]["logs"]) / "training_histories.json")
    print("done.")


if __name__ == "__main__":
    main()
