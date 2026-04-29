"""Step 5: robustness checks.

Compulsory: run any 2 of {A annotator subsampling, B OOD corruptions,
C class-conditional}. Default runs A and B. Pass --include-c to also
run the bonus 3rd check.

Usage:
    python scripts/05_robustness.py
    python scripts/05_robustness.py --include-c
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import data as data_mod
from src import models as models_mod
from src import evaluate as eval_mod
from src import robustness as rob
from src import viz
from src.utils import (load_config, ensure_dirs, get_device, load_checkpoint,
                       set_seed, save_json)


def make_test_loader_unnormalised(cfg):
    """Test loader where images are returned in [0, 1] range, not normalised.
    We normalise inside the eval loop after corrupting."""
    splits = np.load(Path(cfg["paths"]["tables"]) / "splits.npz")
    from torchvision import transforms
    raw_t = transforms.ToTensor()       # converts PIL -> [0,1] float tensor
    test_ds = data_mod.CIFAR10HSoft(
        cfg["paths"]["data_root"], cfg["paths"]["cifar10h_probs"],
        indices=splits["test_idx"], transform=raw_t,
    )
    return DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                      shuffle=False, num_workers=2, pin_memory=True)


def normalise(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(data_mod.CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(data_mod.CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def load_best_model(cfg, device, loss_name):
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / f"best_{loss_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} missing — train it first via script 02.")
    model = models_mod.build_model(head_kind=cfg["model"]["head"]).to(device)
    model.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Check A: annotator subsampling.
# ---------------------------------------------------------------------------

def check_annotator_subsampling(cfg, device, loss_name):
    """How does mean KL between *resampled* p and the model's q change as
    we use fewer annotators per image?"""
    print(f"\n[robustness A] annotator subsampling (model={loss_name})")
    splits = np.load(Path(cfg["paths"]["tables"]) / "splits.npz")
    test_ds = data_mod.CIFAR10HSoft(
        cfg["paths"]["data_root"], cfg["paths"]["cifar10h_probs"],
        indices=splits["test_idx"], transform=data_mod.get_eval_transform(),
    )
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=2, pin_memory=True)
    model = load_best_model(cfg, device, loss_name)
    P, Q, _, _ = eval_mod.collect_predictions(model, loader, device)
    rows = []
    for frac in cfg["annotator_subsample"]["fractions"]:
        P_sub = rob.resample_annotators(
            P, frac, cfg["annotator_subsample"]["approx_annotators_per_image"], seed=cfg["seed"])
        kl = eval_mod.kl_per_image(P_sub, Q)
        from src.evaluate import _entropy_bits
        Hp = _entropy_bits(P_sub); Hq = _entropy_bits(Q)
        rows.append({
            "fraction": frac,
            "approx_annotators": int(round(frac * cfg["annotator_subsample"]["approx_annotators_per_image"])),
            "KL_mean": float(kl.mean()),
            "KL_std": float(kl.std()),
            "Hp_mean": float(Hp.mean()),
            "Hq_mean": float(Hq.mean()),
        })
    df = pd.DataFrame(rows)
    out_csv = Path(cfg["paths"]["tables"]) / f"robustness_A_subsampling_{loss_name}.csv"
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Check B: OOD corruptions.
# ---------------------------------------------------------------------------

def check_corruptions(cfg, device, loss_name):
    print(f"\n[robustness B] OOD corruptions (model={loss_name})")
    loader = make_test_loader_unnormalised(cfg)
    model = load_best_model(cfg, device, loss_name)
    severities = cfg["corruptions"]["severities"]
    kinds = cfg["corruptions"]["types"]

    mean_pred_entropy = {k: [] for k in kinds}
    rows = []
    for kind in kinds:
        for sev in severities:
            preds = []
            with torch.no_grad():
                for x_raw, _, _, _ in loader:
                    x_raw = x_raw.to(device, non_blocking=True)
                    x_corr = rob.corrupt_batch(x_raw, kind, sev)
                    x = normalise(x_corr)
                    q = F.softmax(model(x), dim=-1).cpu().numpy()
                    preds.append(q)
            Q = np.concatenate(preds, axis=0)
            from src.evaluate import _entropy_bits
            Hq = _entropy_bits(Q)
            mean_pred_entropy[kind].append(float(Hq.mean()))
            rows.append({"kind": kind, "severity": sev, "mean_pred_entropy": float(Hq.mean())})
            print(f"  {kind:>16s} sev={sev}  mean H_q = {Hq.mean():.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg["paths"]["tables"]) / f"robustness_B_corruptions_{loss_name}.csv", index=False)
    viz.plot_corruption_response(
        severities, mean_pred_entropy,
        str(Path(cfg["paths"]["figures"]) / f"robustness_corruption_response_{loss_name}.png"),
    )
    return df


# ---------------------------------------------------------------------------
# Check C: class-conditional performance (bonus by default).
# ---------------------------------------------------------------------------

def check_class_conditional(cfg, device, loss_name):
    print(f"\n[robustness C] class-conditional performance (model={loss_name})")
    splits = np.load(Path(cfg["paths"]["tables"]) / "splits.npz")
    test_ds = data_mod.CIFAR10HSoft(
        cfg["paths"]["data_root"], cfg["paths"]["cifar10h_probs"],
        indices=splits["test_idx"], transform=data_mod.get_eval_transform(),
    )
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=2, pin_memory=True)
    model = load_best_model(cfg, device, loss_name)
    P, Q, Y, _ = eval_mod.collect_predictions(model, loader, device)
    per_class = rob.class_conditional_metrics(P, Q, Y)
    save_json(per_class, Path(cfg["paths"]["tables"]) / f"robustness_C_class_{loss_name}.json")
    viz.plot_per_class_bar(per_class, "KL_mean",
                            str(Path(cfg["paths"]["figures"]) / f"robustness_per_class_KL_{loss_name}.png"))
    viz.plot_per_class_bar(per_class, "pearson_H",
                            str(Path(cfg["paths"]["figures"]) / f"robustness_per_class_pearsonH_{loss_name}.png"))
    return per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="composite",
                    help="Which trained model to use for robustness checks.")
    ap.add_argument("--include-c", action="store_true", help="Also run the 3rd robustness check.")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    set_seed(cfg["seed"], deterministic=cfg["deterministic"])
    ensure_dirs(cfg)
    device = get_device()
    print(f"device: {device}")

    check_annotator_subsampling(cfg, device, args.loss)
    check_corruptions(cfg, device, args.loss)
    if args.include_c:
        check_class_conditional(cfg, device, args.loss)


if __name__ == "__main__":
    main()
