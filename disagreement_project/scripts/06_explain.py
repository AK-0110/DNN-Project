"""Step 6: explainability and analysis.

Produces:
  - Grad-CAM panel comparing low-disagreement vs high-disagreement images
  - Failure cases panel (largest |H(p) - H(q)|)
  - "manual inspection" image grid: highest-entropy images, with a CSV
    template ready for the team to fill in disagreement-source categories.
    The viva expectation is that this is then filled in manually.

Usage:
    python scripts/06_explain.py
    python scripts/06_explain.py --loss kl    # use a different trained model
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
from src import explain as explain_mod
from src import viz
from src.utils import (load_config, ensure_dirs, get_device, load_checkpoint,
                       set_seed)


def load_best_model(cfg, device, loss_name):
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / f"best_{loss_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} missing — train it first via script 02.")
    model = models_mod.build_model(head_kind=cfg["model"]["head"]).to(device)
    model.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="composite")
    ap.add_argument("--n-cam", type=int, default=4, help="images per row in Grad-CAM panel")
    ap.add_argument("--n-failures", type=int, default=8)
    ap.add_argument("--n-manual", type=int, default=20, help="size of the manual inspection set")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    set_seed(cfg["seed"], deterministic=cfg["deterministic"])
    ensure_dirs(cfg)
    device = get_device()

    splits = np.load(Path(cfg["paths"]["tables"]) / "splits.npz")
    test_ds = data_mod.CIFAR10HSoft(
        cfg["paths"]["data_root"], cfg["paths"]["cifar10h_probs"],
        indices=splits["test_idx"], transform=data_mod.get_eval_transform(),
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                             shuffle=False, num_workers=2, pin_memory=True)

    model = load_best_model(cfg, device, args.loss)
    _, artifacts = eval_mod.evaluate(model, test_loader, device, cfg["eval"]["precision_at_k"])

    # --- Grad-CAM on low/high disagreement ---
    low_pos, high_pos = explain_mod.select_extreme_entropy(artifacts, n=args.n_cam)
    cam = explain_mod.GradCAM(model, model.backbone.layer4)

    images_panel, cams_panel, titles = [], [], []
    for label, positions in (("low", low_pos), ("high", high_pos)):
        for pos in positions:
            img, _, _, _ = test_ds[int(pos)]
            x = img.unsqueeze(0).to(device)
            x.requires_grad_(False)
            heat = cam(x)
            images_panel.append(img)
            cams_panel.append(heat)
            top_class = int(np.argmax(artifacts["pred_probs"][pos]))
            titles.append(f"{label}\n{data_mod.CIFAR10_CLASSES[top_class]} "
                          f"(H={artifacts['true_entropy'][pos]:.2f})")
    cam.remove_hooks()
    viz.plot_gradcam_panel(
        images_panel, cams_panel, titles,
        str(Path(cfg["paths"]["figures"]) / f"gradcam_low_high_{args.loss}.png"),
    )
    print(f"Grad-CAM panel saved: low ({args.n_cam}) + high ({args.n_cam}) entropy images.")

    # --- failure case panel ---
    fail_pos = explain_mod.select_failure_cases(artifacts, n=args.n_failures)
    imgs, true_p, pred_p = [], [], []
    fail_rows = []
    for pos in fail_pos:
        img, _, _, dataset_idx = test_ds[int(pos)]
        imgs.append(img)
        true_p.append(artifacts["true_probs"][pos])
        pred_p.append(artifacts["pred_probs"][pos])
        fail_rows.append({
            "rank": len(fail_rows) + 1,
            "test_position": int(pos),
            "cifar10h_index": int(dataset_idx),
            "true_entropy": float(artifacts["true_entropy"][pos]),
            "pred_entropy": float(artifacts["pred_entropy"][pos]),
            "abs_entropy_error": float(abs(artifacts["true_entropy"][pos] - artifacts["pred_entropy"][pos])),
            "true_top": int(np.argmax(artifacts["true_probs"][pos])),
            "pred_top": int(np.argmax(artifacts["pred_probs"][pos])),
            "hypothesis": "(write a brief hypothesis here)",
        })
    pd.DataFrame(fail_rows).to_csv(
        Path(cfg["paths"]["tables"]) / f"failure_cases_{args.loss}.csv", index=False)
    viz.plot_failure_cases(
        imgs, np.array(true_p), np.array(pred_p),
        str(Path(cfg["paths"]["figures"]) / f"failure_cases_{args.loss}.png"),
    )
    print(f"Failure case panel and table saved.")

    # --- manual inspection set ---
    H = artifacts["true_entropy"]
    manual_pos = np.argsort(-H)[:args.n_manual]
    manual_rows = []
    manual_imgs, manual_probs = [], []
    for pos in manual_pos:
        img, _, _, dataset_idx = test_ds[int(pos)]
        manual_imgs.append(img)
        manual_probs.append(artifacts["true_probs"][pos])
        manual_rows.append({
            "test_position": int(pos),
            "cifar10h_index": int(dataset_idx),
            "true_entropy": float(artifacts["true_entropy"][pos]),
            "true_top_class": data_mod.CIFAR10_CLASSES[int(np.argmax(artifacts["true_probs"][pos]))],
            "true_top_prob": float(artifacts["true_probs"][pos].max()),
            "disagreement_source_category":
                "(ambiguous_object_identity / poor_image_quality / "
                "multi_object / boundary_case / other)",
            "notes": "(your reasoning)",
        })
    pd.DataFrame(manual_rows).to_csv(
        Path(cfg["paths"]["tables"]) / "manual_disagreement_inspection.csv", index=False)

    viz.plot_low_high_entropy_grid(
        manual_imgs[:8], manual_probs[:8],
        manual_imgs[8:16] if len(manual_imgs) >= 16 else manual_imgs[8:],
        manual_probs[8:16] if len(manual_probs) >= 16 else manual_probs[8:],
        str(Path(cfg["paths"]["figures"]) / "manual_inspection_grid.png"),
    )
    print(f"Manual inspection CSV template saved with {args.n_manual} rows. "
          "Each team member should fill in the category column for the viva.")


if __name__ == "__main__":
    main()
