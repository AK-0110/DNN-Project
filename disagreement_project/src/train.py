"""Training loops.

Two functions:

  - train_soft       : trains a DisagreementModel on soft annotator labels.
  - pretrain_hard    : pretrains the backbone on CIFAR-10 hard labels (50k).

Both share early stopping, AMP support, and consistent logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import utils


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_metric: list = field(default_factory=list)   # e.g. mean KL on val
    best_epoch: int = -1
    best_val_loss: float = float("inf")


def _build_optimizer(params, kind: str, lr: float, wd: float, momentum: float = 0.9):
    kind = kind.lower()
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    raise ValueError(f"Unknown optimizer: {kind}")


def _build_scheduler(opt, kind: str, epochs: int, warmup: int = 0):
    kind = (kind or "none").lower()
    if kind == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup))
    elif kind == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.1)
    else:
        sched = None
    return sched


def _val_kl(model: nn.Module, loader: DataLoader, device) -> float:
    """Mean KL(p||q) on the loader. Used as a model-selection metric."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, p, _, _ = batch
            x, p = x.to(device, non_blocking=True), p.to(device, non_blocking=True)
            log_q = F.log_softmax(model(x), dim=-1)
            log_p = torch.log(p.clamp(min=1e-12))
            kl = (p * (log_p - log_q)).sum(dim=-1)
            total += float(kl.sum().item())
            n += x.size(0)
    return total / max(1, n)


def train_soft(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    cfg: dict,
    device: torch.device,
    ckpt_path: str,
    log_prefix: str = "",
) -> TrainHistory:
    """Soft-label distribution training with early stopping.

    Saves the best (min validation KL) checkpoint to ckpt_path.
    """
    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"].get("warmup_epochs", 0)
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"

    opt = _build_optimizer(
        model.parameters(),
        cfg["train"]["optimizer"],
        cfg["train"]["lr"],
        cfg["train"]["weight_decay"],
    )
    sched = _build_scheduler(opt, cfg["train"]["scheduler"], epochs, warmup)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    stopper = utils.EarlyStopper(patience=cfg["train"]["early_stopping_patience"])
    hist = TrainHistory()

    for epoch in range(epochs):
        # linear warmup
        if epoch < warmup:
            for g in opt.param_groups:
                g["lr"] = cfg["train"]["lr"] * (epoch + 1) / max(1, warmup)

        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{log_prefix}epoch {epoch+1}/{epochs}", leave=False)
        for x, p, _hard, _idx in pbar:
            x = x.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, p)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{running/n:.4f}")
        train_loss = running / max(1, n)

        if sched is not None and epoch >= warmup:
            sched.step()

        val_loss = _val_kl(model, val_loader, device)
        hist.train_loss.append(train_loss)
        hist.val_loss.append(val_loss)
        hist.val_metric.append(val_loss)   # same as val KL here

        is_best = stopper.step(val_loss)
        if is_best:
            hist.best_epoch = epoch
            hist.best_val_loss = val_loss
            utils.save_checkpoint(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": cfg,
                },
                ckpt_path,
            )
        print(f"{log_prefix}epoch {epoch+1:3d}  train={train_loss:.4f}  val_KL={val_loss:.4f}"
              + ("  <- best" if is_best else ""))
        if stopper.should_stop:
            print(f"{log_prefix}early stop at epoch {epoch+1}")
            break

    return hist


# ---------------------------------------------------------------------------
# Hard-label pretraining on CIFAR-10 (50,000 images).
# ---------------------------------------------------------------------------

def pretrain_hard(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    ckpt_path: str,
) -> None:
    """Standard cross-entropy training on CIFAR-10 hard labels.

    The 50k hard-label images are used here ONLY for representation learning.
    They are never treated as soft-label disagreement targets.
    """
    epochs = cfg["pretrain"]["epochs"]
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    opt = _build_optimizer(
        model.parameters(),
        cfg["pretrain"]["optimizer"],
        cfg["pretrain"]["lr"],
        cfg["pretrain"]["weight_decay"],
        momentum=cfg["pretrain"].get("momentum", 0.9),
    )
    sched = _build_scheduler(opt, cfg["pretrain"]["scheduler"], epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running, n, correct = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"[pretrain] epoch {epoch+1}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
            correct += int((logits.argmax(-1) == y).sum().item())
            pbar.set_postfix(loss=f"{running/n:.4f}", acc=f"{correct/n:.3f}")
        if sched is not None:
            sched.step()
        print(f"[pretrain] epoch {epoch+1:3d}  loss={running/n:.4f}  acc={correct/n:.3f}")

    utils.save_checkpoint({"model": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"[pretrain] saved to {ckpt_path}")
