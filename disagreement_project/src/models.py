"""Model architectures.

Backbone: a CIFAR-adapted ResNet-18 (3x3 stem, no max-pool) is the default,
because the standard ImageNet ResNet-18 stem (7x7 stride-2 conv + max-pool)
destroys spatial information on 32x32 inputs.

Heads:
    - LinearHead         : single linear layer + softmax (default)
    - MLPHead            : 2-layer MLP + softmax
    - TemperatureHead    : linear + softmax with a learnable temperature

All heads output a 10-dimensional probability distribution. The forward pass
returns LOGITS; softmax is applied inside the loss / evaluation code so that
log-softmax can be used numerically safely.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CIFAR-style ResNet-18.
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18CIFAR(nn.Module):
    """ResNet-18 with a 3x3 stem and no initial max-pool. Feature dim = 512."""

    feature_dim = 512

    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, 512)
        return x


# ---------------------------------------------------------------------------
# Heads.
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, in_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feats):
        return self.fc(feats)        # logits


class MLPHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden: int = 256, num_classes: int = 10,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feats):
        return self.net(feats)


class TemperatureHead(nn.Module):
    """Linear logits scaled by a learnable temperature.

    Smaller T sharpens the softmax, larger T flattens it. Useful when the
    predicted distributions are systematically too sharp or too flat.
    """
    def __init__(self, in_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        # store log-temperature for stability; T = exp(log_T)
        self.log_t = nn.Parameter(torch.zeros(1))

    def forward(self, feats):
        logits = self.fc(feats)
        T = self.log_t.exp().clamp(min=1e-2, max=1e2)
        return logits / T


# ---------------------------------------------------------------------------
# Full model.
# ---------------------------------------------------------------------------

class DisagreementModel(nn.Module):
    """Backbone + head. forward() returns logits."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=-1)


def build_model(head_kind: str = "linear") -> DisagreementModel:
    backbone = ResNet18CIFAR()
    head_kind = head_kind.lower()
    if head_kind == "linear":
        head = LinearHead(backbone.feature_dim, 10)
    elif head_kind == "mlp":
        head = MLPHead(backbone.feature_dim, 256, 10)
    elif head_kind == "temperature":
        head = TemperatureHead(backbone.feature_dim, 10)
    else:
        raise ValueError(f"Unknown head kind: {head_kind}")
    return DisagreementModel(backbone, head)


def init_from_imagenet_resnet18(model: DisagreementModel, strict: bool = False) -> None:
    """Best-effort port of torchvision ImageNet ResNet-18 weights into the
    CIFAR-style backbone. The 7x7 stem and max-pool from ImageNet are simply
    skipped (CIFAR backbone uses a 3x3 stem instead).
    """
    from torchvision.models import resnet18, ResNet18_Weights
    try:
        src = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    except Exception:
        # offline fallback: leave random init
        return
    own = model.backbone.state_dict()
    matched, skipped = 0, 0
    for k, v in src.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            matched += 1
        else:
            skipped += 1
    model.backbone.load_state_dict(own, strict=False)
    print(f"[init_from_imagenet] matched={matched} skipped={skipped}")
