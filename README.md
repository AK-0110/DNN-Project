# Predicting Human Annotator Disagreement on CIFAR-10H

This project builds a deep neural network that predicts the **full human annotator label distribution** for CIFAR-10 images, rather than predicting a single hard class. The model learns what makes images inherently ambiguous.

## Project structure

```
disagreement_project/
├── README.md                       <- you are here
├── requirements.txt
├── src/
│   ├── data.py                     <- CIFAR-10 + CIFAR-10H loading, splits, sanity checks
│   ├── models.py                   <- ResNet-18 (CIFAR-adapted) + alternative heads
│   ├── losses.py                   <- KL, JSD, cosine, EMD, custom composite loss
│   ├── train.py                    <- training loop with early stopping
│   ├── evaluate.py                 <- distribution metrics, entropy correlations, Precision@K
│   ├── explain.py                  <- Grad-CAM, failure-case analysis
│   ├── robustness.py               <- annotator subsampling, corruptions, class-wise eval
│   ├── viz.py                      <- all plotting helpers
│   └── utils.py                    <- seeding, checkpoints, config helpers
├── scripts/
│   ├── 01_prepare_data.py          <- downloads CIFAR-10/10H, runs sanity checks, makes plots
│   ├── 02_train_all_losses.py      <- trains the model under each loss function
│   ├── 03_run_ablations.py         <- ablation studies (init, head, training strategy)
│   ├── 04_evaluate_all.py          <- core performance evaluation + comparison table
│   ├── 05_robustness.py            <- robustness checks
│   └── 06_explain.py               <- Grad-CAM + failure cases + manual inspection grid
├── configs/
│   └── default.yaml                <- single config file controlling all runs
├── notebooks/
│   └── walkthrough.ipynb           <- end-to-end notebook (optional, mirrors scripts)
└── outputs/                        <- created at runtime: checkpoints, figures, tables
```

## Setup

Tested with Python 3.10+, PyTorch 2.x, single GPU (T4/V100/A100/RTX-class) or CPU (slow).

```bash
# 1. create a fresh environment (recommended)
python -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
```

`requirements.txt` pins the libraries used: torch, torchvision, numpy, pandas, matplotlib, scikit-learn, scipy, seaborn, pyyaml, pot (for EMD), grad-cam, tqdm.

## Data

Two datasets are used together:

* **CIFAR-10** — 50,000 train + 10,000 test images, 32×32, hard labels. Downloaded automatically by torchvision.
* **CIFAR-10H** — 10,000 soft-label distributions, one per CIFAR-10 *test* image, each from ~50 human annotators. The file used is `cifar10h-probs.npy` (shape `(10000, 10)`, rows sum to 1). Download once from the [CIFAR-10H repo](https://github.com/jcpeterson/cifar-10h) and place it at `data/cifar10h-probs.npy`. The data prep script will check for it and tell you what to do if it is missing.

The 10,000 CIFAR-10H images are split (with a fixed seed) into:

| Partition | Count |
| --- | --- |
| Training (soft labels) | 6,000 |
| Validation | 2,000 |
| Testing | 2,000 |

The 50,000 CIFAR-10 train images are used **only** for hard-label pretraining or backbone initialisation, never as soft-label targets — this is the data asymmetry the assignment calls out.

## How to run end-to-end

The scripts are numbered and meant to be run in order. Each script writes to `outputs/` so later scripts can pick up its results.

```bash
# 0. (one-time) download CIFAR-10H probs to outputs/data/cifar10h-probs.npy
mkdir -p outputs/data
# wget -O outputs/data/cifar10h-probs.npy https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy

# 1. download CIFAR-10, align with CIFAR-10H, run sanity checks, produce data-stage plots
python scripts/01_prepare_data.py

# 2. train under each loss (KL, JSD, custom composite). Optional: --include-emd for the bonus loss.
python scripts/02_train_all_losses.py

# 3. ablation studies — picks 3 of {backbone init, loss, training strategy, head}
python scripts/03_run_ablations.py

# 4. core performance evaluation: distribution metrics, entropy correlations, Precision@K, summary table
python scripts/04_evaluate_all.py

# 5. robustness — picks 2 of {annotator subsampling, corruptions, class-conditional}
python scripts/05_robustness.py

# 6. explainability — Grad-CAM on low/high disagreement, failure cases, manual inspection grid
python scripts/06_explain.py
```

All figures land in `outputs/figures/`, all tables in `outputs/tables/`, all checkpoints in `outputs/checkpoints/`.

## What is implemented vs. the assignment requirements

| Requirement | Where |
| --- | --- |
| Soft-label distribution prediction (not hard classifier) | `src/models.py`, `src/train.py` |
| CIFAR-10 + CIFAR-10H pipeline + 6k/2k/2k split | `src/data.py`, `scripts/01_prepare_data.py` |
| Sanity checks (rows sum to 1, alignment, entropy histogram) | `scripts/01_prepare_data.py` |
| Required data-stage plots (entropy histogram, per-class entropy, confusion-style matrix, low/high entropy grid) | `scripts/01_prepare_data.py` |
| CNN backbone adapted to 32×32 (CIFAR-style ResNet-18) | `src/models.py` |
| Architecture diagram + parameter count table | `outputs/figures/architecture.png`, `outputs/tables/param_counts.csv` |
| KL Divergence (mandatory) | `src/losses.py` |
| Second standard loss: JSD + cosine + soft-target CE all available | `src/losses.py` |
| Custom composite loss (KL + entropy-error + focal-style high-disagreement weighting) | `src/losses.py` (`CompositeDisagreementLoss`) |
| EMD / Wasserstein loss (bonus) | `src/losses.py` |
| Same training protocol across loss comparisons | `scripts/02_train_all_losses.py` |
| Loss curves + validation metric curve | `src/viz.py`, written during training |
| Core performance (KL, JSD, cosine, Pearson/Spearman entropy, Precision@K=100/200/500) | `src/evaluate.py`, `scripts/04_evaluate_all.py` |
| Summary comparison table across losses | `outputs/tables/loss_comparison.csv` |
| 3 ablations: backbone init, loss, training strategy | `scripts/03_run_ablations.py` |
| 2 robustness checks: annotator subsampling, OOD corruptions | `src/robustness.py`, `scripts/05_robustness.py` |
| Grad-CAM on low/high disagreement images | `src/explain.py`, `scripts/06_explain.py` |
| Failure case analysis | `scripts/06_explain.py` |
| Manual disagreement source inspection grid | `scripts/06_explain.py` |

## Notes on choices the assignment asked us to justify

* **Backbone**: CIFAR-style ResNet-18 (3×3 stem, no max-pool) rather than ImageNet ResNet-18 — 32×32 inputs lose too much spatial information through the ImageNet 7×7 stride-2 stem and max-pool.
* **Head**: single linear + temperature-scaled softmax. Compared against an MLP head in the head-architecture ablation.
* **Hard/soft asymmetry**: the default training strategy is "hard-label pretrain on 50k CIFAR-10 → soft-label fine-tune on 6k CIFAR-10H". This is compared against random init and against soft-label-only training in the training-data-strategy ablation.
* **Custom loss**: KL + λ·|H(p) − H(q)| + γ·focal weighting on high-entropy images. Justification: pure KL drives mean behaviour but does not pressure the model to match the *shape* of disagreement (entropy), and high-disagreement images are a minority of the dataset and otherwise get drowned out.

## Reproducibility

Seed is set in `src/utils.py::set_seed` and read from `configs/default.yaml`. cuDNN benchmarking is disabled when `deterministic: true` (slower but reproducible).
