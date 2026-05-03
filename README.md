# Predicting Human Annotator Disagreement on CIFAR-10H

---


## Abstract

Standard image classifiers are trained to predict a single hard label per image, treating any disagreement among human annotators as label noise to be averaged away. This project takes the opposite view: we treat human disagreement as the *signal*. We train a deep neural network to predict the full 10-class probability distribution that approximately 50 human annotators produced for each image in the CIFAR-10H dataset, rather than just its majority class. The result is a model that learns not only what an image contains, but also how ambiguous it is.

We compare three loss functions under an identical training protocol — KL divergence (the standard baseline), Jensen–Shannon divergence, and a custom composite loss combining KL with an entropy-matching penalty and focal weighting on high-disagreement images — and evaluate them on distribution-level metrics, uncertainty correlation with humans, and top-K retrieval of ambiguous cases.

---


## Project objectives

1. Build a CNN that predicts soft annotator distributions rather than hard class labels.
2. Compare three loss functions, including one custom-designed loss, under matched training conditions.
3. Evaluate using distribution divergences, entropy correlation with humans, and Precision@K for ambiguity retrieval.
4. Conduct ablation studies on backbone initialisation, loss function, and prediction head.
5. Test robustness under annotator subsampling and out-of-distribution image corruptions.
6. Provide qualitative explanations of model behaviour through Grad-CAM and failure case analysis.

---

## Methodology

### Data

The project uses two complementary datasets. CIFAR-10 provides 50,000 training and 10,000 test images at 32×32 resolution with single hard labels. CIFAR-10H provides soft annotator distributions for the 10,000 CIFAR-10 *test* images, with approximately 50 independent human annotators per image.

The 10,000 CIFAR-10H images are split with a fixed seed into 6,000 training, 2,000 validation, and 2,000 test. The 50,000 CIFAR-10 training images are used only for backbone pretraining, never as soft-label targets — this asymmetry between hard-label data (abundant) and soft-label data (scarce) is a defining constraint of the problem.

Sanity checks confirm that all annotator distributions sum to 1.0 within 1×10⁻⁴ tolerance, and that the majority-vote labels agree with the original CIFAR-10 hard labels on 99.21% of images.


### Model architecture

We use a CIFAR-adapted ResNet-18 backbone — a 3×3 stem with no max-pool, replacing the 7×7 stride-2 stem of the standard ImageNet variant which would discard too much spatial information at 32×32 resolution. The backbone produces a 512-dimensional feature vector, which is passed through a linear head to produce 10 logits, then softmax-normalised. Total trainable parameters: 11.17 million.


### Loss functions

Three loss functions are compared under identical training conditions (same backbone, optimiser, learning rate schedule, data augmentation, and early stopping criterion). KL divergence serves as the standard baseline for distribution matching. Jensen–Shannon divergence provides a symmetric, bounded alternative. The custom composite loss is defined as `KL(p‖q) + λ·|H(p) − H(q)| + γ·focal_weight(H(p))·KL(p‖q)`, where the entropy term penalises mismatch in the *amount* of disagreement and the focal weight up-weights high-entropy images, which are a minority of the dataset and otherwise underrepresented in the gradient.


### Training protocol

The backbone is first pretrained on CIFAR-10 hard labels for 30 epochs, reaching 99.4% accuracy. It is then fine-tuned on the 6,000 CIFAR-10H soft labels for up to 50 epochs with early stopping on validation KL (patience 8). Mixed-precision training is used where supported.


### Evaluation

Models are evaluated on the held-out 2,000-image test set using distribution divergences (mean KL, mean JSD, cosine similarity), uncertainty correlation (Pearson and Spearman correlation between predicted and true entropy), and Precision@K for K ∈ {100, 200, 500} — the overlap between the top-K most-uncertain images by predicted entropy and by true human entropy.


### Robustness analysis

Two robustness checks are conducted. The annotator subsampling experiment resamples per-image label distributions to simulate evaluation with 5, 10, 20, 30, 40, or 50 annotators per image, testing how the apparent error and apparent ground-truth quality vary with crowd size. The out-of-distribution corruption experiment applies Gaussian noise, Gaussian blur, and contrast reduction at five severity levels each, with predicted entropy tracked as a function of severity.


### Explainability

Grad-CAM heatmaps are computed on the final convolutional layer of the backbone for both low-entropy (human-confident) and high-entropy (human-ambiguous) images, allowing direct visual comparison of where the model attends in each case. Failure case analysis selects the test images with the largest absolute mismatch between predicted and true entropy, with side-by-side bar plots of the human and model distributions.

---

## How to run

The project requires Python 3.10+ and PyTorch 2.x. A GPU is strongly recommended — a single Colab T4 is sufficient for the full pipeline; CPU is impractically slow.

```bash
# 1. environment setup
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. one-time CIFAR-10H download
mkdir -p outputs/data
wget -O outputs/data/cifar10h-probs.npy \
  https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy
```

The pipeline is then executed in order:

```bash
python scripts/01_prepare_data.py     # data preparation, sanity checks, dataset plots
python scripts/02_train_all_losses.py # train under each loss function
python scripts/03_run_ablations.py    # ablation studies
python scripts/04_evaluate_all.py     # core test-set evaluation
python scripts/05_robustness.py       # robustness checks
python scripts/06_explain.py          # Grad-CAM and failure cases
```

All outputs are written to `outputs/`: figures to `outputs/figures/`, tables to `outputs/tables/`, model checkpoints to `outputs/checkpoints/`.

---


## Summary of results

The KL-trained ResNet-18 is the strongest model on the bulk distribution metrics, achieving a test-set KL of 0.267 and Spearman correlation of 0.434 between its predicted entropies and the true human entropies. The custom composite loss is competitive on the bulk metrics and wins on Precision@100 (0.18 vs. 0.15 for KL), confirming that the focal weighting successfully redirects the model's attention toward the long tail of ambiguous images at a small cost to mean performance. The JSD baseline failed to learn meaningfully under our protocol, with its training loss collapsing near zero from the first epoch — a signature of a degenerate optimum.

The annotator-subsampling experiment surfaced a methodological observation that generalises beyond this dataset: the entropy of the resampled "ground truth" labels falls from 0.212 (50 annotators) to 0.140 (5 annotators) as the crowd shrinks, while the apparent KL of the model rises from 0.328 to 0.386. The model itself does not change — it is the noisier evaluation target that drives the apparent error. Soft-label models should therefore be benchmarked against the largest annotator pool available.

The corruption analysis showed that the model handles Gaussian blur and contrast reduction gracefully, with predicted entropy rising monotonically with severity. Under Gaussian noise, however, predicted entropy peaks at moderate severity and then falls — an indication that at high noise levels the model becomes confidently wrong rather than appropriately uncertain, marking a clear target for future work.

Full results, training curves, ablation tables, and qualitative figures are available in `outputs/` and discussed in detail in the accompanying project report.

---


## Reproducibility

The random seed is set in `configs/default.yaml` and applied via `src/utils.py::set_seed`. With `deterministic: true`, cuDNN benchmarking is disabled, ensuring bit-for-bit reproducibility at a small cost to runtime.

---
