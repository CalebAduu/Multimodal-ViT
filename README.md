# Multimodal-ViT
# Interpretable Multimodal Vision Transformer Framework for Skin Cancer Diagnosis


---

## Overview

This repository contains the implementation of a novel multimodal architecture for 7-class skin lesion classification on the HAM10000 dataset. The model fuses dermoscopic images with structured clinical metadata (age, sex, anatomical site, lesion history) via **bidirectional cross-attention adapters** inserted into a hybrid CNN-ViT backbone, inspired by the design principles of BLIP and MedCLIP.

The key contribution is the `StableProjectedCrossAttnAdapter` — a bidirectional cross-attention module that allows image patch tokens and clinical metadata tokens to mutually attend to each other during feature extraction, enabling richer multimodal fusion than standard late-fusion MLP approaches.

---

## Architecture

```
Input Image (224×224)
        ↓
ResNet-50 CNN Stem       ← local dermoscopic texture features
        ↓
ViT-Base Transformer Blocks (0–7)   ← frozen
        ↓
ViT-Base Transformer Blocks (8–11)  ← fine-tuned
    + StableProjectedCrossAttnAdapter × 4
        ↑ ↕ bidirectional cross-attention
Clinical Metadata (12 tokens)
  • 4 numerical: age, normalised age, anatomical site risk score, images/lesion
  • 8 categorical: sex, anatomical site, age risk group, sun exposure history, etc.
        ↓
[img_cls ‖ mean(meta_tokens)] → Linear → 7 classes
```

**Backbone:** `vit_base_r50_s16_224` (ResNet-50 CNN stem + ViT-Base/16 transformer)  
**Metadata encoding:** per-feature token embeddings (not MLP late fusion)  
**Adapter gates:** initialised to 0.01 for stable fine-tuning of pretrained backbone

---

## Dataset

| Split | Source | Images |
|---|---|---|
| Train | HAM10000 + BCN20000 minority boost (DF, VASC, AKIEC) | ~11,000 |
| Validation | HAM10000 only | ~2,200 |
| Test | HAM10000 only | 1,471 |

Splits are **lesion-stratified** (70/15/15) — no lesion appears in more than one split. Class imbalance handled via `WeightedRandomSampler` (inverse-frequency weights).

**Labels:** NV (0), MEL (1), BCC (2), AKIEC (3), BKL (4), DF (5), VASC (6)

---

## Results

### Internal Test Set (HAM10000, n=1,471)

| Metric | Value |
|---|---|
| Overall Accuracy | 78.53% |
| Macro AUROC | 0.962 |
| Malignant Sensitivity (MEL+BCC+AKIEC) | 0.709 |
| Macro F1 | 0.688 |
| ECE (post temperature scaling) | 0.024 |

### External Validation

| Dataset | Modality | Metric | Score |
|---|---|---|---|
| Derm7pt | Dermoscopy | Macro AUROC | 0.853 |
| PAD-UFES-20 | Smartphone | Macro AUROC | 0.760 |
| Fitzpatrick17k | Clinical photo | Binary malignancy AUROC | 0.630 |
| DDI | Clinical photo | Binary malignancy AUROC | 0.700 |

### Ablation Study (malignant sensitivity, test set)

| Model | Mal Sens | Macro AUROC |
|---|---|---|
| Image-only ViT (Exp 1) | 0.593 | 0.932 |
| Late concat MLP fusion (Exp 2) | 0.579 | 0.956 |
| Hadamard fusion (Exp 3) | 0.626 | 0.959 |
| One-way cross-attention (Exp 4) | 0.594 | 0.955 |
| No BCN20000 boost (Exp 5) | 0.634 | 0.962 |
| **Champion (bidirectional, full data)** | **0.709** | **0.962** |

All pairwise malignant sensitivity comparisons are statistically significant (paired bootstrap, n=10,000, Holm-Bonferroni corrected, p < 0.05).

---

## Repository Structure

```
├── HAM-TestSplit-main.ipynb       # Main notebook — data pipeline, model, training, evaluation
├── HAM_Ablations.ipynb            # Ablation study notebook — 5 experiments + external baselines
├── saved_models/
│   ├── figures/                   # Thesis figures (ROC, PR, calibration, external validation)
│   ├── gradcam/                   # Score-CAM and Attention Rollout visualisations
│   ├── temperature_scaling.json   # Fitted temperature T for post-hoc calibration
│   ├── ablation_summary.csv
│   ├── ablation_significance_tests.csv
│   ├── metadata_importance_single.csv
│   └── metadata_importance_group.csv
└── README.md
```

> **Note:** Model checkpoints (`.pth` files) and datasets are not included due to file size. See Setup below.

---

## Setup

### Requirements

```bash
pip install torch torchvision timm
pip install scikit-learn pandas numpy matplotlib seaborn
pip install opencv-python tqdm scipy
```

### Dataset

1. Download [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) from Harvard Dataverse
2. Download [BCN20000](https://arxiv.org/abs/1908.02288) for minority class augmentation
3. Place images in `ham10000/images/` and metadata CSV in `ham10000/`

### Training

Open `HAM-TestSplit-main.ipynb` and run cells sequentially:
- Cells 1–15: data loading, feature engineering, splits, dataloaders
- Cell 16: `StableProjectedCrossAttnAdapter` definition
- Cell 17: `ViT_HAM10000_HybridCNN` model definition
- Cell 18: model instantiation
- Cell 19: training configuration (loss, optimiser, scheduler)
- Cell 20: training loop (50 epochs, patience 30)
- Cell 21+: evaluation, calibration, external validation, interpretability

---

## Key Design Decisions

**Why per-feature token embeddings over MLP?**  
Each clinical feature (age, sex, site, etc.) is encoded as a separate 768-dim token. This allows the cross-attention to learn feature-specific spatial relationships — e.g. "age matters for these image patches, anatomical site for those." An MLP compresses all metadata into one vector before fusion, losing individual feature-to-patch correspondence.

**Why bidirectional cross-attention?**  
Image tokens attend to metadata (metadata guides which regions are important) and metadata tokens attend to image tokens (image updates the clinical context). This is inspired by BLIP's bidirectional encoder, adapted for structured tabular metadata rather than text.

**Why near-zero gate initialisation?**  
Adapter gates initialised at 0.01 ensure the pretrained ViT representations are preserved at training start. The model first learns on stable pretrained features, then gradually incorporates the adapter contributions as gates grow.

---

## Interpretability

- **Score-CAM:** gradient-free CNN activation masking, reliable on frozen stems
- **Attention Rollout:** propagates transformer attention weights through all blocks
- **Metadata importance:** single-feature and group ablation on test set
- **Temperature scaling:** post-hoc calibration (ECE reduced from 0.135 → 0.024)

---

## Citation

If you use this code, please cite:

```
@mastersthesis{hull2025multimodal,
  author    = {Caleb},
  title     = {Interpretable Multimodal Vision Transformer Framework for Skin Cancer Diagnosis},
  school    = {University of Hull},
  year      = {2025},
  supervisor = {Dr Temitayo Matthew Fagbola}
}
```

---

## Acknowledgements

- HAM10000 dataset: Tschandl et al. 2018
- BCN20000 dataset: Combalia et al. 2019
- timm library: Ross Wightman
- BLIP / MedCLIP design principles: Li et al. 2022, Wang et al. 2022
