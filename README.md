# DenseNet-169 for Skin Lesion Classification — ISIC 2024

Binary malignant vs. benign classification on the **ISIC 2024 (SLICE-3D)** dataset using **DenseNet-169** implemented in PyTorch.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |
| AUC-ROC | — |

> Fill in after running `evaluate.py`. Results are reported on the held-out test set (15% stratified split).

Model weights: [Download best_densenet169.pth](LINK_TO_RELEASE)

---

## Project Structure

```
densenet169-isic2024/
├── config.py        # All hyperparameters and paths
├── dataset.py       # ISICDataset, transforms, stratified split
├── model.py         # DenseNet-169 with custom classification head
├── train.py         # Training loop (BCE + cosine LR)
├── evaluate.py      # Test metrics, confusion matrix, ROC curve
├── requirements.txt
├── README.md
└── outputs/
    ├── best_densenet169.pth   # Best model checkpoint
    ├── history.json           # Training history
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── loss_acc_curves.png
```

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/densenet169-isic2024.git
cd densenet169-isic2024
pip install -r requirements.txt
```

### 2. Download Dataset (Kaggle API)

```bash
# Install kaggle CLI and place kaggle.json in ~/.kaggle/
kaggle competitions download -c isic-2024-challenge
unzip isic-2024-challenge.zip -d ./data/isic2024
```

### 3. Train

```bash
python train.py
```

### 4. Evaluate

```bash
python evaluate.py
```

---

## Architecture

DenseNet-169 uses **dense connectivity**: each layer `l` receives feature maps from **all preceding layers** within a dense block as input. With growth rate k=32 and 4 dense blocks (6, 12, 32, 32 layers), the model has ~14M parameters.

**Custom head:**
```
features (DenseNet backbone)
  → AdaptiveAvgPool2d(1)
  → Flatten
  → Dropout(0.4)
  → Linear(1664 → 512) → ReLU
  → Dropout(0.4)
  → Linear(512 → 1)   ← raw logit
```

---

## Key Decisions

| Design Choice | Rationale |
|---|---|
| Pretrained on ImageNet | Transfer learning critical for limited medical data |
| Weighted BCE loss (pos_weight≈24) | ISIC 2024 is ~4% malignant — severe class imbalance |
| WeightedRandomSampler | Oversamples malignant cases each epoch |
| Cosine LR schedule | Smooth convergence without manual LR tuning |
| Youden's J threshold | Optimises sensitivity + specificity jointly |
| Stratified 70/15/15 split | Preserves class ratio across all three sets |

---

## Dataset

**ISIC 2024 Challenge (SLICE-3D)**
- 401,059 skin lesion crops
- Binary labels: malignant (1) vs. benign (0)
- Source: https://challenge2024.isic-archive.com/

---

## Citation

```
@misc{isic2024,
  title  = {ISIC 2024 Challenge: SLICE-3D Skin Lesion Classification},
  year   = {2024},
  url    = {https://challenge2024.isic-archive.com/}
}

@inproceedings{huang2017densely,
  title     = {Densely Connected Convolutional Networks},
  author    = {Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle = {CVPR},
  year      = {2017}
}
```
