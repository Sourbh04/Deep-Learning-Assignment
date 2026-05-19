# 🔬 DenseNet-169 for Skin Lesion Classification — ISIC 2024 (SLICE-3D)

> **Deep Learning Assignment — 6th Semester, May 2026**
> Submitted to: **Dr. Baijnath Kaushik**

---

## 👥 Team Members

| Name | Roll Number |
|------|-------------|
| Sourbh Sharma | 23BCS089 |
| Manish Kumar | 23BCS050 |
| Ayush Patel | 23BCS022 |

---

## 📌 Project Overview

This project implements a **DenseNet-169** convolutional neural network fine-tuned for **binary skin lesion classification** on the [ISIC 2024 – SLICE-3D](https://www.kaggle.com/competitions/isic-2024-challenge) dataset. The task is to classify skin lesion images as either **benign** or **malignant**, a clinically critical problem in early melanoma detection.

The dataset contains **~401,000** binary-labelled dermoscopic crop images, with a significant class imbalance (approximately 24:1 benign-to-malignant ratio). The model is trained end-to-end with strategies to address this imbalance and achieves an **AUC-ROC of 0.9109** on the held-out test set.

---

## 📁 Project Structure

```
densenet169_project/
├── config.py          # All hyperparameters, paths, and settings
├── dataset.py         # Dataset class, augmentations, and DataLoader factory
├── model.py           # DenseNet-169 architecture with custom classification head
├── train.py           # Training loop with cosine LR scheduling
├── evaluate.py        # Evaluation, metrics, and plot generation
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── outputs/
    ├── best_densenet169.pth     # Best model checkpoint (by val AUC)
    ├── history.json             # Training loss/accuracy history
    ├── test_results.json        # Final test metrics
    ├── confusion_matrix.png     # Confusion matrix plot
    └── roc_curve.png            # ROC curve plot
```

---

## 🏗️ Model Architecture

**Backbone:** DenseNet-169 pretrained on ImageNet-1K (torchvision)

- 4 Dense Blocks: [6, 12, 32, 32] layers, growth rate k = 32
- Transition layers: BatchNorm → Conv 1×1 → AvgPool 2×2
- Final feature map: **1664 channels** at 7×7 (for 224×224 input)

**Custom Classification Head:**
```
Dropout(0.4) → Linear(1664 → 512) → ReLU → Dropout(0.4) → Linear(512 → 1)
```

- Output: Raw logit → sigmoid for probability
- Total parameters: **13,337,473** (all trainable)

---

## 📊 Dataset

| Split | Total | Malignant | Benign |
|-------|-------|-----------|--------|
| Train | 280,741 | 275 | 280,466 |
| Val | 60,159 | 59 | 60,100 |
| Test | 60,159 | 59 | 60,100 |

- **Source:** [Kaggle – ISIC 2024 Challenge](https://www.kaggle.com/competitions/isic-2024-challenge)
- **Images:** JPEG dermoscopic crops (224×224 resized)
- **Labels:** Binary — `0` = Benign, `1` = Malignant
- **Class imbalance ratio:** ~24:1 (benign:malignant)

### Handling Class Imbalance

Two complementary strategies are used:

1. **`WeightedRandomSampler`** — oversamples the minority (malignant) class each epoch so batches are approximately balanced
2. **`BCEWithLogitsLoss(pos_weight=24.0)`** — applies a 24× higher penalty for false negatives during loss computation

---

## 🔄 Data Augmentation

**Training:**
- Random horizontal & vertical flips (p=0.5 each)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine translation (±5%)
- ImageNet normalization

**Validation / Test:**
- Resize to 224×224
- ImageNet normalization only

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Image size | 224×224 |
| Batch size | 32 |
| Epochs | 30 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR scheduler | CosineAnnealingLR |
| Loss function | BCEWithLogitsLoss |
| pos_weight | 24.0 |
| Dropout | 0.4 |
| Seed | 42 |
| Device | CUDA (Tesla P100-PCIE-16GB on Kaggle) |

Best model checkpoint is saved based on **validation AUC-ROC**.

---

## 📈 Results

### Test Set Metrics

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.9109** |
| Accuracy | 0.8225 |
| Precision | 0.0046 |
| Recall (Sensitivity) | 0.8305 |
| F1-Score | 0.0091 |
| Decision Threshold (Youden J) | ~7.75e-05 |

> ⚠️ **Note on Precision/F1:** The extremely low precision and F1 reflect the severe class imbalance (only 59 malignant cases out of 60,159 test samples). The model correctly flags **83% of malignant cases (recall)** and achieves a strong **AUC of 0.91**, which is the primary metric for this task in clinical screening.

### Confusion Matrix

|  | Predicted Benign | Predicted Malignant |
|--|-----------------|-------------------|
| **True Benign** | 49,432 (TN) | 10,668 (FP) |
| **True Malignant** | 10 (FN) | 49 (TP) |

### ROC Curve

The model achieves **AUC = 0.9109**, indicating strong discriminative ability despite the extreme class imbalance.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Sourbh04/Deep-Learning-Assignment.git
cd Deep-Learning-Assignment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install pandas scikit-learn pillow tqdm matplotlib seaborn
```

### 3. Download the Dataset

You need a Kaggle API token (`~/.kaggle/kaggle.json`).

```bash
kaggle competitions download -c isic-2024-challenge
mkdir -p data/isic2024
unzip -oq isic-2024-challenge.zip -d data/isic2024
```

Expected structure:
```
data/isic2024/
├── train-metadata.csv
└── train-image/
    └── image/
        ├── ISIC_0000000.jpg
        ├── ISIC_0000001.jpg
        └── ...
```

### 4. Train the Model

```bash
python train.py
```

This will:
- Perform a stratified 70/15/15 train/val/test split
- Train DenseNet-169 for 30 epochs
- Save the best checkpoint to `outputs/best_densenet169.pth`
- Save training history to `outputs/history.json`

### 5. Evaluate the Model

```bash
python evaluate.py
```

This will:
- Load the best checkpoint
- Find the optimal decision threshold using Youden's J statistic on the validation set
- Run inference on the test set
- Print metrics and save plots to `outputs/`

---

## 🔧 Running on Kaggle

This project was developed and run on **Kaggle Notebooks** (Tesla P100-PCIE-16GB GPU).

```python
# Copy project files
!cp -r /kaggle/input/datasets/sourbhsharma23bcs089/densenet169-project/densenet169_project /kaggle/working/project
%cd /kaggle/working/project

# Download dataset
!kaggle competitions download -c isic-2024-challenge
!mkdir -p data/isic2024
!unzip -oq isic-2024-challenge.zip -d data/isic2024

# Fix torchvision ColorJitter hue bug (if needed on older environments)
!sed -i 's/hue=[^,)]*/hue=0.0/g' /kaggle/working/project/dataset.py

# Train
!python train.py

# Evaluate
!python evaluate.py
```

> **Note:** The initial run encountered a `torch.AcceleratorError` due to CUDA version mismatch (cu128 vs cu121). This was resolved by reinstalling `torch==2.3.0+cu121`. A separate `OverflowError` in `ColorJitter` (hue parameter out of uint8 bounds) was fixed by setting `hue=0.0`.

---

## 🧠 Key Design Decisions

### Why DenseNet-169?
- Dense connections encourage feature reuse and gradient flow — ideal for fine-grained texture classification in dermoscopy
- Pre-trained ImageNet weights provide strong low-level feature extractors (edges, textures) transferable to skin lesion images
- DenseNet-169 offers a good balance between depth (1664-dim feature vector) and parameter efficiency vs. larger variants

### Why AUC-ROC as the Primary Metric?
In clinical screening, **missing a malignant lesion (false negative) is far more costly** than a false alarm. AUC-ROC evaluates the model across all thresholds and is robust to class imbalance, making it the standard metric for this competition and similar medical imaging tasks.

### Threshold Tuning
Rather than using a fixed 0.5 threshold, the optimal threshold is found on the **validation set** using **Youden's J statistic** (maximizes sensitivity + specificity − 1), which is more principled for imbalanced datasets.

---

## 📚 References

- [ISIC 2024 – Skin Cancer Detection with 3D-TBP Challenge](https://www.kaggle.com/competitions/isic-2024-challenge)
- Huang, G. et al. (2017). *Densely Connected Convolutional Networks*. CVPR. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
- PyTorch Documentation: [torchvision.models.densenet169](https://pytorch.org/vision/stable/models/densenet.html)
- Youden, W.J. (1950). *Index for rating diagnostic tests*. Cancer.

---

## 📄 License

This project is submitted as an academic assignment. Code is for educational purposes only.

---

*Assignment submitted for: Deep Learning (6th Semester) — May 2026 | Instructor: Dr. Baijnath Kaushik*
