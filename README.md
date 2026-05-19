# 🔬 DenseNet-169 for Skin Lesion Classification — ISIC 2024 (SLICE-3D)

> Deep Learning Assignment — 6th Semester, May 2026  
> Shri Mata Vaishno Devi University (SMVDU)

---

## 👥 Team Members

| Name | Roll Number |
|------|-------------|
| Sourbh Sharma | 23BCS089 |
| Manish Kumar | 23BCS050 |
| Ayush Patel | 23BCS022 |

---

## 📌 Project Overview

This project implements a **DenseNet-169** based deep learning model for **binary skin lesion classification** using the **ISIC 2024 – SLICE-3D** dataset.

The objective is to classify dermoscopic skin lesion images into:

- **Benign**
- **Malignant**

The entire workflow — including preprocessing, augmentation, model training, validation, threshold tuning, and evaluation — is implemented inside a single Jupyter Notebook.

---

## 🧠 Model Used

### DenseNet-169

DenseNet-169 was selected because of:

- Efficient feature reuse through dense connections
- Strong gradient flow in deep networks
- Good transfer learning performance using pretrained ImageNet weights
- Strong performance in medical image classification tasks

### Classification Head

```text
Dropout(0.4)
→ Linear(1664 → 512)
→ ReLU
→ Dropout(0.4)
→ Linear(512 → 1)
```

---

## 📊 Dataset Information

### Dataset

**ISIC 2024 – Skin Cancer Detection with 3D-TBP Challenge**

https://www.kaggle.com/competitions/isic-2024-challenge

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Total Images | ~401K |
| Task | Binary Classification |
| Input Size | 224 × 224 |
| Classes | Benign / Malignant |
| Class Imbalance | ~24:1 |

---

## ⚖️ Handling Class Imbalance

Since the dataset is highly imbalanced, two strategies were used:

### WeightedRandomSampler
Oversamples malignant samples during training.

### BCEWithLogitsLoss with `pos_weight`

```python
BCEWithLogitsLoss(pos_weight=torch.tensor([24.0]))
```

This increases the penalty for malignant misclassification.

---

## 🔄 Data Augmentation

### Training Augmentations

- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation
- Color Jitter
- Random Affine Transform
- ImageNet Normalization

### Validation/Test Processing

- Resize to 224×224
- ImageNet Normalization

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Model | DenseNet-169 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 30 |
| Scheduler | CosineAnnealingLR |
| Loss Function | BCEWithLogitsLoss |
| Device | Tesla P100 GPU |

---

## 📈 Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

### Final ROC-AUC

```text
AUC = 0.9109
```

Despite severe class imbalance, the model achieved strong discriminative performance for malignant lesion detection.

---

## 📓 Notebook Contents

The notebook includes:

- Environment setup
- Dataset download using Kaggle API
- Data preprocessing
- Train/validation/test split
- Data augmentation
- DenseNet-169 model definition
- Training loop implementation
- Validation and testing
- Threshold optimization using Youden’s J statistic
- ROC Curve generation
- Confusion Matrix generation
- Final metric evaluation

---

## 🚀 Running the Notebook

### 1. Clone Repository

```bash
git clone https://github.com/Sourbh04/Deep-Learning-Assignment.git
cd Deep-Learning-Assignment
```

### 2. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install pandas scikit-learn matplotlib seaborn pillow tqdm
```

### 3. Download Dataset

```bash
kaggle competitions download -c isic-2024-challenge
```

### 4. Open Notebook

Run the notebook:

```bash
jupyter notebook DenseNet169_ISIC2024.ipynb
```

or upload it directly to **Kaggle Notebooks**.

---

## 💻 Development Environment

| Component | Details |
|----------|---------|
| Platform | Kaggle Notebooks |
| GPU | Tesla P100-PCIE-16GB |
| Framework | PyTorch |
| CUDA Version | cu121 |

---

## 🛠️ Issues Encountered

### CUDA Compatibility Error

Resolved by installing compatible CUDA 12.1 PyTorch packages.

### ColorJitter OverflowError

Resolved by setting:

```python
hue=0.0
```

---

## 📚 References

1. ISIC 2024 Challenge  
   https://www.kaggle.com/competitions/isic-2024-challenge

2. Huang et al. — *Densely Connected Convolutional Networks*  
   https://arxiv.org/abs/1608.06993

3. PyTorch DenseNet Documentation  
   https://pytorch.org/vision/stable/models/densenet.html

---

## 📄 License

This repository is submitted as an academic deep learning assignment for educational purposes only.

---

<p align="center">
<b>Deep Learning Assignment — SMVDU — May 2026</b>
</p>
