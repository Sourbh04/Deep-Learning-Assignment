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

The complete implementation — including preprocessing, augmentation, model training, validation, threshold optimization, and evaluation — is contained inside a single Jupyter Notebook.

---

## 🧠 Model Used

### DenseNet-169

DenseNet-169 was selected because of:

- Efficient feature reuse through dense connections
- Strong gradient flow in deep networks
- Strong transfer learning capability using pretrained ImageNet weights
- Excellent performance in medical image classification tasks

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

Because the dataset is highly imbalanced, two techniques were used:

### WeightedRandomSampler
Oversamples malignant samples during training.

### BCEWithLogitsLoss with `pos_weight`

```python
BCEWithLogitsLoss(pos_weight=torch.tensor([24.0]))
```

This assigns a higher penalty to malignant misclassification.

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

## 📈 Model Performance

### Final Evaluation Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9109** |
| Accuracy | 0.8225 |
| Recall | 0.8305 |
| Precision | 0.0046 |
| F1-Score | 0.0091 |

### Confusion Matrix

|  | Predicted Benign | Predicted Malignant |
|--|------------------|---------------------|
| True Benign | 49,432 | 10,668 |
| True Malignant | 10 | 49 |

### Key Observation

Despite the extreme class imbalance, the model successfully identified most malignant lesions and achieved a strong ROC-AUC score.

---

## 📂 Repository Contents

```text
Deep-Learning-Assignment/
│
├── DenseNet169_ISIC2024.ipynb
└── README.md
```

---

## 📓 Notebook Features

The notebook contains:

- Environment setup
- Dataset download using Kaggle API
- Data preprocessing pipeline
- Train/validation/test split
- Data augmentation
- DenseNet-169 implementation
- Model training loop
- Validation and testing
- Threshold optimization using Youden’s J statistic
- ROC Curve generation
- Confusion Matrix generation
- Final evaluation metrics

---

## 🏆 Outputs Generated

The notebook generates:

- Trained DenseNet-169 model
- Training & validation metrics
- ROC-AUC evaluation
- Confusion Matrix
- ROC Curve visualization
- Threshold optimization results
- Final test predictions and performance metrics

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

Run:

```bash
jupyter notebook DenseNet169_ISIC2024.ipynb
```

or upload directly to **Kaggle Notebooks**.

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
