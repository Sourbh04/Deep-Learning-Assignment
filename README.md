# 🔬 DenseNet-169 for Skin Lesion Classification — ISIC 2024 (SLICE-3D)

> **Deep Learning Assignment — 6th Semester, May 2026**  
> **Instructor:** Dr. Baijnath Kaushik  
> **University:** Shri Mata Vaishno Devi University (SMVDU)

---

## 👥 Team Members

| Name | Roll Number |
|------|-------------|
| Sourbh Sharma | 23BCS089 |
| Manish Kumar | 23BCS050 |
| Ayush Patel | 23BCS022 |

---

## 📌 Project Overview

This project implements a **DenseNet-169** based deep learning pipeline for **binary skin lesion classification** using the **ISIC 2024 – SLICE-3D** dataset.

The objective is to classify dermoscopic skin lesion images into:

- **Benign (0)**
- **Malignant (1)**

The notebook covers the complete workflow:

- Dataset download using Kaggle API
- Data preprocessing & augmentation
- Class imbalance handling
- DenseNet-169 fine-tuning
- Model training & validation
- Threshold optimization
- Evaluation using ROC-AUC and other metrics
- Confusion matrix & ROC curve visualization

The model was trained and evaluated on a **Tesla P100 GPU** in Kaggle Notebooks.

---

## 🧠 Why DenseNet-169?

DenseNet-169 was selected because:

- Dense connections improve gradient flow
- Feature reuse reduces parameter redundancy
- Pretrained ImageNet weights help transfer learning
- Performs well for fine-grained medical image classification
- Efficient balance between performance and model size

The architecture is especially useful for dermoscopic image analysis where texture and local patterns are important.

---

## 📊 Dataset Information

### Dataset

**ISIC 2024 – Skin Cancer Detection with 3D-TBP Challenge**

Dataset Link:  
https://www.kaggle.com/competitions/isic-2024-challenge

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Total Images | ~401K |
| Image Type | JPEG dermoscopic crops |
| Task | Binary Classification |
| Input Size | 224 × 224 |
| Class Imbalance | ~24:1 (Benign : Malignant) |

### Data Split

| Split | Total Samples |
|------|----------------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

## ⚖️ Handling Class Imbalance

The dataset is highly imbalanced, so two strategies are used:

### 1. WeightedRandomSampler

Oversamples malignant samples during training so batches remain approximately balanced.

### 2. BCEWithLogitsLoss with `pos_weight`

A higher penalty is assigned to malignant misclassification:

```python
BCEWithLogitsLoss(pos_weight=torch.tensor([24.0]))
```

This improves sensitivity toward malignant lesions.

---

## 🏗️ Model Architecture

### Backbone

**DenseNet-169** pretrained on ImageNet.

### Custom Classification Head

```text
Dropout(0.4)
→ Linear(1664 → 512)
→ ReLU
→ Dropout(0.4)
→ Linear(512 → 1)
```

### Output

- Single logit output
- Sigmoid activation during inference
- Binary probability prediction

### Total Parameters

**13.3 Million+ trainable parameters**

---

## 🔄 Data Augmentation

### Training Augmentations

- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation (±15°)
- Color Jitter
- Random Affine Translation
- ImageNet Normalization

### Validation/Test Processing

- Resize to 224×224
- ImageNet normalization only

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
| Weight Decay | 1e-4 |
| Dropout | 0.4 |
| Loss Function | BCEWithLogitsLoss |
| Seed | 42 |
| Device | CUDA / Tesla P100 |

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- ROC-AUC Score
- Confusion Matrix

### Primary Metric: ROC-AUC

ROC-AUC is prioritized because the dataset is highly imbalanced and false negatives are clinically dangerous.

---

## 📊 Final Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9109 |
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

Despite extreme class imbalance, the model successfully detects most malignant cases and achieves strong ROC-AUC performance.

---

## 📁 Project Structure

```text
Deep-Learning-Assignment/
│
├── DenseNet169_ISIC2024.ipynb
├── README.md


```

---

## 🚀 Running the Project

### 1. Clone Repository

```bash
git clone https://github.com/Sourbh04/Deep-Learning-Assignment.git
cd Deep-Learning-Assignment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
kaggle competitions download -c isic-2024-challenge
mkdir -p data/isic2024
unzip -oq isic-2024-challenge.zip -d data/isic2024
```

### 4. Train Model

```bash
python train.py
```

### 5. Evaluate Model

```bash
python evaluate.py
```

---

## 💻 Kaggle Notebook Workflow

The notebook includes:

- Environment setup
- Compatible CUDA/PyTorch installation
- Kaggle dataset download
- Dynamic file generation (`config.py`, `dataset.py`, `model.py`, etc.)
- Training loop implementation
- Validation & testing pipeline
- ROC curve and confusion matrix generation

The project was executed successfully on:

```text
Tesla P100-PCIE-16GB GPU
PyTorch 2.3.0 + cu121
```

---

## 🛠️ Issues Encountered & Fixes

### CUDA Compatibility Error

Initial runs produced:

```text
torch.AcceleratorError
```

### Fix

Installed compatible CUDA 12.1 PyTorch build:

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

---

### ColorJitter OverflowError

Older torchvision environments caused hue overflow issues.

### Fix

```python
hue=0.0
```

---

## 📚 References

1. ISIC 2024 Challenge  
   https://www.kaggle.com/competitions/isic-2024-challenge

2. DenseNet Paper  
   Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017  
   https://arxiv.org/abs/1608.06993

3. PyTorch DenseNet Documentation  
   https://pytorch.org/vision/stable/models/densenet.html

4. Youden, W.J. (1950) — *Index for rating diagnostic tests*

---

## 📄 License

This repository is submitted as an academic deep learning assignment for educational purposes only.

---

## ✨ Acknowledgement

Special thanks to:

- Dr. Baijnath Kaushik
- Kaggle ISIC 2024 organizers
- PyTorch & torchvision contributors

---

<p align="center">
<b>Deep Learning Assignment — SMVDU — May 2026</b>
</p>
