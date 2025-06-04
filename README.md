# 🤖 Project: Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs

This project focuses on building a deep learning-based diagnostic pipeline for classifying scintigraphic DICOM images of horse joints. It is structured into multiple steps, starting with a binary classification of **FTU (Functionally Targeted Uptake)** vs **Non-FTU** regions.

---

## 📌 Step 1: FTU vs Non-FTU Classification

This module implements a CNN-based classification system to distinguish between **FTU (Functionally Targeted Uptake - Leg joint region)** and **Non-FTU** regions in scintigraphic images of horse joints.

This is the **first step** in the broader project:
**"Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs"**

---

## 📁 Directory Structure

FTU vs Non FTU Classification
- `Dataset.py` – DICOM dataset loader with preprocessing and augmentation
- `Model.py` – FTUCNN architecture definition
- `Train.py` – Script for training the model
- `Test.py` – Script for running inference on test data
- `Evaluation.py` – Generates classification report and confusion matrix
- `model_FTU_nonftu.pth` – Saved trained model 

---

## 🧠 Model Details

| Component       | Description                                         |
|----------------|-----------------------------------------------------|
| Architecture    | Custom CNN (`FTUCNN`) with BatchNorm               |
| Input Shape     | 224 × 224 RGB                                      |
| Optimizer       | AdamW                                              |
| Learning Rate   | 0.0001                                             |
| Epochs          | 10                                                 |
| Loss Function   | CrossEntropyLoss with class weights                |
| Regularization  | BatchNorm + Weight Decay (0.0001)                  |
| Augmentations   | FTU: Contrast jitter + rotation<br>Non-FTU: None   |

---

## 🧪 Results

### ✅ Classification Report
- **Validation Accuracy:** `99.05%`
- **FTU Recall:** `99%`
- **FTU Precision:** `95%`
- **Non-FTU Precision:** `100%`

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Non-FTU   | 1.00      | 0.99   | 0.99     | 2152    |
| FTU       | 0.95      | 0.99   | 0.97     | 376     |
| **Accuracy** |         |        | **0.99** | **2528** |

#### 📉 Confusion Matrix

![Confusion Matrix](Images/FTU%20classification(cm).png)

#### 📋 Classification Report Screenshot

![Classification Report](Images/FTU%20classification%20report.png)

---




