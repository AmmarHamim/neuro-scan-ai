
<div align="center">

# 🧠 Neuro Scan AI

### AI-powered Brain CT Scan Disease Classification System

[![🚀 Live Demo](https://img.shields.io/badge/LIVE_DEMO-Streamlit-ff4b4b?style=for-the-badge)](https://neuro-scan-brain-ai.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org)

### 👉 Live App: https://neuro-scan-brain-ai.streamlit.app

</div>

---

## 📌 Overview

Neuro Scan AI is a deep learning-based web application that classifies brain CT scan images into **5 disease categories** using a fine-tuned ResNet-50 model.

It provides instant predictions with confidence scores through an interactive Streamlit interface.

---

## 🧠 Supported Classes

| Class | Condition | Severity |
|------|-----------|----------|
| 0 | Normal | 🟢 None |
| 1 | Glioma Tumor | 🔴 Critical |
| 2 | Meningioma Tumor | 🟠 High |
| 3 | Pituitary Tumor | 🟡 Moderate |
| 4 | Hemorrhage | 🔴 Critical |

---

## ⚙️ Model Details

| Component | Value |
|-----------|------|
| Backbone | ResNet-50 (ImageNet pretrained) |
| Input Size | 224 × 224 |
| Loss Function | Focal Loss (γ = 2.0) |
| Optimizer | AdamW |
| Framework | PyTorch |

---


## 🖥️ Features

- 📤 Upload CT scan images  
- ⚡ Real-time AI prediction  
- 📊 Confidence score visualization  
- 🧠 Multi-class classification (5 classes)  
- 🎯 Clean and responsive Streamlit UI  

---

## 🚀 Live Demo

👉 https://neuro-scan-brain-ai.streamlit.app

---

### 🧠 Prediction Result
![Prediction](assets/prediction.png)

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**.  
> It is not a medical device and should not be used for clinical diagnosis.

Always consult a certified medical professional.
