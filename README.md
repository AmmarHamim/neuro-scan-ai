<div align="center">

# 🧠 NeuroScan AI

### Brain CT Disease Detection powered by Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neuroscan-ai.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Upload a brain CT scan — get instant disease classification with confidence scores*

![NeuroScan AI Demo](https://img.shields.io/badge/Status-Live-brightgreen)

</div>

---

## 🔍 What it does

NeuroScan AI analyses axial brain CT scan images and detects 5 conditions in under 2 seconds:

| # | Condition | Severity |
|---|-----------|----------|
| 0 | ✅ Normal | None |
| 1 | 🔴 Glioma Tumor | CRITICAL |
| 2 | 🟠 Meningioma Tumor | HIGH |
| 3 | 🟡 Pituitary Tumor | MODERATE |
| 4 | 🔴 Hemorrhage | CRITICAL |

For each scan it outputs:
- **Disease name** with confidence %
- **Anomaly score** (0–100%)
- **Activation heatmap** showing which region triggered the prediction
- **Probability bar chart** across all 5 classes
- **Severity badge** (Critical / High / Moderate / None)

---

## 🚀 Live Demo

👉 **[neuroscan-ai.streamlit.app](https://neuroscan-ai.streamlit.app)**

---

## 📁 Project Structure

```
neuro-scan-ai/
├── app.py                    # Streamlit web application
├── test_brain_ct.py          # Terminal test script (single + batch)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Dark theme config
├── models/
│   └── README.md             # Instructions to add model weights
├── test_samples/
│   └── README.md             # Instructions to add test images
├── notebooks/
│   ├── Brain_CT_Training_Colab.ipynb    # Google Colab training notebook
│   └── Brain_CT_Training_Kaggle.ipynb   # Kaggle training notebook (15k+ images)
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1 — Clone
```bash
git clone https://github.com/YOUR_USERNAME/neuro-scan-ai.git
cd neuro-scan-ai
```

### 2 — Install
```bash
pip install -r requirements.txt
```

### 3 — Add model
Download `brain_classifier_final.pth` from the [Releases](https://github.com/YOUR_USERNAME/neuro-scan-ai/releases) page and place it in the `models/` folder:
```
models/
└── brain_classifier_final.pth
```

### 4 — Run
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## 🧪 Test Script

```bash
# Single image
python test_brain_ct.py \
  --image scan.png \
  --model ./models/brain_classifier_final.pth

# Save result as image
python test_brain_ct.py \
  --image scan.png \
  --model ./models/brain_classifier_final.pth \
  --save result.png

# Batch test a folder
python test_brain_ct.py \
  --folder ./test_samples/ \
  --model ./models/brain_classifier_final.pth

# Batch + save all results
python test_brain_ct.py \
  --folder ./test_samples/ \
  --model ./models/brain_classifier_final.pth \
  --save ./results/
```

---

## 🏗️ Model Architecture

```
Input (224×224 CT slice)
        │
   ResNet-50 Backbone
   (ImageNet V2 pretrained)
        │
   ┌────┴────┐
   │  frozen │  conv1 · bn1 · layer1 · layer2
   └────┬────┘
        │
   ┌────┴────┐
   │fine-tune│  layer3 · layer4
   └────┬────┘
        │
   Classification Head
   Linear(2048→512) → BN → GELU → Dropout(0.4)
   Linear(512→256)  → BN → GELU → Dropout(0.2)
   Linear(256→5)
        │
   Softmax → [Normal, Glioma, Meningioma, Pituitary, Hemorrhage]
```

| Item | Detail |
|------|--------|
| Backbone | ResNet-50 (ImageNet V2) |
| Loss | Focal Loss (γ=2.0) |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingWarmRestarts |
| Input | 224×224 grayscale→RGB |
| Training images | ~15,000+ |
| Epochs | Up to 50 (early stopping) |

---

## 📊 Training Datasets

| Dataset | Images | Classes |
|---------|--------|---------|
| Brain Tumor MRI — sartajbhuvaji | 3,264 | Glioma · Meningioma · Pituitary · Normal |
| Brain Tumor MRI — masoudnickparvar | 7,023 | Glioma · Meningioma · Pituitary · Normal |
| Head CT Hemorrhage — felipekitamura | 200 | Normal · Hemorrhage |
| Brain CT Hemorrhage — abdulkader90 | ~5,000 | Normal · Hemorrhage |
| **Total** | **~15,000+** | **5 classes** |

All datasets are publicly available on [Kaggle](https://kaggle.com).

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo → set **Main file** to `app.py`
4. Add your model URL in **Secrets**:

```toml
# .streamlit/secrets.toml  (never commit this file)
MODEL_URL = "https://your-model-direct-download-link.pth"
```

> 💡 **Free model hosting options:**
> - [GitHub Releases](https://github.com/YOUR_USERNAME/neuro-scan-ai/releases) (up to 2 GB)
> - [Hugging Face Hub](https://huggingface.co) (unlimited, free)
> - Google Drive (get direct download link)

---

## 💻 Requirements

```
Python     3.9 / 3.10
PyTorch    >= 2.0.0
CUDA       optional (runs fine on CPU)
RAM        4 GB minimum
```

---

## ⚠️ Disclaimer

> NeuroScan AI is intended for **research and educational purposes only**.
> It is **not a certified medical device** and must **not be used for clinical diagnosis**.
> Always consult a qualified neurologist or radiologist for medical decisions.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">

Built with ❤️ using PyTorch · Streamlit · ResNet-50

**⭐ Star this repo if you found it useful!**

</div>
