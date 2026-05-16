import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neuro Scan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c10;
    color: #e2e8f0;
}
.stApp { background: #080c10; }
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2a38;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

.brain-header {
    background: linear-gradient(135deg, #0d1117 0%, #0f1f2e 50%, #0d1117 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
}
.brain-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    color: #f0f9ff;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}
.brain-header p { color: #64748b; font-size: 0.92rem; font-weight: 300; }
.brain-header .accent { color: #38bdf8; }

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 22px;
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin: 20px 0;
}
.pill-normal  { background:#071a0f; color:#4ade80; border:1.5px solid #4ade80; }
.pill-anomaly { background:#1a070a; color:#f87171; border:1.5px solid #f87171; }

.metric-box {
    background: #0d1117;
    border: 1px solid #1e2a38;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.metric-box .m-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
.metric-box .m-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid #111827;
}
.prob-row:last-child { border-bottom: none; }
.prob-label { font-size: 0.82rem; color: #94a3b8; width: 180px; flex-shrink: 0; }
.prob-bar-bg {
    flex: 1; height: 8px;
    background: #1e2a38; border-radius: 4px; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 4px; }
.prob-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem; color: #64748b;
    width: 52px; text-align: right; flex-shrink: 0;
}

.sec-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem; color: #334155;
    text-transform: uppercase; letter-spacing: 3px; margin: 24px 0 12px;
}

.upload-zone {
    border: 2px dashed #1e3a5f; border-radius: 16px;
    padding: 60px 20px; text-align: center; background: #0a141e; margin-top: 20px;
}
.upload-zone .uz-icon { font-size: 3rem; margin-bottom: 16px; }
.upload-zone .uz-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem; color: #38bdf8; margin-bottom: 8px;
}
.upload-zone .uz-sub { color: #334155; font-size: 0.85rem; }

.model-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 8px;
}
.dot-green { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
.dot-red   { background: #f87171; box-shadow: 0 0 6px #f87171; }

.sev-badge {
    display: inline-block; padding: 3px 12px; border-radius: 4px;
    font-family: 'Space Mono', monospace; font-size: 0.75rem;
    font-weight: 700; letter-spacing: 1px;
}
.sev-critical { background:#2d0a0a; color:#fca5a5; border:1px solid #ef4444; }
.sev-high     { background:#2d1a0a; color:#fbbf24; border:1px solid #f59e0b; }
.sev-moderate { background:#1a2d0a; color:#86efac; border:1px solid #22c55e; }
.sev-none     { background:#0a1a2d; color:#7dd3fc; border:1px solid #38bdf8; }

hr { border-color: #1e2a38 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model definition ──────────────────────────────────────────────────────────
class BrainCTClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        backbone.eval()
        with torch.no_grad():
            feat_dim = backbone(torch.zeros(1, 3, 224, 224)).shape[1]
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ── Config ────────────────────────────────────────────────────────────────────
DEVICE   = torch.device('cpu')
IMG_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

DEFAULT_CLASSES = [
    'Normal', 'Glioma Tumor', 'Meningioma Tumor',
    'Pituitary Tumor', 'Hemorrhage'
]
SEVERITY_MAP = {
    'Normal':           ('None',     'sev-none'),
    'Glioma Tumor':     ('CRITICAL', 'sev-critical'),
    'Meningioma Tumor': ('HIGH',     'sev-high'),
    'Pituitary Tumor':  ('MODERATE', 'sev-moderate'),
    'Hemorrhage':       ('CRITICAL', 'sev-critical'),
}
PROB_COLORS = {
    'Normal':           '#4ade80',
    'Glioma Tumor':     '#f87171',
    'Meningioma Tumor': '#fb923c',
    'Pituitary Tumor':  '#facc15',
    'Hemorrhage':       '#f43f5e',
}


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    try:
        local_path = Path(model_path)

        # ── Download if not found locally ─────────────────────────
        if not local_path.exists():
            # Try Streamlit secrets first
            model_url = None
            try:
                model_url = st.secrets["MODEL_URL"]
            except Exception:
                pass

            # Fallback: hardcode your HF URL here directly
            if not model_url:
                model_url = "https://huggingface.co/Hamim88/neuro-scan-ai/resolve/main/brain_classifier_final.pth"

            os.makedirs(local_path.parent, exist_ok=True)

            with st.spinner("⬇️ Downloading model weights (~100MB)..."):
                import requests
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

        ck          = torch.load(local_path, map_location=DEVICE)
        classes     = ck.get('classes', DEFAULT_CLASSES)
        num_classes = ck.get('num_classes', len(classes))
        model       = BrainCTClassifier(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(ck['model_state'])
        model.eval()
        meta = {
            'classes':      classes,
            'num_classes':  num_classes,
            'val_acc':      ck.get('val_acc', None),
            'test_acc':     ck.get('test_acc', None),
            'best_epoch':   ck.get('best_epoch', ck.get('epoch', '?')),
            'dataset_size': ck.get('dataset_size', '?'),
        }
        return model, meta, None

    except Exception as e:
        return None, None, str(e)


def make_heatmap(img_pil, model):
    """Layer4 activation map overlay."""
    model.eval()
    tensor      = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    activations = []

    def hook(module, inp, out):
        activations.append(out.detach())

    h = model.backbone.layer4.register_forward_hook(hook)
    with torch.no_grad():
        model(tensor)
    h.remove()

    feat_map  = activations[0].squeeze(0).mean(0).numpy()
    feat_map  = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
    import cv2
    heat_up   = cv2.resize(feat_map, (IMG_SIZE, IMG_SIZE))
    orig      = np.array(img_pil.convert('L').resize((IMG_SIZE, IMG_SIZE)))
    orig_rgb  = np.stack([orig] * 3, axis=-1).astype(np.float32) / 255.0
    heat_rgb  = cm.get_cmap('jet')(heat_up)[:, :, :3]
    overlay   = (0.55 * orig_rgb + 0.45 * heat_rgb).clip(0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def predict(img_pil, model, classes):
    tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().numpy()
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    return {
        'pred_idx':      pred_idx,
        'pred_class':    classes[pred_idx],
        'confidence':    confidence,
        'is_anomaly':    pred_idx != 0,
        'probs':         {classes[i]: float(probs[i]) * 100 for i in range(len(classes))},
        'anomaly_score': (1.0 - float(probs[0])) * 100,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Neuro Scan AI")
    st.markdown("---")

    st.markdown("#### 📂 Model Path")
    model_path = st.text_input(
        "model_path",
        value="./models/brain_classifier_final.pth",
        label_visibility="collapsed",
    )

    model, meta, err = load_model(model_path)

    if model:
        classes = meta['classes']
        st.markdown(
            '<div style="margin-top:8px">'
            '<span class="model-dot dot-green"></span>'
            '<span style="color:#4ade80;font-size:0.85rem;font-family:monospace">Model loaded ✓</span>'
            '</div>', unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("#### 📊 Model Info")
        info = [
            ("Classes",   str(meta['num_classes'])),
            ("Epoch",     str(meta['best_epoch'])),
            ("Val Acc",   f"{meta['val_acc']*100:.1f}%" if meta['val_acc'] else "—"),
            ("Test Acc",  f"{meta['test_acc']*100:.1f}%" if meta['test_acc'] else "—"),
            ("Dataset",   f"{meta['dataset_size']:,}" if isinstance(meta['dataset_size'], int) else str(meta['dataset_size'])),
        ]
        for k, v in info:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:5px 0;border-bottom:1px solid #111827">'
                f'<span style="color:#475569;font-size:0.8rem">{k}</span>'
                f'<span style="color:#94a3b8;font-size:0.8rem;font-family:monospace">{v}</span>'
                f'</div>', unsafe_allow_html=True
            )
        st.markdown("---")
        st.markdown("#### 🏷️ Classes")
        for i, cls in enumerate(classes):
            st.markdown(
                f'<div style="font-size:0.8rem;padding:3px 0;color:#64748b">'
                f'<code style="color:#38bdf8">[{i}]</code> {cls}</div>',
                unsafe_allow_html=True
            )
        st.markdown("---")
        st.markdown("#### ⚙️ Options")
        show_heatmap = st.checkbox("Show activation heatmap", value=True)
        threshold    = st.slider("Anomaly threshold", 0.1, 0.9, 0.5, 0.05)
    else:
        classes      = DEFAULT_CLASSES
        show_heatmap = True
        threshold    = 0.5
        st.markdown(
            '<div style="margin-top:8px">'
            '<span class="model-dot dot-red"></span>'
            '<span style="color:#f87171;font-size:0.85rem;font-family:monospace">Not loaded</span>'
            '</div>', unsafe_allow_html=True
        )
        if err:
            st.error(f"{err}")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#1e3a5f;text-align:center;font-family:monospace">'
        'Neuro Scan AI v1.0 · ResNet-50</div>',
        unsafe_allow_html=True
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brain-header">
    <h1>🧠 Neuro Scan <span class="accent">AI</span></h1>
    <p>Upload a brain CT scan — detect anomalies &amp; classify disease type with confidence scores.</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload CT scan",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    label_visibility="collapsed"
)

if not uploaded:
    st.markdown("""
    <div class="upload-zone">
        <div class="uz-icon">🩻</div>
        <div class="uz-title">Drop a brain CT scan image here</div>
        <div class="uz-sub">PNG · JPG · BMP · TIFF &nbsp;|&nbsp; Axial brain CT slices work best</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Predict ───────────────────────────────────────────────────────────────────
img_pil = Image.open(uploaded).convert("RGB")

if not model:
    st.error("❌ No model loaded. Fix the path in the sidebar.")
    st.stop()

with st.spinner("Analysing scan..."):
    result  = predict(img_pil, model, classes)
    heatmap = make_heatmap(img_pil, model) if show_heatmap else None

anomaly_score = result['anomaly_score'] / 100
is_anomaly    = anomaly_score > threshold
pred_class    = result['pred_class']
confidence    = result['confidence']

# ── Status pill ───────────────────────────────────────────────────────────────
pill_cls  = "pill-anomaly" if is_anomaly else "pill-normal"
pill_text = "⚠  ANOMALY DETECTED" if is_anomaly else "✓  NORMAL"
st.markdown(
    f'<div class="status-pill {pill_cls}">{pill_text}</div>',
    unsafe_allow_html=True
)

# ── Metric cards ──────────────────────────────────────────────────────────────
sev_text, sev_cls = SEVERITY_MAP.get(pred_class, ('—', 'sev-none'))
score_color = "#f87171" if is_anomaly else "#4ade80"
conf_color  = "#f87171" if confidence < 60 else "#facc15" if confidence < 80 else "#4ade80"

m1, m2, m3, m4 = st.columns(4)
for col, label, value, color in [
    (m1, "DIAGNOSIS",     pred_class,                    score_color),
    (m2, "CONFIDENCE",    f"{confidence:.1f}%",          conf_color),
    (m3, "ANOMALY SCORE", f"{result['anomaly_score']:.1f}%", score_color),
    (m4, "THRESHOLD",     f"{threshold*100:.0f}%",       "#64748b"),
]:
    col.markdown(
        f'<div class="metric-box">'
        f'<div class="m-label">{label}</div>'
        f'<div class="m-value" style="color:{color};'
        f'font-size:{"0.95rem" if label=="DIAGNOSIS" else "1.6rem"}">{value}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

if is_anomaly:
    st.markdown(
        f'<br>Severity &nbsp;'
        f'<span class="sev-badge {sev_cls}">{sev_text}</span>'
        f'&nbsp;&nbsp;<span style="color:#334155;font-size:0.8rem">'
        f'— Please consult a neurologist / radiologist</span>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ── Image panels + probability bars ──────────────────────────────────────────
n_img_cols = 2 if (show_heatmap and heatmap) else 1
img_cols   = st.columns(n_img_cols + 1)   # last col = prob bars

# Original
with img_cols[0]:
    st.markdown('<div class="sec-title">ORIGINAL SCAN</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.imshow(np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))[:, :, 0], cmap='bone')
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# Heatmap
if show_heatmap and heatmap:
    with img_cols[1]:
        st.markdown('<div class="sec-title">ACTIVATION HEATMAP</div>', unsafe_allow_html=True)
        st.image(heatmap, use_container_width=True)
        st.markdown(
            '<div style="font-size:0.72rem;color:#334155;text-align:center;margin-top:6px">'
            '🔴 Red = high activation region</div>',
            unsafe_allow_html=True
        )

# Probability bars
with img_cols[-1]:
    st.markdown('<div class="sec-title">CLASS PROBABILITIES</div>', unsafe_allow_html=True)
    for cls_name, pct in sorted(result['probs'].items(), key=lambda x: x[1], reverse=True):
        bar_color  = PROB_COLORS.get(cls_name, '#64748b')
        is_pred    = cls_name == pred_class
        lbl_style  = f"color:{'#e2e8f0' if is_pred else '#64748b'};font-weight:{'600' if is_pred else '400'}"
        st.markdown(
            f'<div class="prob-row">'
            f'  <div class="prob-label" style="{lbl_style}">{cls_name}</div>'
            f'  <div class="prob-bar-bg">'
            f'    <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{bar_color}"></div>'
            f'  </div>'
            f'  <div class="prob-pct">{pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Gauge ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-title">ANOMALY SCORE GAUGE</div>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 0.9))
fig.patch.set_facecolor('#080c10')
ax.set_facecolor('#080c10')
ax.barh(0, 1.0,          color='#0d1117',                          height=0.5)
ax.barh(0, anomaly_score, color='#f87171' if is_anomaly else '#4ade80', height=0.5, alpha=0.9)
ax.axvline(threshold, color='#fbbf24', linewidth=2, linestyle='--')
ax.text(threshold, 0.38, f'θ={threshold*100:.0f}%',
        color='#fbbf24', ha='center', fontsize=9, fontfamily='monospace')
ax.text(anomaly_score, -0.38, f'{anomaly_score*100:.1f}%',
        color='#f87171' if is_anomaly else '#4ade80',
        ha='center', fontsize=10, fontweight='bold', fontfamily='monospace')
ax.set_xlim(0, 1); ax.set_ylim(-0.55, 0.55); ax.axis('off')
plt.tight_layout(pad=0)
st.pyplot(fig, use_container_width=True)
plt.close()
