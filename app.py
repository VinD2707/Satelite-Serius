# ============================================================
# Fire Risk Mapping App
# Sentinel-2 Segmentation + Fire Risk Index (FRI)
# ============================================================

import os
import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
CKPT_PATH = "unet13_best.pth"
META_PATH = "unet13_meta.npz"
LR_COEF_PATH = "lr_risk_coeffs.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL DEFINITION (MUST MATCH TRAINING)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_channels=13, out_channels=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

# ============================================================
# UTILS
# ============================================================
def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))

def to_rgb(img):
    r, g, b = img[:,:,3], img[:,:,2], img[:,:,1]
    rgb = np.stack([r,g,b], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return rgb

# ============================================================
# LOAD ARTIFACTS (SAFE)
# ============================================================
@st.cache_resource
def load_artifacts():
    # model
    model = UNetSmall(in_channels=13, out_channels=1)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    # normalization
    meta = np.load(META_PATH)
    mean13 = meta["mean_13"]
    std13 = meta["std_13"]
    thr_pixel = float(ckpt.get("thr", 0.55))

    # LR coefficients
    lr_npz = np.load(LR_COEF_PATH)
    intercept = lr_npz["intercept"].reshape(-1)
    coef = lr_npz["coef"].reshape(-1)

    return model, mean13, std13, thr_pixel, intercept, coef

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Fire Risk Mapping (FRI)", layout="wide")
st.title("🔥 Fire Risk Mapping berbasis Segmentasi Sentinel-2 + FRI")

st.sidebar.header("Threshold Settings")
t_pixel = st.sidebar.slider("Pixel High-risk Threshold", 0.3, 0.9, 0.55, 0.05)
t_low = st.sidebar.slider("FRI LOW Threshold", 0.1, 0.6, 0.35, 0.05)
t_high = st.sidebar.slider("FRI HIGH Threshold", 0.4, 0.9, 0.68, 0.05)

uploaded = st.file_uploader("Upload 1 patch .npz (Sen2Fire)", type=["npz"])

# ============================================================
# MAIN LOGIC
# ============================================================
if uploaded:
    model, mean13, std13, thr_ckpt, lr_b, lr_w = load_artifacts()

    with np.load(uploaded) as z:
        x = z["image"]
        aux = z["aux"]

    if x.shape[0] == 12:
        x = np.transpose(x, (1,2,0))

    # build 13ch
    x13 = np.concatenate([x, aux[...,None]], axis=-1)
    x13 = (x13 - mean13) / (std13 + 1e-6)

    X = torch.from_numpy(x13).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        logits = model(X)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()

    mask_high = prob >= t_pixel
    area_high = mask_high.mean()
    conf_high = prob[mask_high].mean() if mask_high.any() else 0.0

    score = lr_b[0] + lr_w[0]*area_high + lr_w[1]*conf_high
    fri = sigmoid_np(score)

    if fri < t_low:
        level = "LOW"
    elif fri < t_high:
        level = "MEDIUM"
    else:
        level = "HIGH"

    # ============================================================
    # VISUALIZATION
    # ============================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fire Probability Map")
        fig, ax = plt.subplots()
        im = ax.imshow(prob, cmap="magma")
        plt.colorbar(im, ax=ax)
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.subheader("Risk Overlay")
        fig, ax = plt.subplots()
        ax.imshow(to_rgb(x))
        ax.imshow(mask_high, cmap="Reds", alpha=0.45)
        ax.axis("off")
        ax.set_title(f"FRI={fri:.3f} | {level}")
        st.pyplot(fig)

    st.markdown("### 📊 Summary")
    st.write({
        "FRI": float(fri),
        "Risk Level": level,
        "High-risk area (%)": float(area_high * 100),
        "Confidence": float(conf_high)
    })
