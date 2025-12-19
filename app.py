import os
import io
import pickle
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================
# Config / filenames (root)
# =========================
CKPT_PATH = "unet13_best.pth"
META_PATH = "unet13_meta.npz"
LR_PATH   = "lr_risk.pkl"

# Default thresholds (dari hasil kamu sebelumnya)
DEFAULT_T_PIXEL_HIGH = 0.55
DEFAULT_T_LOW  = 0.35083405887107555
DEFAULT_T_HIGH = 0.6820491097886825

# =========================
# Model definition (MUST MATCH TRAINING)
# =========================
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
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

# =========================
# Helpers
# =========================
def safe_torch_load(path, map_location="cpu"):
    """
    PyTorch 2.6+ default weights_only=True bisa bikin error untuk objek numpy.
    Kita paksa load full dict (trusted file milik sendiri).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # older torch doesn't support weights_only
        return torch.load(path, map_location=map_location)

def to_hwc(x):
    # (12,H,W) -> (H,W,12)
    if x.ndim == 3 and x.shape[0] in [12, 13] and x.shape[1] > 64 and x.shape[2] > 64:
        return np.transpose(x, (1,2,0))
    return x

def pick_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    # fallback: cari yang paling mirip
    for k in npz.files:
        low = k.lower()
        for c in candidates:
            if c.lower() in low:
                return k
    return None

def build_13ch_tensor(x12_hwc, aerosol_hw, mean13, std13):
    # x12_hwc: (H,W,12)
    # aerosol_hw: (H,W)
    if aerosol_hw is None:
        aerosol_hw = np.zeros((x12_hwc.shape[0], x12_hwc.shape[1]), dtype=np.float32)

    x13 = np.concatenate([x12_hwc, aerosol_hw[..., None]], axis=-1)  # (H,W,13)
    x13 = x13.astype(np.float32)

    mean13 = np.array(mean13, dtype=np.float32).reshape(1,1,13)
    std13  = np.array(std13, dtype=np.float32).reshape(1,1,13)
    x13n = (x13 - mean13) / (std13 + 1e-8)

    # to torch (1,13,H,W)
    x13n = np.transpose(x13n, (2,0,1))
    X = torch.from_numpy(x13n).unsqueeze(0)
    return X

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def rgb_composite_from_12(x12_hwc, b4=3, b3=2, b2=1):
    # gunakan indeks band_map kamu: B2=1, B3=2, B4=3
    R = x12_hwc[..., b4].astype(np.float32)
    G = x12_hwc[..., b3].astype(np.float32)
    B = x12_hwc[..., b2].astype(np.float32)

    rgb = np.stack([R,G,B], axis=-1)

    # simple robust normalization (percentile stretch)
    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    rgb = np.clip(rgb, 0, 1)
    return rgb

def classify_risk(fri, t_low, t_high):
    if fri < t_low:
        return "LOW"
    elif fri <= t_high:
        return "MEDIUM"
    else:
        return "HIGH"

# =========================
# Load artifacts (cached)
# =========================
@st.cache_resource
def load_artifacts():
    # device
    device = torch.device("cpu")

    # load ckpt dict
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = safe_torch_load(CKPT_PATH, map_location=device)

    # model
    model = UNetSmall(in_channels=13, out_channels=1, base=32).to(device)
    if "model_state" not in ckpt:
        raise KeyError("Key 'model_state' not found in checkpoint. Pastikan file unet13_best.pth adalah hasil torch.save({...}).")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # mean/std from ckpt (utama), fallback to meta npz
    mean13 = ckpt.get("mean_13", None)
    std13  = ckpt.get("std_13", None)
    thr    = float(ckpt.get("thr", DEFAULT_T_PIXEL_HIGH))

    if (mean13 is None or std13 is None) and os.path.exists(META_PATH):
        with np.load(META_PATH, allow_pickle=True) as m:
            # kalau meta kamu beda key, aman: kita coba cari
            k_mean = pick_key(m, ["mean_13", "MEAN_13", "mean13"])
            k_std  = pick_key(m, ["std_13", "STD_13", "std13"])
            k_thr  = pick_key(m, ["thr", "THR", "t_pixel_high"])
            if mean13 is None and k_mean: mean13 = m[k_mean]
            if std13  is None and k_std:  std13  = m[k_std]
            if k_thr: thr = float(m[k_thr])

    if mean13 is None or std13 is None:
        raise ValueError("mean_13/std_13 tidak ditemukan. Pastikan ckpt menyimpan mean_13 dan std_13 atau file unet13_meta.npz berisi mean/std.")

    # load lr risk model
    if not os.path.exists(LR_PATH):
        raise FileNotFoundError(f"Risk model not found: {LR_PATH}")
    with open(LR_PATH, "rb") as f:
        lr_model = pickle.load(f)

    return device, model, lr_model, mean13, std13, thr

# =========================
# UI
# =========================
st.set_page_config(page_title="Fire Risk Mapping (Sentinel-2 + FRI)", layout="wide")
st.title("Fire Risk Mapping berbasis Segmentasi Sentinel-2 + Fire Risk Index (FRI)")

with st.sidebar:
    st.header("Threshold Settings")
    t_pixel_high = st.slider("Pixel High-risk Threshold (t_pixel_high)", 0.05, 0.95, float(DEFAULT_T_PIXEL_HIGH), 0.01)
    t_low  = st.slider("FRI Threshold LOW (t_low)", 0.0, 1.0, float(DEFAULT_T_LOW), 0.01)
    t_high = st.slider("FRI Threshold HIGH (t_high)", 0.0, 1.0, float(DEFAULT_T_HIGH), 0.01)
    st.caption("Catatan: t_pixel_high adalah threshold di peta probabilitas (pixel-wise). t_low/t_high untuk klasifikasi FRI (patch-wise).")

uploaded = st.file_uploader("Upload 1 file patch .npz (Sen2Fire)", type=["npz"])

# Load artifacts once
device, model, lr_model, mean13, std13, thr_ckpt = load_artifacts()

st.info(f"Artifacts loaded: model=UNetSmall(13ch), checkpoint='{CKPT_PATH}', lr='{LR_PATH}'")

if uploaded is None:
    st.stop()

# =========================
# Read NPZ safely
# =========================
npz_bytes = uploaded.read()
npz_file = np.load(io.BytesIO(npz_bytes), allow_pickle=True)

k_img = pick_key(npz_file, ["image", "img", "x", "X"])
k_aer = pick_key(npz_file, ["aerosol", "aero", "aerosol_product", "A"])

if k_img is None:
    st.error(f"Tidak menemukan key image di npz. Keys tersedia: {npz_file.files}")
    st.stop()

x = npz_file[k_img]
x = to_hwc(x)

# Pastikan 12 band
if x.ndim != 3 or x.shape[-1] < 12:
    st.error(f"Shape image tidak sesuai. Dapat: {x.shape}. Harusnya (H,W,12) atau (12,H,W).")
    st.stop()

x12 = x[..., :12].astype(np.float32)

aer = None
if k_aer is not None:
    aer_raw = npz_file[k_aer]
    if aer_raw.ndim == 3 and aer_raw.shape[0] == 1:
        aer_raw = aer_raw[0]
    if aer_raw.ndim == 3 and aer_raw.shape[-1] == 1:
        aer_raw = aer_raw[..., 0]
    if aer_raw.ndim == 2:
        aer = aer_raw.astype(np.float32)

# =========================
# Predict prob_map
# =========================
X = build_13ch_tensor(x12, aer, mean13, std13).to(device)

with torch.no_grad():
    logits = model(X)                 # (1,1,H,W)
    probs = torch.sigmoid(logits)[0,0].cpu().numpy().astype(np.float32)

prob_map = probs
mask_high = (prob_map >= float(t_pixel_high))
area_high = float(mask_high.mean())  # fraction of pixels
conf_high = float(prob_map[mask_high].mean()) if mask_high.any() else 0.0

# =========================
# FRI from Logistic Regression
# Features: [area_high, conf_high]
# =========================
feat = np.array([[area_high, conf_high]], dtype=np.float32)
try:
    fri = float(lr_model.predict_proba(feat)[0,1])
except Exception:
    # fallback if model doesn't support predict_proba
    score = float(lr_model.decision_function(feat)[0])
    fri = float(sigmoid_np(score))

risk_level = classify_risk(fri, float(t_low), float(t_high))

# =========================
# Visualize
# =========================
rgb = rgb_composite_from_12(x12, b4=3, b3=2, b2=1)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fire Probability Map")
    fig = plt.figure(figsize=(6,6))
    plt.imshow(prob_map, cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")
    st.pyplot(fig)

with col2:
    st.subheader("Risk Overlay")
    fig = plt.figure(figsize=(6,6))
    plt.imshow(rgb)
    plt.imshow(mask_high.astype(np.float32), cmap="Reds", alpha=0.45)
    plt.axis("off")
    plt.title(f"FRI={fri:.3f} | {risk_level}\nHigh-risk area={area_high*100:.2f}% | Conf={conf_high:.3f}")
    st.pyplot(fig)

st.markdown("### Summary")
st.write({
    "FRI": fri,
    "Risk Level": risk_level,
    "High-risk Area (%)": area_high * 100.0,
    "Confidence (mean p on high pixels)": conf_high,
    "t_pixel_high": float(t_pixel_high),
    "t_low": float(t_low),
    "t_high": float(t_high),
    "npz_keys_found": {"image_key": k_img, "aerosol_key": k_aer}
})
