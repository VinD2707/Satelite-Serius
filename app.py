import os
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
import joblib

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path model + meta
CKPT_PATH = "./checkpoints/unet13_weights_only.pth"   # ganti sesuai file kamu
META_PATH = "./checkpoints/unet13_meta.npz"           # berisi mean_13,std_13,thr,t_pixel_high,t_low,t_high
LR_PATH   = "./checkpoints/lr_risk.joblib"            # optional (kalau pakai Logistic Regression risk)

# Band index mapping (sesuaikan jika dataset kamu beda urutan)
BAND_MAP = {"B2": 1, "B3": 2, "B4": 3, "B8": 7, "B12": 11}  # contoh

# =========================
# MODEL DEFINITION
# (PASTIKAN sama persis dengan training)
# =========================
# Jika kamu sudah punya class UNet di notebook, copy class itu ke sini.
# Di bawah ini hanya placeholder minimal, HARUS kamu ganti dengan UNet kamu.

class DummyUNet(torch.nn.Module):
    def __init__(self, in_ch=13):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_model():
    model = DummyUNet(in_ch=13)  # GANTI: UNet kamu
    model.to(DEVICE)
    model.eval()

    # weights_only file -> isinya state_dict langsung
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    return model


# =========================
# UTILITIES
# =========================
def to_rgb_composite(x_hw12, r_idx, g_idx, b_idx):
    """x_hw12 float/whatever -> normalize to 0..1 for display"""
    rgb = np.stack([x_hw12[..., r_idx], x_hw12[..., g_idx], x_hw12[..., b_idx]], axis=-1).astype(np.float32)

    # robust min-max per-channel
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        v = rgb[..., c]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        if hi - lo < 1e-6:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((v - lo) / (hi - lo), 0, 1)
    return out

def build_13ch_tensor(x_hw12, aerosol_hw1, mean13, std13):
    """
    input:
      x_hw12: (H,W,12)
      aerosol_hw1: (H,W) or (H,W,1)
    output:
      torch tensor (13,H,W) normalized
    """
    if aerosol_hw1.ndim == 3:
        aerosol_hw1 = aerosol_hw1[..., 0]
    x13 = np.concatenate([x_hw12, aerosol_hw1[..., None]], axis=-1)  # (H,W,13)
    x13 = np.transpose(x13, (2, 0, 1)).astype(np.float32)            # (13,H,W)

    mean13 = mean13.reshape(-1, 1, 1).astype(np.float32)
    std13  = std13.reshape(-1, 1, 1).astype(np.float32)
    x13 = (x13 - mean13) / (std13 + 1e-6)

    return torch.from_numpy(x13)

@torch.no_grad()
def predict_prob_map(model, x_hw12, aerosol_hw1, mean13, std13):
    X13 = build_13ch_tensor(x_hw12, aerosol_hw1, mean13, std13).unsqueeze(0).to(DEVICE)  # (1,13,H,W)
    logits = model(X13)  # (1,1,H,W)
    prob = torch.sigmoid(logits)[0,0].detach().cpu().numpy()  # (H,W)
    return prob

def risk_from_prob(prob_map, t_pixel_high, t_low, t_high, lr_model=None):
    """
    Return:
      mask_high, area_high, conf_high, fri, level
    """
    mask_high = (prob_map >= t_pixel_high).astype(np.uint8)
    area_high = float(mask_high.mean())   # ratio 0..1 dari pixel yang high-risk

    conf_high = float(prob_map[mask_high == 1].mean()) if mask_high.sum() > 0 else 0.0

    # --- FRI ---
    if lr_model is not None:
        # fitur sederhana: [area_high, conf_high]
        fri = float(lr_model.predict_proba([[area_high, conf_high]])[0, 1])
    else:
        # fallback: FRI = area_high * conf_high (simple & stable)
        fri = float(area_high * conf_high)

    # --- risk level ---
    if fri < t_low:
        level = "LOW"
    elif fri < t_high:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return mask_high, area_high, conf_high, fri, level


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Fire Risk Mapping (Sentinel-2)", layout="wide")
st.title("🔥 Fire Risk Mapping (Sentinel-2 Patch)")

# Load meta
if not os.path.exists(META_PATH):
    st.error("META file tidak ditemukan. Pastikan ./checkpoints/unet13_meta.npz ada.")
    st.stop()

meta = np.load(META_PATH, allow_pickle=True)
MEAN_13 = meta["mean_13"]
STD_13  = meta["std_13"]
T_PIXEL_HIGH = float(meta["t_pixel_high"]) if "t_pixel_high" in meta else 0.55
T_LOW  = float(meta["t_low"]) if "t_low" in meta else 0.35
T_HIGH = float(meta["t_high"]) if "t_high" in meta else 0.68

# Load LR if exists
lr_model = None
if os.path.exists(LR_PATH):
    try:
        lr_model = joblib.load(LR_PATH)
        st.sidebar.success("Logistic Regression risk loaded.")
    except Exception as e:
        st.sidebar.warning(f"LR load gagal: {e}")

# Load model
if not os.path.exists(CKPT_PATH):
    st.error("Checkpoint model tidak ditemukan. Pastikan file model ada di ./checkpoints/")
    st.stop()

model = load_model()
st.sidebar.success(f"Model loaded on {DEVICE}")

st.sidebar.markdown("### Threshold Settings")
t_pixel_high = st.sidebar.slider("Pixel high-risk threshold", 0.05, 0.95, float(T_PIXEL_HIGH), 0.01)
t_low = st.sidebar.slider("FRI low threshold", 0.00, 1.00, float(T_LOW), 0.01)
t_high = st.sidebar.slider("FRI high threshold", 0.00, 1.00, float(T_HIGH), 0.01)

uploaded = st.file_uploader("Upload patch .npz (Sen2Fire format)", type=["npz"])

if uploaded is None:
    st.info("Silakan upload 1 file patch .npz untuk memulai.")
    st.stop()

# Read npz
with np.load(uploaded) as data:
    # sesuaikan key sesuai dataset kamu
    # umumnya: image, label/mask, aerosol
    x = data["image"]
    if x.shape[0] == 12:  # CHW -> HWC
        x = np.transpose(x, (1,2,0))
    aerosol = data["aerosol"] if "aerosol" in data else None
    y = data["label"] if "label" in data else (data["mask"] if "mask" in data else None)

if aerosol is None:
    st.error("Key aerosol tidak ditemukan di npz. Pastikan patch punya aerosol.")
    st.stop()

if aerosol.ndim == 3:
    aerosol = aerosol[..., 0]

# Predict
prob_map = predict_prob_map(model, x, aerosol, MEAN_13, STD_13)
mask_high, area_high, conf_high, fri, level = risk_from_prob(
    prob_map,
    t_pixel_high=t_pixel_high,
    t_low=t_low,
    t_high=t_high,
    lr_model=lr_model
)

# Visuals
rgb = to_rgb_composite(x, BAND_MAP["B4"], BAND_MAP["B3"], BAND_MAP["B2"])

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
    plt.imshow(mask_high, cmap="Reds", alpha=0.45)
    plt.axis("off")
    plt.title(f"FRI={fri:.3f} | {level} | high={area_high*100:.2f}% | conf={conf_high:.3f}")
    st.pyplot(fig)

st.markdown("### Summary")
st.write({
    "FRI": float(fri),
    "Risk Level": level,
    "High-risk Area (%)": float(area_high * 100.0),
    "High-risk Confidence": float(conf_high),
    "Pixel high-risk threshold": float(t_pixel_high),
    "FRI thresholds": {"LOW<": float(t_low), "MEDIUM<": float(t_high), "HIGH>=": float(t_high)},
})