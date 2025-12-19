# app.py (ROOT-BASED: file artefak ada di folder root repo)

import io
import numpy as np
import joblib
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------------
# Paths (semua di ROOT repo)
# -------------------------
CKPT_PATH = "./unet13_best.pth"
META_PATH = "./unet13_meta.npz"
LR_PATH   = "./lr_risk.pkl"

# ============================================================
# 1) PASTE KELAS U-NET KAMU DI SINI (HARUS SAMA PERSIS)
# ============================================================
# Contoh placeholder (JANGAN dipakai kalau beda dengan training-mu)
# -> Ganti ini dengan class U-Net yang kamu pakai saat training.
class UNetYOUR(nn.Module):
    def __init__(self, in_ch=13, out_ch=1):
        super().__init__()
        # TODO: paste arsitektur U-Net aslimu di sini
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------------
# Helper: pilih key npz secara aman
# -------------------------
def pick_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None

def load_npz_from_upload(uploaded_file):
    """Load npz dari upload Streamlit. Output: x12(H,W,12), aerosol(H,W) or None, keys(list)."""
    data = np.load(io.BytesIO(uploaded_file.read()))
    keys = list(data.keys())

    x_key = pick_key(data, ["image", "x", "s2", "img", "bands"])
    a_key = pick_key(data, ["aerosol", "aot", "aero", "a"])

    if x_key is None:
        raise ValueError(f"Tidak menemukan key citra 12-band. Keys tersedia: {keys}")

    x = data[x_key]  # (12,H,W) atau (H,W,12)
    if x.ndim != 3:
        raise ValueError(f"Citra harus 3D (12,H,W) atau (H,W,12). Dapat: {x.shape}")

    # ubah ke HWC
    if x.shape[0] == 12:
        x = np.transpose(x, (1,2,0))
    if x.shape[-1] != 12:
        raise ValueError(f"Jumlah channel harus 12. Dapat: {x.shape}")

    x = x.astype(np.float32)

    # aerosol optional
    a = None
    if a_key is not None:
        a = data[a_key]
        if np.isscalar(a):
            a = np.full((x.shape[0], x.shape[1]), float(a), dtype=np.float32)
        else:
            a = np.array(a)
            if a.ndim == 3 and a.shape[0] == 1:
                a = a[0]
            if a.ndim != 2:
                # fallback reshape kalau size cocok
                if a.size == x.shape[0]*x.shape[1]:
                    a = a.reshape(x.shape[0], x.shape[1]).astype(np.float32)
                else:
                    raise ValueError(f"Aerosol harus 2D (H,W) atau scalar. Dapat: {a.shape}")
        a = a.astype(np.float32)

    return x, a, keys

def to_rgb_composite(x12_hwc, idx_r, idx_g, idx_b):
    """Simple RGB composite + percentile stretch."""
    rgb = np.stack([x12_hwc[..., idx_r], x12_hwc[..., idx_g], x12_hwc[..., idx_b]], axis=-1)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = (rgb - p2) / (p98 - p2 + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    return rgb

def build_13ch_tensor(x12_hwc, aerosol_hw, mean13, std13):
    H, W, _ = x12_hwc.shape
    if aerosol_hw is None:
        aerosol_hw = np.zeros((H, W), dtype=np.float32)

    x13 = np.concatenate([x12_hwc, aerosol_hw[..., None]], axis=-1)  # H,W,13
    x13 = (x13 - mean13) / (std13 + 1e-6)                           # broadcast
    x13_chw = np.transpose(x13, (2,0,1))                            # 13,H,W
    return torch.from_numpy(x13_chw).float()

@torch.no_grad()
def predict_prob_map(model, x12_hwc, aerosol_hw, mean13, std13, device):
    X = build_13ch_tensor(x12_hwc, aerosol_hw, mean13, std13).unsqueeze(0).to(device)  # 1,13,H,W
    logits = model(X)                      # 1,1,H,W
    prob = torch.sigmoid(logits)[0,0].detach().cpu().numpy()
    return prob

def compute_risk_features(prob_map, t_pixel_high=0.55):
    mask_high = (prob_map >= t_pixel_high).astype(np.uint8)
    area_high = float(mask_high.mean())  # fraction 0..1
    conf_high = float(prob_map[mask_high==1].mean()) if mask_high.sum() > 0 else 0.0
    return mask_high, area_high, conf_high

def classify_risk_from_fri(fri, t_low, t_high):
    if fri < t_low:
        return "LOW"
    elif fri < t_high:
        return "MEDIUM"
    else:
        return "HIGH"

# -------------------------
# Cache load artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # meta
    meta = np.load(META_PATH)
    if "mean_13" not in meta or "std_13" not in meta:
        raise ValueError(f"unet13_meta.npz harus punya key mean_13 & std_13. Keys: {list(meta.keys())}")

    mean13 = meta["mean_13"].astype(np.float32).reshape(1,1,-1)
    std13  = meta["std_13"].astype(np.float32).reshape(1,1,-1)

    # lr model
    lr = joblib.load(LR_PATH)

    # load model
    model = UNetYOUR(in_ch=13, out_ch=1).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # kamu simpan key "model_state"
    if "model_state" not in ckpt:
        raise ValueError(f"Checkpoint tidak punya key 'model_state'. Keys: {list(ckpt.keys())}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # defaults
    # threshold pixel high biasanya kamu pilih dari sweep dice (contoh 0.55)
    t_pixel_high_default = 0.55

    return device, model, lr, mean13, std13, t_pixel_high_default

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Fire Risk Mapping", layout="wide")
st.title("Fire Risk Mapping berbasis Segmentasi Sentinel-2 + Fire Risk Index (FRI)")

device, model, lr, mean13, std13, t_pix_default = load_artifacts()

st.sidebar.header("Pengaturan Threshold")

t_pixel_high = st.sidebar.slider("Pixel high-risk threshold (t_pixel_high)", 0.10, 0.95, float(t_pix_default), 0.01)

# pakai hasil val-mu (dari distribusi FRI)
t_low  = st.sidebar.number_input("FRI threshold LOW/MEDIUM (t_low)", value=0.350834, format="%.6f")
t_high = st.sidebar.number_input("FRI threshold MEDIUM/HIGH (t_high)", value=0.682049, format="%.6f")

uploaded = st.file_uploader("Upload 1 patch .npz (12-band + aerosol optional)", type=["npz"])

if uploaded is None:
    st.info("Upload file .npz untuk menghasilkan peta probabilitas dan level risiko.")
    st.stop()

try:
    x12, aerosol, keys = load_npz_from_upload(uploaded)

    prob_map = predict_prob_map(model, x12, aerosol, mean13, std13, device)
    mask_high, area_high, conf_high = compute_risk_features(prob_map, t_pixel_high=t_pixel_high)

    # FRI via Logistic Regression: fitur [area_high, conf_high]
    fri = float(lr.predict_proba([[area_high, conf_high]])[0,1])
    level = classify_risk_from_fri(fri, t_low=t_low, t_high=t_high)

    band_map = {"B2":1, "B3":2, "B4":3, "B8":7, "B11":10, "B12":11}
    rgb = to_rgb_composite(x12, band_map["B4"], band_map["B3"], band_map["B2"])

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Fire Probability Map")
        fig = plt.figure(figsize=(6,6))
        plt.imshow(prob_map, cmap="magma")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        st.pyplot(fig)

    with c2:
        st.subheader("Risk Overlay")
        fig = plt.figure(figsize=(6,6))
        plt.imshow(rgb)
        plt.imshow(mask_high, cmap="Reds", alpha=0.45)
        plt.axis("off")
        plt.title(f"FRI={fri:.3f} | {level}\nHigh-risk={area_high*100:.2f}% | Conf={conf_high:.3f}")
        st.pyplot(fig)

    st.markdown("### Ringkasan")
    st.write({
        "npz_keys": keys,
        "FRI": fri,
        "risk_level": level,
        "area_high(%)": area_high*100,
        "conf_high": conf_high,
        "t_pixel_high": t_pixel_high,
        "t_low": t_low,
        "t_high": t_high
    })

except Exception as e:
    st.error(str(e))
    st.stop()
