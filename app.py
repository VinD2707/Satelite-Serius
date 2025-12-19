import os
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ============================================================
# UI CONFIG
# ============================================================
st.set_page_config(
    page_title="Fire Risk Mapping (Sentinel-2 + FRI)",
    layout="wide",
)

st.title("Fire Risk Mapping berbasis Segmentasi Sentinel-2 + Fire Risk Index (FRI)")
st.caption("Upload 1 file patch .npz (Sen2Fire), lalu app akan menghitung peta probabilitas api + FRI + klasifikasi risiko.")


# ============================================================
# MODEL DEFINITION (harus sama persis dengan training)
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

        logits = self.out(d1)
        return logits


# ============================================================
# UTILS
# ============================================================
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def robust_minmax(img, eps=1e-6):
    mn = float(np.nanmin(img))
    mx = float(np.nanmax(img))
    if mx - mn < eps:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - mn) / (mx - mn + eps)
    return out.astype(np.float32)

def to_rgb_composite(x_hwc, idx_r, idx_g, idx_b):
    """
    x_hwc: (H,W,C) float
    idx_r/g/b: index channel
    """
    r = robust_minmax(x_hwc[..., idx_r])
    g = robust_minmax(x_hwc[..., idx_g])
    b = robust_minmax(x_hwc[..., idx_b])
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return rgb

def load_npz_patch(uploaded_file):
    """
    Robust loader untuk Sen2Fire patch npz.
    - Auto cari key image (image/x/img)
    - Auto cari key aux (aux/aerosol/auxiliary)
    - Auto perbaiki CHW/HWC dan dimensi aux
    - Kalau aux tidak ada: isi nol (H,W)
    """
    z = np.load(uploaded_file)
    keys = list(z.files)

    # --- pilih key image ---
    img_key_candidates = ["image", "img", "x", "X", "s2", "S2"]
    img_key = next((k for k in img_key_candidates if k in keys), None)
    if img_key is None:
        # fallback: cari array 3D terbesar (biasanya image)
        arr3 = [(k, z[k]) for k in keys if z[k].ndim == 3]
        if not arr3:
            raise ValueError(f"NPZ tidak memiliki array 3D untuk image. keys={keys}")
        img_key = max(arr3, key=lambda t: t[1].size)[0]

    x = z[img_key]

    # --- pastikan image jadi HWC ---
    if x.ndim != 3:
        raise ValueError(f"Image harus 3D, tapi shape={x.shape}, key={img_key}, keys={keys}")

    # kalau channel di depan (C,H,W)
    if x.shape[0] in [12, 13] and x.shape[-1] not in [12, 13]:
        x = np.transpose(x, (1, 2, 0))

    H, W, C = x.shape
    if C < 3:
        raise ValueError(f"Image channel terlalu kecil: shape={x.shape}, key={img_key}, keys={keys}")

    # --- pilih key aux ---
    aux_key_candidates = ["aux", "aerosol", "aerosol_mask", "A", "auxiliary"]
    aux_key = next((k for k in aux_key_candidates if k in keys), None)

    aux = None
    if aux_key is not None:
        aux = z[aux_key]

        # aux bisa: (H,W) / (H,W,1) / (1,H,W)
        if aux.ndim == 3:
            if aux.shape[0] == 1 and aux.shape[1] == H and aux.shape[2] == W:
                aux = aux[0]
            elif aux.shape[2] == 1 and aux.shape[0] == H and aux.shape[1] == W:
                aux = aux[:, :, 0]
            else:
                aux = aux[..., 0]

        if aux.ndim != 2:
            raise ValueError(f"Aux harus 2D setelah diproses, tapi shape={aux.shape}, key={aux_key}, keys={keys}")

        if aux.shape != (H, W):
            raise ValueError(f"Mismatch ukuran aux vs image: aux={aux.shape} image={(H,W)} key={aux_key} keys={keys}")

    if aux is None:
        aux = np.zeros((H, W), dtype=np.float32)

    x = x.astype(np.float32)
    aux = aux.astype(np.float32)
    return x, aux, keys, img_key, aux_key

def build_x13(x_hwc_12, aux_hw):
    """
    x_hwc_12: (H,W,12)
    aux_hw  : (H,W)
    -> (H,W,13)
    """
    if x_hwc_12.ndim != 3:
        raise ValueError(f"x harus HWC, tapi ndim={x_hwc_12.ndim}, shape={x_hwc_12.shape}")
    if aux_hw.ndim != 2:
        raise ValueError(f"aux harus HW, tapi ndim={aux_hw.ndim}, shape={aux_hw.shape}")

    if x_hwc_12.shape[:2] != aux_hw.shape:
        raise ValueError(f"Mismatch x vs aux: x={x_hwc_12.shape} aux={aux_hw.shape}")

    x13 = np.concatenate([x_hwc_12, aux_hw[..., None]], axis=-1)
    return x13.astype(np.float32)

@torch.no_grad()
def predict_prob_map(model, x13_hwc, mean13, std13, device):
    """
    model: UNetSmall
    x13_hwc: (H,W,13)
    mean13/std13: (13,) atau broadcastable ke (H,W,13)
    return prob_map (H,W) float32
    """
    # normalize
    x = (x13_hwc - mean13) / (std13 + 1e-6)

    # to tensor (1,13,H,W)
    X = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).float().to(device)

    logits = model(X)           # (1,1,H,W)
    probs = torch.sigmoid(logits).squeeze(0).squeeze(0)  # (H,W)
    return probs.detach().cpu().numpy().astype(np.float32)

def compute_risk_features(prob_map, t_pixel_high):
    """
    prob_map: (H,W) in [0,1]
    t_pixel_high: threshold high-risk pixel
    output:
      mask_high (H,W) bool
      area_high (0..1)
      conf_high (0..1)
    """
    mask_high = (prob_map >= float(t_pixel_high))
    area_high = float(mask_high.mean())
    if mask_high.any():
        conf_high = float(prob_map[mask_high].mean())
    else:
        conf_high = 0.0
    return mask_high, area_high, conf_high

def compute_fri_from_lr(area_high, conf_high, lr_intercept, lr_coef):
    """
    lr_intercept: scalar
    lr_coef: shape (2,) -> [coef_area, coef_conf]
    """
    z = float(lr_intercept) + float(lr_coef[0]) * float(area_high) + float(lr_coef[1]) * float(conf_high)
    return float(sigmoid_np(z))

def classify_risk(fri, t_low, t_high):
    if fri < t_low:
        return "LOW"
    elif fri < t_high:
        return "MEDIUM"
    else:
        return "HIGH"


# ============================================================
# LOAD ARTIFACTS (no pickle)
# ============================================================
@st.cache_resource(show_spinner=True)
def load_artifacts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- meta (mean/std/thr)
    meta_path = "unet13_meta.npz"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Tidak ketemu {meta_path} di repo root.")

    meta = np.load(meta_path)

    # try common key names
    # kamu dulu simpan: mean_13, std_13, thr
    mean_key = "mean_13" if "mean_13" in meta.files else ("mean13" if "mean13" in meta.files else None)
    std_key  = "std_13"  if "std_13"  in meta.files else ("std13"  if "std13"  in meta.files else None)
    thr_key  = "thr"     if "thr"     in meta.files else None

    if mean_key is None or std_key is None:
        raise ValueError(f"unet13_meta.npz tidak punya mean/std. keys={list(meta.files)}")

    mean13 = meta[mean_key].astype(np.float32)
    std13  = meta[std_key].astype(np.float32)

    # pastikan shape (13,)
    mean13 = mean13.reshape(-1)
    std13  = std13.reshape(-1)
    if mean13.shape[0] != 13 or std13.shape[0] != 13:
        raise ValueError(f"mean/std harus 13 elemen. mean={mean13.shape}, std={std13.shape}")

    thr_ckpt = float(meta[thr_key]) if thr_key is not None else 0.75

    # --- model weights
    ckpt_path = "unet13_best.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Tidak ketemu {ckpt_path} di repo root.")

    model = UNetSmall(in_channels=13, out_channels=1, base=32).to(device)
    model.eval()

    # torch 2.6+ default weights_only=True kadang bikin error kalau ckpt berisi numpy
    # Kita coba beberapa metode aman:
    ckpt = None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception:
        # fallback allowlist (kalau error "Unsupported global numpy._core.multiarray._reconstruct")
        try:
            import numpy as _np
            import torch.serialization as _ts
            with _ts.safe_globals([_np.core.multiarray._reconstruct]):
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e2:
            raise RuntimeError(f"Gagal load checkpoint: {e2}")

    # ckpt bisa dict dengan key model_state, atau langsung state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    # strict=False supaya kalau ada minor mismatch, app tetap jalan dan kamu bisa lihat log
    if missing or unexpected:
        st.warning(f"load_state_dict strict=False. missing={missing}, unexpected={unexpected}")

    # --- LR risk coeffs (npz)
    lr_path = "lr_risk_coeffs.npz"
    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"Tidak ketemu {lr_path} di repo root.")

    lr_npz = np.load(lr_path)
    # kita dukung beberapa nama umum
    # ideal: intercept (1,), coef (2,)
    if "intercept" in lr_npz.files:
        lr_intercept = float(np.array(lr_npz["intercept"]).reshape(-1)[0])
    elif "b0" in lr_npz.files:
        lr_intercept = float(np.array(lr_npz["b0"]).reshape(-1)[0])
    else:
        raise ValueError(f"lr_risk_coeffs.npz tidak punya intercept/b0. keys={list(lr_npz.files)}")

    if "coef" in lr_npz.files:
        lr_coef = np.array(lr_npz["coef"]).reshape(-1).astype(np.float32)
    elif "weights" in lr_npz.files:
        lr_coef = np.array(lr_npz["weights"]).reshape(-1).astype(np.float32)
    else:
        raise ValueError(f"lr_risk_coeffs.npz tidak punya coef/weights. keys={list(lr_npz.files)}")

    if lr_coef.shape[0] != 2:
        raise ValueError(f"Koef LR harus 2 elemen [area_high, conf_high]. coef_shape={lr_coef.shape}")

    return device, model, mean13, std13, thr_ckpt, lr_intercept, lr_coef


# ============================================================
# SIDEBAR: THRESHOLDS
# ============================================================
with st.sidebar:
    st.header("Threshold Settings")

    # default dari meta untuk seg-thr kalau kamu mau pakai (tapi di risk mapping kita pakai pixel-high threshold)
    device, model, mean13, std13, thr_ckpt, lr_intercept, lr_coef = load_artifacts()

    t_pixel_high = st.slider(
        "Pixel High-risk Threshold (t_pixel_high)",
        min_value=0.10, max_value=0.95, value=0.55, step=0.01
    )
    t_low = st.slider(
        "FRI Threshold LOW (t_low)",
        min_value=0.00, max_value=0.99, value=0.35, step=0.01
    )
    t_high = st.slider(
        "FRI Threshold HIGH (t_high)",
        min_value=0.00, max_value=0.99, value=0.68, step=0.01
    )

    st.caption("Catatan: t_pixel_high adalah threshold di peta probabilitas (pixel-wise).")
    st.caption("t_low / t_high adalah threshold untuk klasifikasi risiko patch (FRI).")

    with st.expander("Loaded artifacts info"):
        st.write("Device:", str(device))
        st.write("mean13 shape:", mean13.shape)
        st.write("std13 shape:", std13.shape)
        st.write("thr_ckpt (seg thr):", thr_ckpt)
        st.write("LR intercept:", lr_intercept)
        st.write("LR coef [area_high, conf_high]:", lr_coef.tolist())


# ============================================================
# MAIN: UPLOAD + RUN
# ============================================================
st.subheader("Upload 1 file patch .npz (Sen2Fire)")

uploaded = st.file_uploader("Upload .npz patch", type=["npz"])

if uploaded is None:
    st.info("Silakan upload 1 file .npz untuk memulai.")
    st.stop()

# --- Load patch robustly
try:
    x12, aux, keys, img_key, aux_key = load_npz_patch(uploaded)

    st.caption(f"✅ NPZ loaded | keys={keys}")
    st.caption(f"image_key={img_key} shape={x12.shape} | aux_key={aux_key} shape={aux.shape}")

except Exception as e:
    st.error("Gagal membaca file NPZ. Ini biasanya karena nama key / shape tidak sesuai.")
    st.write("Error detail:")
    st.exception(e)

    # debug keys
    try:
        uploaded.seek(0)
        z_debug = np.load(uploaded)
        st.write("Keys di file ini:", list(z_debug.files))
        for k in z_debug.files[:15]:
            arr = z_debug[k]
            st.write(f"- {k}: shape={arr.shape}, dtype={arr.dtype}, ndim={arr.ndim}")
    except Exception as e2:
        st.write("Tidak bisa membuka NPZ untuk debug.")
        st.exception(e2)

    st.stop()

# --- Build 13ch
try:
    # Pastikan x12 memang 12 channel; kalau lebih dari 12, ambil 12 pertama
    if x12.shape[-1] >= 12:
        x12_use = x12[..., :12]
    else:
        raise ValueError(f"Image channel < 12, shape={x12.shape}")

    x13 = build_x13(x12_use, aux)

except Exception as e:
    st.error("Gagal membangun tensor 13 channel (image+aux).")
    st.exception(e)
    st.stop()

# --- Predict
try:
    prob_map = predict_prob_map(model, x13, mean13, std13, device)
    mask_high, area_high, conf_high = compute_risk_features(prob_map, t_pixel_high)

    fri = compute_fri_from_lr(area_high, conf_high, lr_intercept, lr_coef)
    level = classify_risk(fri, t_low, t_high)

except Exception as e:
    st.error("Gagal melakukan inference model / risk computation.")
    st.exception(e)
    st.stop()

# ============================================================
# VISUALIZATION
# ============================================================
# default band indices (0-based):
# kamu dulu pakai: {"B2":1, "B3":2, "B4":3, "B8":7, "B12":11}
# berarti B4=3, B3=2, B2=1 untuk RGB
band_defaults = {
    "B2 (Blue)": 1,
    "B3 (Green)": 2,
    "B4 (Red)": 3,
    "B8 (NIR)": 7,
    "B11 (SWIR1)": 10,
    "B12 (SWIR2)": 11,
}

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.markdown("### Fire Probability Map")
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(prob_map, cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Pixel-wise Fire Probability")
    plt.axis("off")
    st.pyplot(fig)
    plt.close(fig)

with colB:
    st.markdown("### Risk Mapping (Overlay)")
    # RGB
    rgb = to_rgb_composite(x12_use, band_defaults["B4 (Red)"], band_defaults["B3 (Green)"], band_defaults["B2 (Blue)"])

    fig2 = plt.figure(figsize=(7, 6))
    plt.imshow(rgb)
    # overlay high-risk
    plt.imshow(mask_high.astype(np.float32), cmap="Reds", alpha=0.45)
    plt.title(f"FRI={fri:.3f} | {level}\nHigh-risk={area_high*100:.2f}% | Conf={conf_high:.3f}")
    plt.axis("off")
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")
st.markdown("## Summary")

st.write({
    "FRI": fri,
    "risk_level": level,
    "area_high": area_high,
    "conf_high": conf_high,
    "t_pixel_high": float(t_pixel_high),
    "t_low": float(t_low),
    "t_high": float(t_high),
})

st.caption("Interpretasi cepat:")
st.markdown(
    "- **Fire Probability Map**: warna lebih terang = probabilitas api lebih tinggi per pixel.\n"
    "- **Overlay merah**: hanya pixel dengan **prob ≥ t_pixel_high**.\n"
    "- **area_high**: persentase area patch yang masuk high-risk.\n"
    "- **conf_high**: rata-rata probabilitas pada area high-risk.\n"
    "- **FRI**: skor patch-level dari Logistic Regression (fit dari val), lalu diklasifikasikan menjadi LOW/MEDIUM/HIGH."
)
