import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import LungCTPredictor  # noqa: E402

st.set_page_config(
    page_title="Deteksi Kanker Paru-Paru — Citra CT",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #f4f7fa; }

    .main-header {
        background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1rem; }

    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 5px solid #f9a825;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #5d4a00;
        margin-bottom: 1.5rem;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-top: 1rem;
        color: white;
    }
    .result-card.cancer { background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%); }
    .result-card.no_cancer { background: linear-gradient(135deg, #1e8449 0%, #27ae60 100%); }
    .result-card.rejected { background: linear-gradient(135deg, #616161 0%, #9e9e9e 100%); }
    .result-card h2 { margin: 0 0 0.3rem 0; font-size: 1.5rem; }
    .result-card p { margin: 0.2rem 0; opacity: 0.95; }

    .prob-bar-track {
        background-color: #e0e0e0;
        border-radius: 20px;
        height: 26px;
        width: 100%;
        overflow: hidden;
        margin-top: 0.6rem;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        transition: width 0.4s ease;
    }

    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #0f4c75; }
    .metric-card .label { font-size: 0.8rem; color: #666; margin-top: 0.2rem; }

    section[data-testid="stSidebar"] { background-color: #0f2d3d; }
    section[data-testid="stSidebar"] * { color: #e8f1f5 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_predictor():
    return LungCTPredictor()


@st.cache_data
def load_metrics():
    reports_dir = ROOT / "reports"
    try:
        ensemble = json.loads((reports_dir / "ensemble_comparison.json").read_text())
        return ensemble.get("ensemble_v5_v6", {})
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🫁 Tentang Sistem")
    st.markdown(
        "Sistem klasifikasi biner **cancer / no_cancer** dari citra CT paru-paru, "
        "menggunakan transfer learning EfficientNet-B0 dengan ensemble 2 model."
    )

    metrics = load_metrics()
    if metrics:
        st.markdown("### 📊 Performa Model (test set)")
        c1, c2 = st.columns(2)
        c1.metric("Akurasi", f"{metrics.get('accuracy', 0) * 100:.1f}%")
        c2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

    st.markdown("### 🔍 Cara Kerja")
    st.markdown(
        "1. Citra diperiksa dulu — apakah benar citra CT paru-paru "
        "(deteksi out-of-distribution).\n"
        "2. Kalau valid, dua model (v5 + v6) memberi prediksi, hasilnya dirata-rata.\n"
        "3. Skor probabilitas ditampilkan bersama rekomendasi."
    )

    st.markdown("---")
    st.markdown(
        "<small>Dikembangkan sebagai bagian skripsi — "
        "Muhammad Ilhamsyah Ridwan, Teknik Informatika, FMIPA Unpad.</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🫁 Sistem Bantu Deteksi Kanker Paru-Paru</h1>
        <p>Klasifikasi citra CT paru-paru berbasis deep learning (EfficientNet-B0)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <b>Perhatian:</b> Sistem ini adalah alat bantu skrining hasil riset skripsi
        (bukan alat diagnostik medis yang tersertifikasi). Hasil prediksi
        <b>tidak boleh dijadikan dasar diagnosis atau keputusan klinis</b> tanpa
        konfirmasi dari dokter/radiolog yang berkompeten.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
try:
    predictor = load_predictor()
    model_load_error = None
except Exception as e:  # noqa: BLE001
    predictor = None
    model_load_error = str(e)

if model_load_error:
    st.error(f"Gagal memuat model: {model_load_error}")
    st.stop()

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("#### 📤 Unggah Citra CT Paru-Paru")
    uploaded_file = st.file_uploader(
        "Pilih file citra (PNG/JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Citra yang diunggah", use_container_width=True)
    else:
        st.info("Silakan unggah citra CT paru-paru (format PNG/JPG) untuk mulai analisis.")

with col_right:
    st.markdown("#### 🩺 Hasil Analisis")

    if uploaded_file is None:
        st.markdown(
            "<div style='color:#888; padding: 2rem; text-align:center;'>"
            "Hasil akan muncul di sini setelah citra diunggah.</div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Menganalisis citra..."):
            result = predictor.predict(image)

        if not result["is_valid_ct"]:
            st.markdown(
                f"""
                <div class="result-card rejected">
                    <h2>⚠️ Bukan Citra CT Paru-Paru</h2>
                    <p>Skor kemiripan: {result['ood_score']:.2f} (ambang batas: {result['ood_threshold']:.2f})</p>
                    <p>Sistem tidak dapat memberikan prediksi karena citra ini tidak
                    terdeteksi sebagai citra CT paru-paru. Pastikan file yang diunggah
                    benar dan coba lagi.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            pred_class = result["predicted_class"]
            cancer_prob = result["cancer_probability"] * 100
            no_cancer_prob = 100 - cancer_prob

            if pred_class == "cancer":
                st.markdown(
                    f"""
                    <div class="result-card cancer">
                        <h2>🔴 Terindikasi Cancer</h2>
                        <p>Tingkat keyakinan model: <b>{cancer_prob:.1f}%</b></p>
                        <p>Disarankan segera konsultasi dengan dokter spesialis paru
                        atau radiolog untuk pemeriksaan lanjutan.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card no_cancer">
                        <h2>🟢 Tidak Terindikasi Cancer</h2>
                        <p>Tingkat keyakinan model: <b>{no_cancer_prob:.1f}%</b></p>
                        <p>Hasil tidak menunjukkan indikasi kanker. Tetap disarankan
                        pemeriksaan rutin sesuai anjuran dokter.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("##### Distribusi Probabilitas")
            st.markdown(
                f"""
                <div>Cancer</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{cancer_prob:.1f}%; background-color:#e74c3c;">
                        {cancer_prob:.1f}%
                    </div>
                </div>
                <div style="margin-top:0.8rem;">No Cancer</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{no_cancer_prob:.1f}%; background-color:#27ae60;">
                        {no_cancer_prob:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Detail teknis"):
                st.json(result)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.8rem;'>"
    "Model: EfficientNet-B0 (ensemble v5+v6) · Dataset: Kaggle Lung Cancer CT Images + "
    "IQ-OTH/NCCD · Riset skripsi Teknik Informatika, FMIPA Unpad</div>",
    unsafe_allow_html=True,
)
