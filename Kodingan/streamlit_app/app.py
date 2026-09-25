"""Streamlit prototype for the lung cancer CT classifier: EfficientNet-B0 and
ResNet50 combined through the stacking ensemble, matching the final
configuration reported in Bab IV. ACTIVE_VARIANT below decides which trained
checkpoints are served, and img_size must match what they were trained at."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import CLASS_NAMES
from inference.ood_predictor import OODDetector
from inference.predictor import EnsemblePredictor

OOD_MODEL_PATH = ROOT / "outputs" / "models" / "ood_detector.pt"
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"
MANIFEST_DIR = ROOT / "outputs" / "manifests"

# --- konfigurasi varian aktif -------------------------------------------------
# Satu tempat untuk menentukan model mana yang dilayani aplikasi. img_size WAJIB
# sama dengan resolusi saat model dilatih: memberi citra 224px ke model yang
# di-fine-tune pada 512px menghasilkan prediksi yang salah tanpa pesan error.
VARIANTS = {
    "combined": {
        "model_dir": ROOT / "outputs" / "models_combined",
        "img_size": 224,
        "suffix": "_combined",
        "pool": "combined_canonical_pool.csv",
        "judul": "Klasifikasi Kanker Paru-Paru pada Citra CT",
        "deskripsi": "EfficientNet-B0 + ResNet50, ensemble 10 model, dilatih pada "
                     "gabungan 7 dataset Kaggle dan LIDC-IDRI (2.935 citra, crop nodul).",
    },
    "full": {
        "model_dir": ROOT / "outputs" / "models_full",
        "img_size": 512,
        "suffix": "_full",
        "pool": "combined_full_pool.csv",
        "judul": "Klasifikasi Kanker Paru-Paru pada Citra CT",
        "deskripsi": "EfficientNet-B0 + ResNet50, ensemble 10 model, dilatih pada "
                     "gabungan 7 dataset Kaggle dan LIDC-IDRI (3.101 citra, irisan penuh 512px).",
    },
}
ACTIVE_VARIANT = "full"
CFG = VARIANTS[ACTIVE_VARIANT]

MODEL_DIR = CFG["model_dir"]
IMG_SIZE = CFG["img_size"]
SUFFIX = CFG["suffix"]

st.set_page_config(page_title="Klasifikasi Kanker Paru-Paru pada Citra CT", layout="wide")


@st.cache_resource
def load_ood():
    return OODDetector(OOD_MODEL_PATH)


@st.cache_resource
def load_predictor():
    return EnsemblePredictor(MODEL_DIR, img_size=IMG_SIZE)


CLASS_MESSAGES = {
    "Malignant": ("error", "Terdeteksi kemungkinan **Malignant (ganas)**. Disarankan tindak lanjut medis segera."),
    "Benign": ("warning", "Terdeteksi kemungkinan **Benign (jinak)**. Tetap disarankan pemeriksaan lanjutan oleh radiolog."),
    "Normal": ("success", "Tidak terdeteksi tanda mencurigakan (**Normal**). Tetap lakukan skrining rutin sesuai anjuran dokter."),
}

PAGES = ["Beranda", "Prediksi Citra CT", "Perbandingan Model", "Kurva Training", "Audit Dataset", "Tentang & Metodologi"]
page = st.sidebar.radio("Navigasi", PAGES)

if page == "Beranda":
    st.title(CFG["judul"])
    st.write("Prototipe aplikasi hasil penelitian: " + CFG["deskripsi"])
    st.caption(f"Model aktif: {MODEL_DIR.name} | resolusi masukan {IMG_SIZE}x{IMG_SIZE} piksel")
    eval_path = REPORT_DIR / f"evaluation_summary{SUFFIX}.json"
    if eval_path.exists():
        summary = json.loads(eval_path.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Akurasi", f"{summary['final_metrics']['accuracy']*100:.1f}%")
        c2.metric("Macro-F1", f"{summary['final_metrics']['macro_f1']:.3f}")
        c3.metric("Cancer Recall", f"{summary['final_metrics']['cancer_recall']*100:.1f}%")
        c4.metric("ROC-AUC", f"{summary['final_metrics']['roc_auc']:.3f}")
    else:
        st.info("Hasil evaluasi belum tersedia -- jalankan src/evaluation/evaluate_models.py terlebih dahulu.")

elif page == "Prediksi Citra CT":
    st.title("Prediksi Citra CT Paru-Paru")
    config = st.radio("Konfigurasi ensemble", ["Full 10-model", "EfficientNet-B0 saja", "ResNet50 saja"], horizontal=True)
    uploaded = st.file_uploader("Unggah citra CT (jpg/jpeg/png/bmp)", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Pratinjau citra", width=300)

        # lapisan validasi input: citra yang bukan CT paru ditolak sebelum
        # model klasifikasi utama dijalankan sama sekali
        ood = load_ood()
        gate = ood.predict(image)
        if gate["available"] and not gate["is_lung_ct"]:
            st.error(f"Citra ditolak: terdeteksi BUKAN citra CT paru-paru "
                     f"(keyakinan CT paru hanya {gate['prob_lung_ct']*100:.1f}%).")
            st.stop()
        if gate["available"]:
            st.success(f"Citra tervalidasi sebagai CT paru-paru "
                       f"({gate['prob_lung_ct']*100:.1f}% keyakinan).")

        predictor = load_predictor()
        members = predictor.member_keys
        if config == "EfficientNet-B0 saja":
            members = [k for k in members if k.startswith("efficientnet_b0")]
        elif config == "ResNet50 saja":
            members = [k for k in members if k.startswith("resnet50")]

        result = predictor.predict(image, members=members)
        pred_class = result["predicted_class"]
        kind, msg = CLASS_MESSAGES[pred_class]
        getattr(st, kind)(msg)

        probs_df = pd.DataFrame({"Kelas": CLASS_NAMES, "Probabilitas": result["ensemble_probs"]})
        st.bar_chart(probs_df.set_index("Kelas"))

        with st.expander("Detail probabilitas per model"):
            per_model = pd.DataFrame(result["per_model_probs"]).T
            per_model.columns = CLASS_NAMES
            st.dataframe(per_model.style.background_gradient(cmap="Blues", axis=1))


elif page == "Perbandingan Model":
    st.title("Perbandingan Model Tunggal vs Ensemble")
    single_path = REPORT_DIR / f"single_model_results{SUFFIX}.csv"
    ens_path = REPORT_DIR / f"ensemble_results{SUFFIX}.csv"
    if single_path.exists() and ens_path.exists():
        st.subheader("Model tunggal (10 fold)")
        st.dataframe(pd.read_csv(single_path))
        st.subheader("Konfigurasi ensemble")
        st.dataframe(pd.read_csv(ens_path))
    else:
        st.info("Jalankan src/evaluation/evaluate_models.py terlebih dahulu.")

elif page == "Kurva Training":
    st.title("Kurva Training per Fold")
    arch = st.selectbox("Arsitektur", ["efficientnet_b0", "resnet50"])
    fold = st.slider("Fold", 0, 4, 0)
    hist_path = REPORT_DIR / f"history_{ACTIVE_VARIANT}_{arch}_fold{fold}.json"
    if hist_path.exists():
        hist = pd.read_json(hist_path)
        st.line_chart(hist.set_index("epoch")[["train_loss", "loss"]].rename(columns={"loss": "val_loss"}))
        st.line_chart(hist.set_index("epoch")[["train_acc", "acc", "macro_f1", "cancer_recall"]])
    else:
        st.info("Riwayat training belum tersedia untuk kombinasi ini.")

elif page == "Audit Dataset":
    st.title("Audit Dataset")
    labels_path = MANIFEST_DIR / "lidc_series_labels.csv"
    pool_path = MANIFEST_DIR / CFG["pool"]
    if pool_path.exists():
        pool = pd.read_csv(pool_path)
        st.bar_chart(pool["canonical_label"].value_counts())
        st.write(f"Total citra kanonik: {len(pool)} dari {pool['split_group'].nunique()} pasien")
    if labels_path.exists():
        with st.expander("Detail hasil pelabelan otomatis (skor malignansi)"):
            st.dataframe(pd.read_csv(labels_path))

else:
    st.title("Tentang & Metodologi")
    st.markdown("""
    Penelitian ini mengklasifikasikan citra CT paru-paru ke dalam tiga kelas
    (Benign, Malignant, Normal) menggunakan transfer learning EfficientNet-B0
    (dengan ResNet50 sebagai pembanding), fine-tuning dua fase, data augmentation,
    dan ensemble model dari 5-fold cross-validation kedua arsitektur.

    Dataset yang digunakan: **LIDC-IDRI** (The Cancer Imaging Archive), label
    kelas diturunkan dari skor keganasan nodul (1-5) yang diberikan radiolog
    pada anotasi XML resmi.
    """)
