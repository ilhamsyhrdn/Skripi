"""Prototipe Streamlit untuk klasifikasi kanker paru-paru pada citra CT.

Aplikasi melayani konfigurasi final sebagaimana dilaporkan Bab IV: ensemble
stacking atas sepuluh model (5 EfficientNet-B0 + 5 ResNet50) dengan meta-learner
regresi logistik, didahului lapisan validasi input yang menolak unggahan di luar
domain. ACTIVE_VARIANT menentukan checkpoint mana yang dilayani, dan img_size
wajib sama dengan resolusi saat model dilatih.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import CLASS_NAMES
from inference.ood_predictor import OODDetector
from inference.stacking_predictor import StackingPredictor

OOD_MODEL_PATH = ROOT / "outputs" / "models" / "ood_detector.pt"
REPORT_DIR = ROOT / "outputs" / "reports"
MANIFEST_DIR = ROOT / "outputs" / "manifests"

# --- konfigurasi varian aktif -------------------------------------------------
# img_size WAJIB sama dengan resolusi saat model dilatih: memberi citra 224px ke
# model yang di-fine-tune pada 512px menghasilkan prediksi salah tanpa pesan error.
VARIANTS = {
    "full_augct": {
        "model_dir": ROOT / "outputs" / "models_full_augct",
        "img_size": 512,
        "suffix": "_full_augct",
        "pool": "combined_full_pool.csv",
        "judul": "Klasifikasi Kanker Paru-Paru pada Citra CT",
        "deskripsi": "EfficientNet-B0 + ResNet50, ensemble stacking 10 model, dilatih pada "
                     "gabungan 7 dataset Kaggle dan LIDC-IDRI (2.811 citra CT, irisan penuh 512px).",
    },
}
ACTIVE_VARIANT = "full_augct"
CFG = VARIANTS[ACTIVE_VARIANT]

MODEL_DIR = CFG["model_dir"]
IMG_SIZE = CFG["img_size"]
SUFFIX = CFG["suffix"]
META_PATH = REPORT_DIR / f"stacking_meta_learner{SUFFIX}.pkl"
AMBANG_BAKU = 0.50

st.set_page_config(page_title="Klasifikasi Kanker Paru-Paru pada Citra CT", layout="wide")


@st.cache_resource
def load_ood():
    return OODDetector(OOD_MODEL_PATH)


@st.cache_resource
def load_predictor():
    return StackingPredictor(MODEL_DIR, META_PATH, img_size=IMG_SIZE)


@st.cache_data
def metrik_final():
    """Metrik konfigurasi final dibaca dari berkas hasil, supaya angka di
    aplikasi selalu identik dengan angka yang dilaporkan pada Bab IV."""
    f = REPORT_DIR / f"stacking_threshold_sweep{SUFFIX}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d.loc[(d["threshold"] - AMBANG_BAKU).abs().idxmin()]


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
    st.caption(f"Model aktif: {MODEL_DIR.name} | ensemble stacking | "
               f"resolusi masukan {IMG_SIZE}x{IMG_SIZE} piksel | ambang baku {AMBANG_BAKU:.2f}")
    m = metrik_final()
    if m is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Akurasi", f"{m['accuracy']*100:.2f}%")
        c2.metric("Macro-F1", f"{m['macro_f1']:.4f}")
        c3.metric("Cancer Recall", f"{m['cancer_recall']*100:.2f}%")
        c4.metric("Presisi Malignant", f"{m['cancer_precision']*100:.2f}%")
    else:
        st.info("Hasil evaluasi belum tersedia -- jalankan src/evaluation/stacking_ensemble.py terlebih dahulu.")

elif page == "Prediksi Citra CT":
    st.title("Prediksi Citra CT Paru-Paru")

    ambang = st.slider(
        "Ambang keputusan kelas Malignant", min_value=0.15, max_value=0.70,
        value=AMBANG_BAKU, step=0.05,
        help="Menurunkan ambang menaikkan cancer recall tetapi menekan kelas minoritas. "
             "Rinciannya ada pada Bab IV, Bagian 4.5.3.")
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
        result = predictor.predict(image, threshold=ambang)
        kind, msg = CLASS_MESSAGES[result["predicted_class"]]
        getattr(st, kind)(msg)

        probs_df = pd.DataFrame({"Kelas": CLASS_NAMES, "Probabilitas": result["ensemble_probs"]})
        st.bar_chart(probs_df.set_index("Kelas"))

        with st.expander("Detail probabilitas per model dasar"):
            per_model = pd.DataFrame(result["per_model_probs"]).T
            per_model.columns = CLASS_NAMES
            st.dataframe(per_model.style.background_gradient(cmap="Blues", axis=1))

        with st.expander("Detail keluaran meta-learner per pasangan fold"):
            per_fold = pd.DataFrame(result["per_fold_probs"]).T
            per_fold.columns = CLASS_NAMES
            st.dataframe(per_fold.style.background_gradient(cmap="Greens", axis=1))

elif page == "Perbandingan Model":
    st.title("Perbandingan Model Tunggal vs Ensemble")
    single_path = REPORT_DIR / f"single_model_results{SUFFIX}.csv"
    ens_path = REPORT_DIR / f"ensemble_results{SUFFIX}.csv"
    sweep_path = REPORT_DIR / f"stacking_threshold_sweep{SUFFIX}.csv"
    if single_path.exists() and ens_path.exists():
        st.subheader("Model tunggal (10 fold)")
        st.dataframe(pd.read_csv(single_path))
        st.subheader("Konfigurasi ensemble")
        st.dataframe(pd.read_csv(ens_path))
    else:
        st.info("Jalankan src/evaluation/evaluate_models.py terlebih dahulu.")
    if sweep_path.exists():
        st.subheader("Pengaruh ambang keputusan pada ensemble stacking")
        st.dataframe(pd.read_csv(sweep_path))

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
    modality_path = MANIFEST_DIR / "lidc_series_modality.csv"
    pool_path = MANIFEST_DIR / CFG["pool"]
    if pool_path.exists():
        pool = pd.read_csv(pool_path)
        st.bar_chart(pool["canonical_label"].value_counts())
        st.write(f"Total citra kanonik: {len(pool)} dari {pool['split_group'].nunique()} grup")
    if modality_path.exists():
        st.subheader("Penyaringan jenis pemindaian LIDC-IDRI")
        st.write("Hanya seri ber-Modality CT yang dipakai; seri DX dan CR adalah foto "
                 "rontgen dada yang tersimpan dalam struktur folder yang sama.")
        mod = pd.read_csv(modality_path)
        st.dataframe(mod.groupby("modality").size().rename("jumlah seri").reset_index())
    if labels_path.exists():
        with st.expander("Detail hasil pelabelan otomatis (skor malignansi)"):
            st.dataframe(pd.read_csv(labels_path))

else:
    st.title("Tentang & Metodologi")
    st.markdown("""
    Penelitian ini mengklasifikasikan citra CT paru-paru ke dalam tiga kelas
    (Benign, Malignant, Normal) menggunakan transfer learning EfficientNet-B0
    (dengan ResNet50 sebagai arsitektur kedua untuk keperluan ensemble),
    fine-tuning dua fase, data augmentation, dan ensemble stacking dari
    5-fold cross-validation kedua arsitektur.

    Dataset yang digunakan adalah gabungan tujuh dataset publik Kaggle dan
    **LIDC-IDRI** (The Cancer Imaging Archive). Label LIDC-IDRI diturunkan dari
    skor keganasan nodul (1-5) yang diberikan radiolog pada anotasi XML resmi,
    lalu dikoreksi memakai diagnosis histopatologi resmi TCIA.

    Seri LIDC-IDRI disaring lebih dahulu berdasarkan tag `Modality` pada berkas
    DICOM, sehingga hanya citra Computed Tomography yang dipakai dan foto
    rontgen dada yang tersimpan berdampingan tidak ikut terbawa.
    """)
