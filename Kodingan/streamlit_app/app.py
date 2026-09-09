"""
Streamlit demo untuk skripsi:
"Optimasi Transfer Learning EfficientNet-B0 untuk Klasifikasi Kanker Paru-Paru pada Citra CT
menggunakan Fine-Tuning, Data Augmentation, dan Ensemble Model"

Jalankan dari folder Kodingan:
    .venv/Scripts/streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.metrics import roc_curve

KODINGAN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KODINGAN_DIR))

from src.data.dataset import CLASS_NAMES  # noqa: E402
from src.inference.predictor import EnsemblePredictor  # noqa: E402

MANIFEST_DIR = KODINGAN_DIR / "outputs" / "manifests"
MODEL_DIR = KODINGAN_DIR / "outputs" / "models"
REPORT_DIR = KODINGAN_DIR / "outputs" / "reports"
DOCS_DIR = KODINGAN_DIR.parent / "Kejanggalan Dataset"

CLASS_COLORS = {"Benign": "#f2b134", "Malignant": "#e63946", "Normal": "#2a9d8f"}

st.set_page_config(page_title="Klasifikasi Kanker Paru-Paru CT", page_icon="🫁", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Memuat 10 model ensemble (EfficientNet-B0 + ResNet50, 5-fold)...")
def load_predictor() -> EnsemblePredictor:
    return EnsemblePredictor(MODEL_DIR)


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_npz(path: Path):
    if not path.exists():
        return None
    return dict(np.load(path))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


ensemble_eval = load_json(REPORT_DIR / "ensemble_evaluation.json")
audit_summary = load_json(MANIFEST_DIR / "audit_summary.json")
split_summary = load_json(MANIFEST_DIR / "split_summary.json")
full_manifest = load_csv(MANIFEST_DIR / "full_manifest.csv")
canonical_pool = load_csv(MANIFEST_DIR / "canonical_pool.csv")
probs_npz = load_npz(REPORT_DIR / "test_probs_labels.npz")


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🫁 Navigasi")
page = st.sidebar.radio(
    "Halaman",
    [
        "🏠 Beranda",
        "🔬 Prediksi Citra CT",
        "📊 Perbandingan Model",
        "📈 Kurva Training",
        "🕵️ Audit Kejanggalan Dataset",
        "ℹ️ Tentang & Metodologi",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Skripsi: Optimasi Transfer Learning EfficientNet-B0 untuk Klasifikasi Kanker "
    "Paru-Paru pada Citra CT menggunakan Fine-Tuning, Data Augmentation, dan Ensemble Model."
)


# --------------------------------------------------------------------------- #
# Page: Beranda
# --------------------------------------------------------------------------- #
def page_home():
    st.title("Klasifikasi Kanker Paru-Paru pada Citra CT")
    st.markdown(
        "Dashboard ini mendampingi skripsi yang mengoptimasi **transfer learning "
        "EfficientNet-B0** (dengan pembanding arsitektur ResNet50) melalui **fine-tuning "
        "bertahap**, **data augmentation**, dan **ensemble model** (5-fold cross-validation "
        "× 2 arsitektur = 10 model), dilatih pada gabungan 7 dataset CT paru-paru publik "
        "setelah melalui **audit deduplikasi & anti-kebocoran data** yang ketat."
    )

    if ensemble_eval and audit_summary:
        best_name = ensemble_eval["recommended_final_ensemble"]
        best = ensemble_eval["ensembles"][best_name]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Akurasi ensemble terbaik", f"{best['accuracy']*100:.1f}%")
        c2.metric("Cancer recall (Malignant)", f"{best['malignant_recall_aka_cancer_recall']*100:.1f}%")
        c3.metric("Macro-F1", f"{best['macro_f1']:.3f}")
        c4.metric("Macro ROC-AUC", f"{best['macro_roc_auc']:.3f}")
        st.caption(f"Model final yang direkomendasikan: **{best_name}** "
                   f"(diuji pada {sum(ensemble_eval['test_label_counts'].values())} citra held-out "
                   f"yang belum pernah dilihat model manapun selama training).")

        st.subheader("Dampak audit deduplikasi terhadap ukuran dataset")
        raw_total = audit_summary["total_files_found"]
        canon_total = audit_summary["canonical_pool_size"]
        fig = go.Figure(go.Bar(
            x=["Total file mentah\n(7 dataset Kaggle)", "Pool kanonik setelah\ndeduplikasi + anti-leakage"],
            y=[raw_total, canon_total],
            text=[f"{raw_total:,}", f"{canon_total:,}"],
            textposition="outside",
            marker_color=["#a8a8a8", "#2a9d8f"],
        ))
        fig.update_layout(yaxis_title="Jumlah citra", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"**{raw_total - canon_total:,} dari {raw_total:,} file ({(1-canon_total/raw_total)*100:.1f}%) "
            f"adalah duplikat/near-duplikat lintas dataset, augmentasi sintetis, atau data tanpa label** "
            f"yang dikeluarkan sebelum training untuk mencegah data leakage. Lihat halaman "
            f"'Audit Kejanggalan Dataset' dan folder `Kejanggalan Dataset/` untuk rincian lengkap."
        )
    else:
        st.warning("Hasil training/audit belum ditemukan. Jalankan pipeline di `Kodingan/src/` terlebih dahulu.")


# --------------------------------------------------------------------------- #
# Page: Prediksi
# --------------------------------------------------------------------------- #
def page_predict():
    st.title("🔬 Prediksi Citra CT Paru-Paru")
    st.markdown(
        "Unggah satu citra CT paru-paru (potongan aksial). Model akan memprediksi kelas "
        "**Normal / Benign / Malignant** menggunakan *soft-voting* ensemble."
    )

    ens_choice = st.radio(
        "Konfigurasi ensemble",
        ["Full 10-model (EfficientNet-B0 + ResNet50)", "EfficientNet-B0 saja (5-fold, cancer recall tertinggi)",
         "ResNet50 saja (5-fold)"],
        horizontal=True,
    )
    uploaded = st.file_uploader("Unggah citra (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is not None:
        image = Image.open(uploaded)
        predictor = load_predictor()
        if "Full 10-model" in ens_choice:
            members = predictor.member_keys
        elif "EfficientNet-B0" in ens_choice:
            members = [k for k in predictor.member_keys if k.startswith("efficientnet_b0")]
        else:
            members = [k for k in predictor.member_keys if k.startswith("resnet50")]

        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.image(image, caption="Citra yang diunggah", use_container_width=True)

        with st.spinner("Menjalankan inferensi ensemble..."):
            result = predictor.predict(image, members=members)

        with col2:
            pred = result["predicted_class"]
            st.markdown(f"### Prediksi: **:{'red' if pred=='Malignant' else 'orange' if pred=='Benign' else 'green'}[{pred}]**")
            probs = result["ensemble_probs"]
            fig = go.Figure(go.Bar(
                x=list(probs.values()), y=list(probs.keys()), orientation="h",
                marker_color=[CLASS_COLORS[c] for c in probs.keys()],
                text=[f"{v*100:.1f}%" for v in probs.values()], textposition="outside",
            ))
            fig.update_layout(xaxis_range=[0, 1], height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            if pred == "Malignant":
                st.error("⚠️ Model memprediksi kemungkinan **ganas (malignant)**. Ini bukan diagnosis "
                         "medis -- rujuk ke pemeriksaan patologi/radiologi lanjutan.")
            elif pred == "Benign":
                st.warning("Model memprediksi kemungkinan **jinak (benign)** -- nodul jinak tetap "
                           "memerlukan pemantauan lanjut menurut pedoman Fleischner.")
            else:
                st.success("Model memprediksi **normal** -- tidak ditemukan indikasi nodul.")

        with st.expander("Detail per model anggota ensemble"):
            per_model_df = pd.DataFrame(result["per_model_probs"]).T
            st.dataframe(per_model_df.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r", axis=1))

        st.subheader("Grad-CAM -- area yang paling memengaruhi prediksi")
        gc_col1, gc_col2 = st.columns(2)
        arch_for_cam = "efficientnet_b0" if members[0].startswith("efficientnet_b0") else "resnet50"
        fold_for_cam = st.slider("Pilih fold model untuk visualisasi Grad-CAM", 0, 4, 0)
        cam, cam_class, cam_probs = predictor.gradcam(image, arch=arch_for_cam, fold=fold_for_cam)
        img_resized = image.convert("RGB").resize((224, 224))
        img_arr = np.asarray(img_resized).astype(float) / 255.0

        overlay = 0.55 * img_arr + 0.45 * plt_cmap_jet(cam)[..., :3]
        with gc_col1:
            st.image(img_resized, caption="Citra input (224x224)", use_container_width=True)
        with gc_col2:
            st.image(np.clip(overlay, 0, 1), caption=f"Grad-CAM ({arch_for_cam} fold{fold_for_cam}, "
                                                      f"kelas: {cam_class})", use_container_width=True)


def plt_cmap_jet(arr: np.ndarray) -> np.ndarray:
    import matplotlib
    return matplotlib.colormaps["jet"](arr)


def apply_malignant_threshold(probs: np.ndarray, malignant_idx: int, threshold: float) -> np.ndarray:
    """Predict Malignant whenever its probability clears `threshold`; otherwise fall back to
    argmax among the remaining classes only. Used for the recall/accuracy trade-off slider."""
    is_malignant = probs[:, malignant_idx] >= threshold
    other_idxs = np.array([i for i in range(probs.shape[1]) if i != malignant_idx])
    other_argmax_local = probs[:, other_idxs].argmax(axis=1)
    other_argmax_global = other_idxs[other_argmax_local]
    preds = np.where(is_malignant, malignant_idx, other_argmax_global)
    return preds


# --------------------------------------------------------------------------- #
# Page: Perbandingan Model
# --------------------------------------------------------------------------- #
def page_compare():
    st.title("📊 Perbandingan Model")
    if not ensemble_eval:
        st.warning("Belum ada hasil evaluasi. Jalankan `src/evaluation/ensemble_eval.py`.")
        return

    st.markdown(
        "Seluruh angka pada halaman ini dihitung pada **held-out test set** (165 citra) yang "
        "**tidak pernah** dipakai untuk training/validasi model manapun -- lihat "
        "`Kejanggalan Dataset/07_strategi_split_dan_validasi.md`."
    )

    rows = []
    for key, m in ensemble_eval["individual_models"].items():
        arch, fold = key.rsplit("_fold", 1)
        rows.append({
            "Model": key, "Tipe": "Single model", "Arsitektur": arch, "Fold": fold,
            "Akurasi": m["accuracy"], "Macro-F1": m["macro_f1"],
            "Cancer Recall (Malignant)": m["malignant_recall_aka_cancer_recall"],
            "Benign Recall": m["benign_recall"], "Macro ROC-AUC": m["macro_roc_auc"],
        })
    for key, m in ensemble_eval["ensembles"].items():
        rows.append({
            "Model": key, "Tipe": "Ensemble", "Arsitektur": "campuran" if "full" in key else key.split("_")[0],
            "Fold": "semua",
            "Akurasi": m["accuracy"], "Macro-F1": m["macro_f1"],
            "Cancer Recall (Malignant)": m["malignant_recall_aka_cancer_recall"],
            "Benign Recall": m["benign_recall"], "Macro ROC-AUC": m["macro_roc_auc"],
        })
    df = pd.DataFrame(rows)

    st.subheader("Tabel metrik lengkap")
    st.dataframe(
        df.style.format({
            "Akurasi": "{:.3f}", "Macro-F1": "{:.3f}", "Cancer Recall (Malignant)": "{:.3f}",
            "Benign Recall": "{:.3f}", "Macro ROC-AUC": "{:.3f}",
        }).background_gradient(subset=["Akurasi", "Macro-F1", "Cancer Recall (Malignant)"], cmap="RdYlGn"),
        use_container_width=True,
        height=460,
    )

    st.subheader("Single model rata-rata vs Ensemble -- apakah ensembling benar-benar membantu?")
    avg = ensemble_eval["single_model_average"]
    best_name = ensemble_eval["recommended_final_ensemble"]
    best = ensemble_eval["ensembles"][best_name]
    comp_df = pd.DataFrame({
        "Metrik": ["Akurasi", "Macro-F1", "Cancer Recall"],
        "Rata-rata 10 model tunggal": [avg["accuracy"], avg["macro_f1"], avg["malignant_recall_aka_cancer_recall"]],
        f"Ensemble terbaik ({best_name})": [best["accuracy"], best["macro_f1"], best["malignant_recall_aka_cancer_recall"]],
    })
    fig = go.Figure()
    fig.add_bar(name="Rata-rata single model", x=comp_df["Metrik"], y=comp_df["Rata-rata 10 model tunggal"])
    fig.add_bar(name="Ensemble terbaik", x=comp_df["Metrik"], y=comp_df[f"Ensemble terbaik ({best_name})"])
    fig.update_layout(barmode="group", height=420, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Grafik metrik per model (diurutkan)")
    metric_choice = st.selectbox("Pilih metrik", ["Akurasi", "Macro-F1", "Cancer Recall (Malignant)", "Macro ROC-AUC"])
    df_sorted = df.sort_values(metric_choice, ascending=True)
    fig2 = px.bar(
        df_sorted, x=metric_choice, y="Model", color="Tipe", orientation="h",
        color_discrete_map={"Single model": "#a8a8a8", "Ensemble": "#2a9d8f"},
        height=560,
    )
    st.plotly_chart(fig2, use_container_width=True)

    if probs_npz is not None:
        st.subheader("Kurva ROC (one-vs-rest) -- Ensemble terbaik")
        labels = probs_npz["labels"]
        key_map = {
            "full_10model_ensemble_equal_weight": "ensemble_full_equal",
            "full_10model_ensemble_val_f1_weighted": "ensemble_full_weighted",
            "efficientnet_b0_5fold_ensemble": "ensemble_effnet",
            "resnet50_5fold_ensemble": "ensemble_resnet",
        }
        npz_key = key_map.get(best_name, "ensemble_full_equal")
        probs = probs_npz[npz_key]
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
        for i, cname in enumerate(CLASS_NAMES):
            y_true = (labels == i).astype(int)
            if len(set(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, probs[:, i])
            auc_val = ensemble_eval["ensembles"][best_name]["roc_auc_per_class"].get(cname)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cname} (AUC={auc_val:.3f})",
                                          line=dict(color=CLASS_COLORS[cname])))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=480)
        st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("Confusion Matrix -- Ensemble terbaik")
    cm = np.array(ensemble_eval["ensembles"][best_name]["confusion_matrix"])
    fig_cm = px.imshow(cm, text_auto=True, x=CLASS_NAMES, y=CLASS_NAMES,
                        labels=dict(x="Prediksi", y="Label sebenarnya", color="Jumlah"),
                        color_continuous_scale="Blues")
    st.plotly_chart(fig_cm, use_container_width=True)

    if probs_npz is not None:
        st.subheader("🎚️ Penyesuaian ambang keputusan kelas Malignant (trade-off recall vs akurasi)")
        st.markdown(
            "Secara default, prediksi diambil dari kelas dengan probabilitas tertinggi (argmax). "
            "Untuk skrining kanker, sering kali lebih aman menurunkan ambang khusus kelas "
            "**Malignant** supaya lebih sedikit kasus kanker yang terlewat (recall lebih tinggi), "
            "meski konsekuensinya lebih banyak *false alarm* (akurasi/presisi sedikit turun). "
            "Geser slider untuk melihat trade-off ini secara langsung."
        )
        key_map = {
            "full_10model_ensemble_equal_weight": "ensemble_full_equal",
            "full_10model_ensemble_val_f1_weighted": "ensemble_full_weighted",
            "efficientnet_b0_5fold_ensemble": "ensemble_effnet",
            "resnet50_5fold_ensemble": "ensemble_resnet",
        }
        probs_for_threshold = probs_npz[key_map.get(best_name, "ensemble_full_equal")]
        labels_for_threshold = probs_npz["labels"]
        malignant_idx = CLASS_NAMES.index("Malignant")

        threshold = st.slider(
            "Ambang probabilitas untuk memprediksi 'Malignant'", 0.05, 0.95, 0.5, 0.05,
            help="Default model = 0.5 (argmax biasa). Turunkan untuk memprioritaskan cancer recall.",
        )
        preds_thresh = apply_malignant_threshold(probs_for_threshold, malignant_idx, threshold)
        acc_t = float((preds_thresh == labels_for_threshold).mean())
        cancer_true = (labels_for_threshold == malignant_idx)
        cancer_recall_t = float((preds_thresh[cancer_true] == malignant_idx).mean()) if cancer_true.any() else float("nan")
        cancer_pred = (preds_thresh == malignant_idx)
        cancer_precision_t = float((labels_for_threshold[cancer_pred] == malignant_idx).mean()) if cancer_pred.any() else float("nan")

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Cancer recall pada ambang ini", f"{cancer_recall_t*100:.1f}%",
                   delta=f"{(cancer_recall_t - ensemble_eval['ensembles'][best_name]['malignant_recall_aka_cancer_recall'])*100:+.1f} pp vs default")
        tc2.metric("Cancer precision pada ambang ini", f"{cancer_precision_t*100:.1f}%")
        tc3.metric("Akurasi keseluruhan pada ambang ini", f"{acc_t*100:.1f}%",
                   delta=f"{(acc_t - ensemble_eval['ensembles'][best_name]['accuracy'])*100:+.1f} pp vs default")

        # sweep for a small reference table
        sweep_thresholds = np.arange(0.1, 0.95, 0.1)
        sweep_rows = []
        for th in sweep_thresholds:
            p = apply_malignant_threshold(probs_for_threshold, malignant_idx, th)
            acc_s = float((p == labels_for_threshold).mean())
            rec_s = float((p[cancer_true] == malignant_idx).mean()) if cancer_true.any() else float("nan")
            sweep_rows.append({"Ambang": round(th, 2), "Cancer Recall": rec_s, "Akurasi": acc_s})
        sweep_df = pd.DataFrame(sweep_rows)
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_df["Ambang"], y=sweep_df["Cancer Recall"], name="Cancer Recall", mode="lines+markers"))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["Ambang"], y=sweep_df["Akurasi"], name="Akurasi", mode="lines+markers"))
        fig_sweep.add_vline(x=threshold, line_dash="dash", annotation_text="ambang terpilih")
        fig_sweep.update_layout(xaxis_title="Ambang probabilitas Malignant", height=380, yaxis_range=[0, 1])
        st.plotly_chart(fig_sweep, use_container_width=True)


# --------------------------------------------------------------------------- #
# Page: Training curves
# --------------------------------------------------------------------------- #
def page_curves():
    st.title("📈 Kurva Training (per fold, per fase)")
    arch = st.selectbox("Arsitektur", ["efficientnet_b0", "resnet50"])
    fold = st.selectbox("Fold", [0, 1, 2, 3, 4])
    hist = load_json(REPORT_DIR / f"{arch}_fold{fold}_history.json")
    if not hist:
        st.warning("History tidak ditemukan untuk kombinasi ini.")
        return
    hdf = pd.DataFrame(hist["history"])
    hdf["global_epoch"] = range(1, len(hdf) + 1)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hdf["global_epoch"], y=hdf["train_loss"], name="Train loss"))
        fig.add_trace(go.Scatter(x=hdf["global_epoch"], y=hdf["val_loss"], name="Val loss"))
        phase_change = hdf[hdf["phase"] == "B-finetune"]["global_epoch"].min()
        if pd.notna(phase_change):
            fig.add_vline(x=phase_change - 0.5, line_dash="dot", annotation_text="mulai fine-tuning")
        fig.update_layout(title="Loss", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hdf["global_epoch"], y=hdf["train_acc"], name="Train acc"))
        fig.add_trace(go.Scatter(x=hdf["global_epoch"], y=hdf["val_acc"], name="Val acc"))
        fig.add_trace(go.Scatter(x=hdf["global_epoch"], y=hdf["val_macro_f1"], name="Val macro-F1"))
        fig.add_trace(go.Scatter(x=hdf["global_epoch"], y=hdf["val_malignant_recall"], name="Val cancer recall"))
        if pd.notna(phase_change):
            fig.add_vline(x=phase_change - 0.5, line_dash="dot")
        fig.update_layout(title="Accuracy / F1 / Recall", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.metric("Best val macro-F1 (checkpoint disimpan)", f"{hist['best_val_macro_f1']:.4f}",
               help=f"Dari epoch {hist['best_epoch']}")
    st.json(hist["test_readout"], expanded=False)


# --------------------------------------------------------------------------- #
# Page: Audit
# --------------------------------------------------------------------------- #
def page_audit():
    st.title("🕵️ Audit Kejanggalan Dataset")
    st.markdown(
        "Ringkasan interaktif dari dokumen lengkap di folder `Kejanggalan Dataset/`. "
        "Semua angka dihitung otomatis oleh `Kodingan/src/audit/build_manifest.py`."
    )
    if not audit_summary or full_manifest is None:
        st.warning("Jalankan `src/audit/build_manifest.py` terlebih dahulu.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total file mentah", f"{audit_summary['total_files_found']:,}")
    c2.metric("Grup duplikat (>1 anggota)", f"{audit_summary['groups_with_gt1_member']:,}")
    c3.metric("Dataset sintetis dikecualikan", f"{audit_summary['synthetic_source_images_excluded']:,}")
    c4.metric("Pool kanonik final", f"{audit_summary['canonical_pool_size']:,}")

    st.subheader("Jumlah gambar mentah per dataset & kelas (klaim asli, sebelum audit)")
    raw_counts = pd.DataFrame(audit_summary["raw_label_counts_by_source"]).T.fillna(0)
    fig = px.bar(raw_counts, barmode="stack", height=450,
                 labels={"value": "Jumlah gambar", "index": "Dataset", "variable": "Label"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribusi kelas: sebelum vs sesudah deduplikasi")
    before = full_manifest[full_manifest["is_labeled"] & (~full_manifest["is_synthetic_source"])]["canonical_label"].value_counts()
    after = canonical_pool["canonical_label"].value_counts() if canonical_pool is not None else pd.Series(dtype=int)
    comp = pd.DataFrame({"Sebelum dedup (raw, non-sintetis)": before, "Sesudah dedup (pool kanonik)": after}).fillna(0)
    fig2 = go.Figure()
    fig2.add_bar(name="Sebelum dedup", x=comp.index, y=comp["Sebelum dedup (raw, non-sintetis)"])
    fig2.add_bar(name="Sesudah dedup", x=comp.index, y=comp["Sesudah dedup (pool kanonik)"])
    fig2.update_layout(barmode="group", height=420)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Distribusi ukuran grup duplikat")
    dup_groups = load_csv(MANIFEST_DIR / "duplicate_groups.csv")
    if dup_groups is not None:
        sizes = dup_groups.groupby("group_id")["group_size"].first()
        fig3 = px.histogram(sizes, nbins=40, labels={"value": "Ukuran grup (jumlah salinan)"}, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    naive = load_json(REPORT_DIR / "naive_leaky_baseline.json")
    if naive and ensemble_eval:
        st.subheader("🔥 Bukti kuantitatif: seberapa besar leakage menggelembungkan akurasi")
        st.markdown(
            "Eksperimen pembanding yang **sengaja mereplikasi kesalahan umum** -- gabungkan semua "
            "data lalu split acak per gambar tanpa deduplikasi -- dilatih dengan model & jadwal "
            "training yang identik dengan pipeline final. Lihat "
            "`Kejanggalan Dataset/08_bukti_kuantitatif_dampak_leakage.md`."
        )
        leak_c1, leak_c2 = st.columns(2)
        leak_c1.metric(
            "Gambar 'test' naif yang grup duplikatnya juga ada di 'train' naif",
            f"{naive['fraction_test_images_duplicate_group_also_in_train']*100:.1f}%",
        )
        best_name = ensemble_eval["recommended_final_ensemble"]
        best = ensemble_eval["ensembles"][best_name]
        comp = pd.DataFrame({
            "Metrik": ["Akurasi", "Macro-F1", "Cancer Recall"],
            "Baseline naif (bocor)": [naive["test_accuracy"], naive["test_macro_f1"], naive["test_malignant_recall"]],
            "Pipeline final (bersih, ensemble)": [best["accuracy"], best["macro_f1"], best["malignant_recall_aka_cancer_recall"]],
        })
        fig_leak = go.Figure()
        fig_leak.add_bar(name="Baseline naif (bocor)", x=comp["Metrik"], y=comp["Baseline naif (bocor)"], marker_color="#e63946")
        fig_leak.add_bar(name="Pipeline final (bersih)", x=comp["Metrik"], y=comp["Pipeline final (bersih, ensemble)"], marker_color="#2a9d8f")
        fig_leak.update_layout(barmode="group", height=420, yaxis_range=[0, 1])
        st.plotly_chart(fig_leak, use_container_width=True)
        st.warning(
            f"Baseline naif melebih-lebihkan akurasi sebesar "
            f"**{(naive['test_accuracy']-best['accuracy'])*100:+.1f} poin persentase** dan macro-F1 "
            f"sebesar **{naive['test_macro_f1']-best['macro_f1']:+.3f}** dibanding hasil jujur pada "
            f"pipeline final -- murni karena kebocoran data, bukan karena model naif benar-benar "
            f"lebih baik."
        )

    st.subheader("Bukti kunci (angka mentah dari audit_summary.json)")
    st.json(audit_summary, expanded=False)

    st.subheader("Dokumen lengkap")
    if DOCS_DIR.exists():
        for md_file in sorted(DOCS_DIR.glob("*.md")):
            with st.expander(md_file.name):
                st.markdown(md_file.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Page: Tentang
# --------------------------------------------------------------------------- #
def page_about():
    st.title("ℹ️ Tentang & Metodologi")
    st.markdown(
        """
### Judul Skripsi
**Optimasi Transfer Learning EfficientNet-B0 untuk Klasifikasi Kanker Paru-Paru pada Citra
Computed Tomography (CT) Menggunakan Fine-Tuning, Data Augmentation, dan Ensemble Model**

### Ringkasan pipeline
1. **Audit & deduplikasi dataset** (`src/audit/build_manifest.py`) -- fingerprinting MD5 +
   perceptual hash seluruh 12.882 citra dari 7 dataset Kaggle, deteksi duplikat lintas
   dataset & dalam dataset, pengecualian dataset augmentasi sintetis dan data tanpa label.
2. **Split leakage-safe** (`src/audit/make_splits.py`) -- held-out test 18% + 5-fold
   `StratifiedGroupKFold` pada sisa data, dikelompokkan per grup duplikat & per rumpun nomor
   kasus supaya tidak ada kebocoran pasien/slice antar split.
3. **Transfer learning + fine-tuning 2 fase** (`src/training/train_cv.py`) -- Fase A: bekukan
   backbone ImageNet, latih hanya head baru. Fase B: buka blok terakhir backbone, latih ulang
   dengan learning rate kecil. Early stopping & checkpoint berdasarkan validation macro-F1.
4. **Data augmentation** hanya pada sisi train (`src/data/dataset.py`) -- rotasi, flip
   horizontal, translasi/scale, brightness/contrast, sesuai tinjauan pustaka Chlap et al.
   (2021) dan proposal bab 2.5/3.2.
5. **Ensemble model** (`src/evaluation/ensemble_eval.py`) -- soft-voting probabilitas dari
   5 fold EfficientNet-B0 + 5 fold ResNet50 (10 model), dibandingkan dengan rata-rata performa
   model tunggal.

### Referensi utama
- Tan & Le (2019) -- EfficientNet: Rethinking Model Scaling for CNNs.
- Chlap et al. (2021) -- A review of medical image data augmentation techniques.
- Kim et al. (2022) -- Transfer learning for medical image classification: a literature review.
- Shi et al. (2022) -- Semi-supervised deep transfer learning for benign-malignant diagnosis
  of pulmonary nodules in chest CT images (metodologi 5-fold CV + ensemble).
- Saha et al. (2024) -- VER-Net: hybrid transfer learning untuk deteksi kanker paru.
- Sandag & Kabo (2024) -- Perbandingan EfficientNet vs ResNet pada citra CT kanker paru.
- Wang et al. (2022) -- Novel deep learning model membedakan nodul paru jinak vs ganas.

Detail lengkap kejanggalan dataset yang ditemukan & cara penanganannya ada di folder
`D:\\skripsi\\Kejanggalan Dataset\\` (juga bisa dibaca di halaman "Audit Kejanggalan Dataset").
        """
    )


PAGES = {
    "🏠 Beranda": page_home,
    "🔬 Prediksi Citra CT": page_predict,
    "📊 Perbandingan Model": page_compare,
    "📈 Kurva Training": page_curves,
    "🕵️ Audit Kejanggalan Dataset": page_audit,
    "ℹ️ Tentang & Metodologi": page_about,
}
PAGES[page]()
