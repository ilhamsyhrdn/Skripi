"""Eksperimen tambahan: persempit task ke malignant vs normal saja (exclude benign),
khusus di dataset IQ-OTH/NCCD -- menguji hipotesis bahwa sebagian paper yang klaim
94-98% akurasi memakai task yang lebih mudah daripada cancer vs no_cancer penuh.

PENTING -- pengaman leakage: dataset ini tidak punya ID pasien publik, dan kita sudah
buktikan sendiri (lihat insiden holdout v6/v7) bahwa banyak slice yang sangat mirip
(kemungkinan dari pasien yang sama) tersebar di file-file bernomor berurutan. Supaya
split train/val/test kali ini tidak mengulang kebocoran yang sama, dipakai clustering
kemiripan visual (cosine similarity pada citra grayscale 64x64) -- semua citra yang
mirip (>0.97 similarity) digabung jadi satu "grup pasien perkiraan", dan SATU GRUP
SELALU utuh masuk ke satu split saja (tidak pernah dipecah).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(r"D:\skripsi")
EXT_ROOT = (
    ROOT
    / "dataset_external"
    / "iqothnccd"
    / "The IQ-OTHNCCD lung cancer dataset"
    / "The IQ-OTHNCCD lung cancer dataset"
)
OUTPUT_ROOT = ROOT / "dataset_split_malignant_vs_normal"
REPORTS_DIR = ROOT / "reports"
RANDOM_SEED = 42
SIMILARITY_THRESHOLD = 0.97

FOLDER_TO_LABEL = {
    "Malignant cases": "malignant",
    "Normal cases": "normal",
    # "Bengin cases" sengaja dikecualikan
}


def main():
    records = []
    for folder, label in FOLDER_TO_LABEL.items():
        for p in sorted((EXT_ROOT / folder).glob("*.jpg")):
            im = Image.open(p).convert("L").resize((64, 64))
            arr = np.array(im, dtype=np.float32).flatten()
            phash = hashlib.md5(np.array(im).tobytes()).hexdigest()
            records.append({"path": str(p), "filename": p.name, "label": label, "phash": phash, "vec": arr})

    manifest = pd.DataFrame(records)
    print(f"Total malignant+normal (sebelum dedup): {len(manifest)}")

    # 1. Dedup exact-duplicate dulu (seperti sebelumnya)
    manifest = manifest.drop_duplicates(subset="phash", keep="first").reset_index(drop=True)
    print(f"Setelah exact-dedup: {len(manifest)}")

    # 2. Clustering kemiripan visual (near-duplicate / kemungkinan pasien sama)
    vecs = np.stack(manifest["vec"].to_numpy())
    norm_vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sims = norm_vecs @ norm_vecs.T

    n = len(manifest)
    cluster_id = -np.ones(n, dtype=int)
    next_cluster = 0
    for i in range(n):
        if cluster_id[i] != -1:
            continue
        similar_idx = np.where(sims[i] >= SIMILARITY_THRESHOLD)[0]
        assigned = cluster_id[similar_idx]
        existing = assigned[assigned != -1]
        if len(existing) > 0:
            cid = existing[0]
        else:
            cid = next_cluster
            next_cluster += 1
        cluster_id[similar_idx] = cid
        cluster_id[i] = cid

    manifest["cluster"] = cluster_id
    n_clusters = manifest["cluster"].nunique()
    print(f"Jumlah grup citra mirip (perkiraan 'pasien'): {n_clusters} dari {n} citra "
          f"(rata-rata {n / n_clusters:.1f} citra/grup)")

    # label mayoritas per cluster (dipakai untuk stratifikasi split)
    cluster_label = manifest.groupby("cluster")["label"].agg(lambda s: s.value_counts().idxmax())
    clusters = cluster_label.index.to_numpy()
    labels_for_split = cluster_label.to_numpy()

    # 3. Split di level CLUSTER, bukan di level citra
    train_c, temp_c = train_test_split(
        clusters, test_size=0.30, stratify=labels_for_split, random_state=RANDOM_SEED
    )
    temp_labels = cluster_label.loc[temp_c].to_numpy()
    val_c, test_c = train_test_split(
        temp_c, test_size=0.50, stratify=temp_labels, random_state=RANDOM_SEED
    )

    cluster_to_split = {}
    for c in train_c:
        cluster_to_split[c] = "train"
    for c in val_c:
        cluster_to_split[c] = "val"
    for c in test_c:
        cluster_to_split[c] = "test"
    manifest["split"] = manifest["cluster"].map(cluster_to_split)

    print(manifest.groupby(["split", "label"]).size().unstack(fill_value=0))

    # verifikasi: pastikan tidak ada cluster yang "bocor" ke >1 split
    leak_check = manifest.groupby("cluster")["split"].nunique()
    assert (leak_check == 1).all(), "ADA CLUSTER YANG BOCOR ANTAR SPLIT!"
    print("Verifikasi OK: setiap grup citra mirip utuh di satu split saja.")

    # 4. Salin file
    import shutil
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    for split in ["train", "val", "test"]:
        for label in ["malignant", "normal"]:
            (OUTPUT_ROOT / split / label).mkdir(parents=True, exist_ok=True)

    for row in manifest.itertuples(index=False):
        dest = OUTPUT_ROOT / row.split / row.label / row.filename
        shutil.copy2(row.path, dest)

    manifest.drop(columns=["vec"]).to_csv(
        REPORTS_DIR / "malignant_vs_normal_manifest.csv", index=False
    )
    print(f"\nSelesai. Data tersimpan di {OUTPUT_ROOT}")
    print(f"Manifest disimpan ke reports/malignant_vs_normal_manifest.csv")


if __name__ == "__main__":
    main()
