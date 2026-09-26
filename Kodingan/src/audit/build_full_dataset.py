"""Build the whole-slice (uncropped) combined dataset and its splits.

Sources:
  1. Kaggle canonical pool           -- folder labels, audited for duplicates
  2. Kaggle pseudo-labelled images   -- labels inferred by k-NN, TRAIN ONLY
  3. LIDC-IDRI whole-slice pool      -- final labels, uncropped

Pseudo-labelled images are forced into the training folds and never into the
held-out test set: measuring accuracy against guessed labels would be
circular. The test set therefore contains only images whose label came from a
folder annotation or from radiologist/pathology data.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

MANIFESTS = Path("D:/skripsi/Kodingan/outputs/manifests")
KAGGLE = MANIFESTS / "kaggle_canonical_pool.csv"
PSEUDO = MANIFESTS / "kaggle_pseudolabeled.csv"
LIDC_FULL = MANIFESTS / "lidc_canonical_pool_full.csv"
MODALITY = MANIFESTS / "lidc_series_modality.csv"

OUT_POOL = MANIFESTS / "combined_full_pool.csv"
OUT_TEST = MANIFESTS / "test_holdout_full.csv"
OUT_TRAINVAL = MANIFESTS / "trainval_folds_full.csv"

N_FOLDS = 5
TEST_FRACTION = 0.18
SEED = 42


def main():
    kaggle = pd.read_csv(KAGGLE)
    pseudo = pd.read_csv(PSEUDO)
    lidc = pd.read_csv(LIDC_FULL)

    # LIDC-IDRI menyimpan foto rontgen dada (Modality DX/CR) berdampingan dengan
    # CT dalam struktur folder yang sama. Nilai pikselnya bukan Hounsfield Unit,
    # sehingga windowing HU menjenuhkannya jadi putih polos. Seri semacam itu
    # dibuang di sini agar pool benar-benar hanya berisi Computed Tomography.
    mod = pd.read_csv(MODALITY, dtype={"uid": str})
    peta = dict(zip(mod["uid"], mod["modality"]))
    lidc["series_uid_file"] = [Path(x).stem for x in lidc["path"]]
    lidc["modality"] = lidc["series_uid_file"].map(peta)
    sebelum = len(lidc)
    lidc = lidc[lidc["modality"] == "CT"].drop(columns=["series_uid_file", "modality"]).reset_index(drop=True)
    print(f"Saring modality: {sebelum} -> {len(lidc)} seri "
          f"({sebelum - len(lidc)} seri bukan CT dibuang)", flush=True)
    assert len(lidc) > 0, "tidak ada seri CT tersisa; jalankan scan_lidc_modality.py dulu"

    kaggle_rows = pd.DataFrame({
        "path": kaggle["path"],
        "canonical_label": kaggle["canonical_label"],
        "split_group": "KAGGLE::" + kaggle["case_key"].astype(str),
        "source_dataset": "Kaggle::" + kaggle["source_dataset"],
        "label_origin": "folder",
        "md5": kaggle["md5"],
        "phash": kaggle["phash"],
    })

    pseudo = pseudo[pseudo["diterima"]].copy()
    pseudo_rows = pd.DataFrame({
        "path": pseudo["path"],
        "canonical_label": pseudo["pseudo_label"],
        "split_group": "PSEUDO::" + pseudo["md5"].astype(str),
        "source_dataset": "Kaggle-pseudolabel::" + pseudo["source_dataset"],
        "label_origin": "pseudo-knn",
        "md5": pseudo["md5"],
        "phash": None,
    })

    lidc_rows = pd.DataFrame({
        "path": lidc["path"],
        "canonical_label": lidc["canonical_label"],
        "split_group": "LIDC::" + lidc["split_group"].astype(str),
        "source_dataset": "LIDC-IDRI",
        "label_origin": "radiolog+patologi",
        "md5": None,
        "phash": None,
    })

    # cross-source duplicate check (LIDC whole-slice vs Kaggle)
    print("Memeriksa duplikasi lintas sumber...", flush=True)
    kaggle_md5 = set(kaggle_rows["md5"].dropna())
    kaggle_ph = set(kaggle_rows["phash"].dropna())
    n_dupe = 0
    for p in lidc_rows["path"]:
        with Image.open(p) as im:
            ph = str(imagehash.phash(im.convert("RGB"), hash_size=8))
        md5 = hashlib.md5(Path(p).read_bytes()).hexdigest()
        if md5 in kaggle_md5 or ph in kaggle_ph:
            n_dupe += 1
    print(f"Duplikat lintas sumber: {n_dupe} (diharapkan 0)", flush=True)

    pool = pd.concat([kaggle_rows, pseudo_rows, lidc_rows], ignore_index=True)
    pool = pool.drop(columns=["md5", "phash"])
    pool.to_csv(OUT_POOL, index=False)

    print("\n=== Pool gabungan (citra penuh) ===", flush=True)
    print(pd.crosstab(pool["canonical_label"], pool["label_origin"]).to_string(), flush=True)
    print(f"Total: {len(pool)} citra | {pool['split_group'].nunique()} grup", flush=True)

    # --- split: pseudo-labelled rows are held out of the test draw ---
    eligible = pool[pool["label_origin"] != "pseudo-knn"].reset_index(drop=True)
    forced_train = pool[pool["label_origin"] == "pseudo-knn"].reset_index(drop=True)

    n_test_folds = max(1, round(1 / TEST_FRACTION))
    sgkf = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(eligible, eligible["canonical_label"],
                                          groups=eligible["split_group"]))
    test_df = eligible.iloc[test_idx].reset_index(drop=True)
    trainval = eligible.iloc[train_idx].reset_index(drop=True)
    trainval = pd.concat([trainval, forced_train], ignore_index=True)

    sgkf2 = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    trainval["fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf2.split(trainval, trainval["canonical_label"],
                                                    groups=trainval["split_group"])):
        trainval.loc[trainval.index[val_idx], "fold"] = fold

    # anti-leakage checks
    assert (trainval["fold"] >= 0).all(), "ada baris tanpa fold"
    for group, sub in trainval.groupby("split_group"):
        assert sub["fold"].nunique() == 1, f"grup {group} tersebar di beberapa fold"
    overlap = set(trainval["split_group"]) & set(test_df["split_group"])
    assert not overlap, f"grup bocor ke test: {list(overlap)[:3]}"
    assert (test_df["label_origin"] != "pseudo-knn").all(), "label tebakan bocor ke data uji"

    test_df.to_csv(OUT_TEST, index=False)
    trainval.to_csv(OUT_TRAINVAL, index=False)

    print("\n=== Pembagian data ===", flush=True)
    print(f"Data uji  : {len(test_df)} citra")
    print(test_df["canonical_label"].value_counts().to_string(), flush=True)
    print(f"\nTrainval  : {len(trainval)} citra (termasuk {len(forced_train)} berlabel tebakan)")
    print(trainval["canonical_label"].value_counts().to_string(), flush=True)
    print("\nJumlah per fold:", flush=True)
    print(trainval["fold"].value_counts().sort_index().to_string(), flush=True)
    print("\nSemua pemeriksaan kebocoran data lolos.", flush=True)


if __name__ == "__main__":
    main()
