"""Override series labels with the OFFICIAL TCIA pathology-confirmed diagnosis
data (tcia-diagnosis-data-2012-04-20.xls) wherever available, since a
biopsy/surgical-resection-confirmed diagnosis is stronger ground truth than
either the radiologist malignancy-score rule or the nearest-neighbour
relabeling used for ambiguous (score==3) nodules.

Source: The Cancer Imaging Archive, LIDC-IDRI collection page,
"Diagnosis Data" section:
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254
File: tcia-diagnosis-data-2012-04-20.xls (patient-level diagnosis, coded
0=unknown, 1=benign/non-malignant, 2=malignant primary lung cancer,
3=malignant metastatic; diagnosis method includes biopsy/surgical resection
for a subset of the 157 patients covered).

Only 157 of 1010 LIDC-IDRI patients have this data (the study's pathology
follow-up was not obtained for every patient), so this is applied as a
priority override on top of the existing radiologist-score-based labels and
nearest-neighbour relabeling -- not a full ground truth for the whole
dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DIAGNOSIS_XLS = Path("D:/skripsi/Dataset/tcia-diagnosis-data.xls")
POOL_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_canonical_pool_crop_relabeled.csv")
OUT_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_canonical_pool_final.csv")

PATIENT_DIAG_MAP = {1: "Benign", 2: "Malignant", 3: "Malignant"}  # 0 = unknown, skip


def main():
    diag_raw = pd.read_excel(DIAGNOSIS_XLS, sheet_name="Diagnosis Truth")
    # first two real columns: Patient ID, Patient-level diagnosis code
    patient_id_col = diag_raw.columns[0]
    patient_diag_col = diag_raw.columns[1]
    diag = diag_raw[[patient_id_col, patient_diag_col]].copy()
    diag.columns = ["patient_id", "diag_code"]
    diag["patient_id"] = diag["patient_id"].astype(str).str.strip()
    diag = diag.dropna(subset=["diag_code"])
    diag["diag_code"] = diag["diag_code"].astype(int)
    diag = diag[diag["diag_code"] != 0]  # 0 = unknown, no usable ground truth
    diag["pathology_label"] = diag["diag_code"].map(PATIENT_DIAG_MAP)
    patient_to_label = dict(zip(diag["patient_id"], diag["pathology_label"]))
    print(f"Patients with usable pathology-confirmed diagnosis: {len(patient_to_label)}", flush=True)

    pool = pd.read_csv(POOL_CSV)
    pool["split_group"] = pool["split_group"].astype(str)
    pool["pathology_confirmed"] = False
    pool["label_before_pathology_override"] = pool["canonical_label"]

    n_overridden = n_confirmed_same = n_no_match = 0
    for idx, row in pool.iterrows():
        pid = row["split_group"]
        if pid in patient_to_label:
            true_label = patient_to_label[pid]
            if row["canonical_label"] != true_label:
                n_overridden += 1
            else:
                n_confirmed_same += 1
            pool.at[idx, "canonical_label"] = true_label
            pool.at[idx, "pathology_confirmed"] = True
        else:
            n_no_match += 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUT_CSV, index=False)

    print(f"\nSeries overridden by pathology data (label changed): {n_overridden}", flush=True)
    print(f"Series confirmed identical by pathology data (label unchanged): {n_confirmed_same}", flush=True)
    print(f"Series with no pathology data available (unchanged): {n_no_match}", flush=True)

    ambiguous_mask = pool["relabeled_from_ambiguous"] == True  # noqa: E712
    ambiguous_with_pathology = pool[ambiguous_mask & pool["pathology_confirmed"]]
    print(f"\nOf the {ambiguous_mask.sum()} ambiguous (score==3) series relabeled by "
          f"nearest-neighbour, {len(ambiguous_with_pathology)} now have a STRONGER "
          f"pathology-confirmed label instead.", flush=True)

    print("\n=== Final label distribution (pathology-corrected) ===", flush=True)
    print(pool["canonical_label"].value_counts(), flush=True)
    print(f"\nSaved: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
