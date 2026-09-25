"""Derive per-series Benign/Malignant/Normal labels from LIDC-IDRI radiologist
XML annotations (malignancy score 1-5 per nodule per radiologist).

Method (documented for Bab III):
  1. Only nodules with <characteristics> (i.e. the >=3mm, fully characterized
     nodules) carry a malignancy score; smaller marks / non-nodules are ignored.
  2. Nodule marks from different readingSessions (radiologists) in the same
     series are grouped into one "consensus nodule" if their 3D centroids are
     close (same slice neighbourhood + nearby xy) -- different radiologists
     drawing the same physical nodule.
  3. Each consensus nodule's malignancy = mean of its radiologists' scores.
  4. Series label = worst (most severe) consensus nodule decides the case:
       any consensus nodule mean > 3      -> Malignant
       else any consensus nodule mean < 3 -> Benign
       else (only score==3 clusters)      -> excluded (ambiguous)
     Series with zero characterized nodules -> Normal (no suspicious finding
     flagged by any of the 4 radiologists).

Output: outputs/manifests/lidc_series_labels.csv with one row per series:
  series_uid, study_uid, n_consensus_nodules, worst_mean_malignancy,
  canonical_label, best_slice_sop_uid, best_slice_z
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean

import pandas as pd

NS = {"n": "http://www.nih.gov"}
XML_ROOT = Path("D:/skripsi/Dataset/lidc_annotations")
OUT_CSV = Path("D:/skripsi/Kodingan/outputs/manifests/lidc_series_labels.csv")

Z_TOL = 5.0       # mm, same-slice-neighbourhood tolerance across radiologists
XY_TOL_PX = 40.0  # pixels, centroid distance tolerance


def centroid_of_roi(roi) -> tuple[float, float, float, str]:
    xs, ys = [], []
    for em in roi.findall("n:edgeMap", NS):
        xs.append(float(em.findtext("n:xCoord", default="0", namespaces=NS)))
        ys.append(float(em.findtext("n:yCoord", default="0", namespaces=NS)))
    z = float(roi.findtext("n:imageZposition", default="0", namespaces=NS))
    sop = roi.findtext("n:imageSOP_UID", default="", namespaces=NS)
    if not xs:
        return (0.0, 0.0, z, sop)
    return (mean(xs), mean(ys), z, sop)


def parse_one_xml(path: Path) -> list[dict]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []

    header = root.find("n:ResponseHeader", NS)
    if header is None:
        return []
    series_uid = header.findtext("n:SeriesInstanceUid", default="", namespaces=NS)
    study_uid = header.findtext("n:StudyInstanceUID", default="", namespaces=NS)
    if not series_uid:
        return []

    marks = []
    for session in root.findall("n:readingSession", NS):
        for nodule in session.findall("n:unblindedReadNodule", NS):
            chars = nodule.find("n:characteristics", NS)
            if chars is None:
                continue
            malig_txt = chars.findtext("n:malignancy", default="", namespaces=NS)
            if not malig_txt:
                continue
            malignancy = int(malig_txt)
            rois = nodule.findall("n:roi", NS)
            if not rois:
                continue
            cxs, cys, czs, sops = [], [], [], []
            for roi in rois:
                cx, cy, z, sop = centroid_of_roi(roi)
                cxs.append(cx); cys.append(cy); czs.append(z); sops.append(sop)
            mid = len(rois) // 2
            marks.append({
                "series_uid": series_uid,
                "study_uid": study_uid,
                "malignancy": malignancy,
                "cx": mean(cxs), "cy": mean(cys), "cz": mean(czs),
                "best_sop": sops[mid], "best_z": czs[mid],
            })
    return marks


def cluster_marks(marks: list[dict]) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for m in marks:
        placed = False
        for cl in clusters:
            rep = cl[0]
            if abs(rep["cz"] - m["cz"]) <= Z_TOL and \
               ((rep["cx"] - m["cx"]) ** 2 + (rep["cy"] - m["cy"]) ** 2) ** 0.5 <= XY_TOL_PX:
                cl.append(m)
                placed = True
                break
        if not placed:
            clusters.append([m])
    return clusters


def main():
    all_marks: dict[str, list[dict]] = {}
    study_of: dict[str, str] = {}
    xml_files = list(XML_ROOT.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files", flush=True)

    for i, xp in enumerate(xml_files, 1):
        for m in parse_one_xml(xp):
            all_marks.setdefault(m["series_uid"], []).append(m)
            study_of[m["series_uid"]] = m["study_uid"]
        if i % 200 == 0:
            print(f"  parsed {i}/{len(xml_files)} xml files", flush=True)

    print(f"Series with >=1 characterized nodule mark: {len(all_marks)}", flush=True)

    rows = []
    for series_uid, marks in all_marks.items():
        clusters = cluster_marks(marks)
        cluster_means = [mean(c["malignancy"] for c in cl) for cl in clusters]
        worst_idx = max(range(len(clusters)), key=lambda i: cluster_means[i])
        worst_mean = cluster_means[worst_idx]
        worst_cluster = clusters[worst_idx]
        rep_mark = worst_cluster[len(worst_cluster) // 2]

        if worst_mean > 3:
            label = "Malignant"
        elif worst_mean < 3:
            label = "Benign"
        else:
            label = "Excluded"

        rows.append({
            "series_uid": series_uid,
            "study_uid": study_of[series_uid],
            "n_consensus_nodules": len(clusters),
            "worst_mean_malignancy": round(worst_mean, 3),
            "canonical_label": label,
            "best_slice_sop_uid": rep_mark["best_sop"],
            "best_slice_z": rep_mark["best_z"],
            "nodule_cx": rep_mark["cx"],
            "nodule_cy": rep_mark["cy"],
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n=== Label distribution (series with >=1 characterized nodule) ===", flush=True)
    print(df["canonical_label"].value_counts(), flush=True)
    print(f"\nSaved: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
