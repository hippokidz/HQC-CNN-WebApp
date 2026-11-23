"""
Adaptive Meat RGB Quantum Feature Extraction - FINAL

Pipeline:
448x448 -> CLAHE -> augmentation -> QFE 128x128
12 qubits, patch 2x2, output 64x64x6

Final adaptive weights (C1):
Cross-channel = (0.20, 0.40, 0.40)
Channel-level = (0.60, 0.40)

QFE dihitung 1x per image per mode.
Fold tidak mengulang QFE.
"""

import csv
import json
import os
import time
import threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
import numpy as np
import pandas as pd
import pennylane as qml
import psutil
from tqdm.auto import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    r"C:\Users\Predator Neo 16\Documents\Dokumen_Herman\Program\Coba\Dataset\Daging_ORI_Specimen_Independent_KFold"
)

AUG_META = ROOT / "augmentation_metadata.csv"
TEST_CSV = ROOT / "holdout_test.csv"
PRE_DIR = ROOT / "preprocessed_448x448"
QFE_DIR = ROOT / "qfe_features"

QFE_SIZE = (128, 128)
PATCH_SIZE = 2
N_QUBITS = 12
NUM_WORKERS = 8

MODES = [
    "none",
    "intra",
    "fixed_cross",
    "adaptive_cross",
    "intra_adaptive_cross"
]

BASE_FEATURE_VERSION = "pearson_v2_grouped_metadata"
C1_FEATURE_VERSION = "C1_pearson_v2_grouped_metadata"

DEV = qml.device("default.qubit", wires=N_QUBITS)


# ============================================================
# RGB STATISTICS
# ============================================================

def safe_pearson(a, b, eps=1e-8):
    a = a - np.mean(a)
    b = b - np.mean(b)
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return 0.0 if d < eps else float(np.clip(np.sum(a * b) / d, -1.0, 1.0))


def local_stats(phi):
    p = phi.reshape(3, 2, 2)
    r, g, b = p[0].ravel(), p[1].ravel(), p[2].ravel()

    return (
        np.mean(np.abs(r - g)),
        np.mean(np.abs(g - b)),
        np.mean(np.abs(b - r)),
        np.std(p),
        np.std(r),
        np.std(g),
        np.std(b),
        safe_pearson(r, g),
        safe_pearson(g, b),
        safe_pearson(b, r),
    )


# ============================================================
# ADAPTIVE ANGLES - FINAL C1
# ============================================================

def adaptive_angles(phi):
    rg_gap, gb_gap, br_gap, rgb_c, rc, gc, bc, rg, gb, br = local_stats(phi)

    rg_d = (1.0 - rg) / 2.0
    gb_d = (1.0 - gb) / 2.0
    br_d = (1.0 - br) / 2.0

    # C1: Gap 0.20 + Global Contrast 0.40 + Decorrelation 0.40
    a_rg = np.pi * np.clip(0.20 * rg_gap + 0.40 * rgb_c + 0.40 * rg_d, 0, 1)
    a_gb = np.pi * np.clip(0.20 * gb_gap + 0.40 * rgb_c + 0.40 * gb_d, 0, 1)
    a_br = np.pi * np.clip(0.20 * br_gap + 0.40 * rgb_c + 0.40 * br_d, 0, 1)

    # Channel Contrast 0.60 + Global Contrast 0.40
    a_r = np.pi * np.clip(0.60 * rc + 0.40 * rgb_c, 0, 1)
    a_g = np.pi * np.clip(0.60 * gc + 0.40 * rgb_c, 0, 1)
    a_b = np.pi * np.clip(0.60 * bc + 0.40 * rgb_c, 0, 1)

    return a_rg, a_gb, a_br, a_r, a_g, a_b


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

def encode(phi):
    for i in range(N_QUBITS):
        qml.RY(phi[i] * np.pi, wires=i)


def intra():
    for a, b, c, d in [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]:
        qml.CNOT(wires=[a, b])
        qml.CNOT(wires=[c, d])
        qml.CNOT(wires=[b, c])


def fixed_cross():
    for r, g, b in [(0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)]:
        qml.CNOT(wires=[r, g])
        qml.CNOT(wires=[g, b])
        qml.CNOT(wires=[b, r])


def adaptive_cross(phi):
    a_rg, a_gb, a_br, a_r, a_g, a_b = adaptive_angles(phi)

    for r, g, b in [(0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)]:
        qml.CRY(a_rg, wires=[r, g])
        qml.CRY(a_gb, wires=[g, b])
        qml.CRY(a_br, wires=[b, r])

    qml.CRZ(a_r, wires=[0, 1])
    qml.CRZ(a_g, wires=[4, 5])
    qml.CRZ(a_b, wires=[8, 9])


@qml.qnode(DEV)
def circuit(phi, mode="intra_adaptive_cross"):
    encode(phi)

    if mode in ("intra", "intra_adaptive_cross"):
        intra()

    if mode == "fixed_cross":
        fixed_cross()
    elif mode in ("adaptive_cross", "intra_adaptive_cross"):
        adaptive_cross(phi)
    elif mode not in ("none", "intra"):
        raise ValueError(f"Unknown mode: {mode}")

    return [
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(4)),
        qml.expval(qml.PauliZ(4) @ qml.PauliZ(8)),
        qml.expval(qml.PauliZ(8) @ qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(5)),
        qml.expval(qml.PauliZ(9)),
    ]


def qconv(image, mode):
    h, w, ch = image.shape
    if ch != 3 or h % PATCH_SIZE or w % PATCH_SIZE:
        raise ValueError("Image harus RGB dan dimensinya habis dibagi patch size.")

    out = np.zeros((h // PATCH_SIZE, w // PATCH_SIZE, 6))

    for r in range(0, h, PATCH_SIZE):
        for c in range(0, w, PATCH_SIZE):
            patch = image[r:r + 2, c:c + 2, :]
            phi = patch.transpose(2, 0, 1).flatten()
            out[r // PATCH_SIZE, c // PATCH_SIZE] = circuit(phi, mode)

    return out


# ============================================================
# MEMORY MONITOR
# ============================================================

class PeakRSS:
    def __init__(self, interval=0.05):
        self.p = psutil.Process(os.getpid())
        self.interval = interval
        self.stop = threading.Event()
        self.peak = 0
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, self.p.memory_info().rss)
            except Exception:
                pass
            self.stop.wait(self.interval)

    def start(self):
        self.peak = self.p.memory_info().rss
        self.t.start()
        return self

    def finish(self):
        self.stop.set()
        self.t.join(timeout=1)
        try:
            self.peak = max(self.peak, self.p.memory_info().rss)
        except Exception:
            pass
        return self.peak / 1024**2


# ============================================================
# NPZ
# ============================================================

def feature_version(mode):
    return C1_FEATURE_VERSION if mode == "intra_adaptive_cross" else BASE_FEATURE_VERSION


def valid_npz(path, label, mode):
    if not os.path.exists(path):
        return False

    try:
        with np.load(path, allow_pickle=False) as d:
            return (
                d["feature_map"].shape == (64, 64, 6)
                and int(d["label"]) == int(label)
                and str(d["mode"]) == mode
                and str(d["feature_version"]) == feature_version(mode)
            )
    except Exception:
        return False


def save_npz(path, fmap, rec, label, mode, split):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp = path + ".tmp.npz"

    np.savez_compressed(
        tmp,
        feature_map=fmap,
        label=label,
        mode=mode,
        feature_version=feature_version(mode),
        source_image=rec["image_path"],
        specimen_id=rec["specimen_id"],
        group_id=rec["group_id"],
        class_name=rec["class_name"],
        surface=rec["surface"],
        augmentation=rec["augmentation"],
        split=split,
        cv_fold=int(rec["cv_fold"]) if str(rec.get("cv_fold", "")) not in ("", "nan") else -1,
        backend="default.qubit",
    )

    os.replace(tmp, path)


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_one(item):
    rec, label, mode, split = item

    path = rec["image_path"]
    out = os.path.join(
        QFE_DIR,
        mode,
        rec["class_name"],
        os.path.basename(path) + f".{mode}.npz"
    )

    t0 = time.perf_counter()
    mem = PeakRSS().start()

    if valid_npz(out, label, mode):
        peak = mem.finish()
        return {
            "status": "skipped",
            "mode": mode,
            "split": split,
            "class_name": rec["class_name"],
            "source_image": path,
            "output_file": out,
            "specimen_id": rec["specimen_id"],
            "group_id": rec["group_id"],
            "surface": rec["surface"],
            "augmentation": rec["augmentation"],
            "cv_fold": rec.get("cv_fold", ""),
            "qfe_time_sec": 0.0,
            "total_image_time_sec": time.perf_counter() - t0,
            "peak_rss_mb": peak,
        }

    try:
        bgr = cv2.imread(path)
        if bgr is None:
            raise ValueError(f"Gagal membaca image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, QFE_SIZE, interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float64) / 255.0

        q0 = time.perf_counter()
        fmap = qconv(rgb, mode)
        qfe_time = time.perf_counter() - q0

        save_npz(out, fmap, rec, label, mode, split)
        peak = mem.finish()

        return {
            "status": "processed",
            "mode": mode,
            "split": split,
            "class_name": rec["class_name"],
            "source_image": path,
            "output_file": out,
            "specimen_id": rec["specimen_id"],
            "group_id": rec["group_id"],
            "surface": rec["surface"],
            "augmentation": rec["augmentation"],
            "cv_fold": rec.get("cv_fold", ""),
            "qfe_time_sec": qfe_time,
            "total_image_time_sec": time.perf_counter() - t0,
            "peak_rss_mb": peak,
        }

    except Exception as e:
        peak = mem.finish()

        return {
            "status": "failed",
            "mode": mode,
            "split": split,
            "class_name": rec.get("class_name", ""),
            "source_image": path,
            "output_file": out,
            "specimen_id": rec.get("specimen_id", ""),
            "group_id": rec.get("group_id", ""),
            "surface": rec.get("surface", ""),
            "augmentation": rec.get("augmentation", ""),
            "cv_fold": rec.get("cv_fold", ""),
            "qfe_time_sec": 0.0,
            "total_image_time_sec": time.perf_counter() - t0,
            "peak_rss_mb": peak,
            "error": str(e),
        }


# ============================================================
# DATA
# ============================================================

def load_records():
    dev = pd.read_csv(AUG_META)
    test = pd.read_csv(TEST_CSV)

    labels = {
        c: i for i, c in enumerate(
            sorted(dev["class_name"].unique())
        )
    }

    records = []

    # Development: metadata sudah menunjuk ke image hasil preprocessing + augmentation
    for _, r in dev.iterrows():
        records.append(
            (r.to_dict(), labels[r["class_name"]], "development")
        )

    # Test: gunakan preprocessed 448x448 CLAHE, tanpa augmentation
    for _, r in test.iterrows():
        raw = os.path.abspath(r["image_path"])
        stem = os.path.splitext(os.path.basename(raw))[0]
        pre = PRE_DIR / "test" / r["class_name"] / f"{stem}_448_CLAHE.png"

        if not pre.exists():
            raise FileNotFoundError(
                f"Preprocessed test tidak ditemukan: {pre}"
            )

        rec = r.to_dict()
        rec["image_path"] = str(pre)
        rec["augmentation"] = "ORIGINAL"
        rec["cv_fold"] = ""

        records.append(
            (rec, labels[r["class_name"]], "test")
        )

    return records, labels


def write_csv(path, rows):
    if not rows:
        return

    fields = sorted({k for r in rows for k in r})

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# RUN MODE
# ============================================================

def run_mode(mode, records):
    items = [
        (rec, label, mode, split)
        for rec, label, split in records
    ]

    mode_dir = QFE_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    results = []
    done = skip = fail = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = [
            ex.submit(process_one, item)
            for item in items
        ]

        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"QFE {mode}",
            unit="img"
        ):
            try:
                result = f.result()
                results.append(result)

                if result["status"] == "processed":
                    done += 1
                elif result["status"] == "skipped":
                    skip += 1
                else:
                    fail += 1

            except Exception as e:
                fail += 1
                results.append({
                    "status": "failed",
                    "mode": mode,
                    "error": str(e)
                })

    write_csv(
        mode_dir / "runtime_metrics.csv",
        results
    )

    return {
        "results": results,
        "processed": done,
        "skipped": skip,
        "failed": fail,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not AUG_META.exists():
        raise FileNotFoundError(AUG_META)

    if not TEST_CSV.exists():
        raise FileNotFoundError(TEST_CSV)

    records, labels = load_records()

    dev_count = sum(
        split == "development"
        for _, _, split in records
    )

    test_count = sum(
        split == "test"
        for _, _, split in records
    )

    print("Development augmented :", dev_count)
    print("Test                   :", test_count)
    print("Total per mode         :", len(records))
    print("Modes                  :", ", ".join(MODES))
    print("QFE input              :", QFE_SIZE)
    print("Patch                  :", PATCH_SIZE)
    print("Qubits                 :", N_QUBITS)
    print("Workers                :", NUM_WORKERS)
    print("Backend                : default.qubit (CPU)")
    print("Final config           : C1")
    print("Cross weights          : (0.20, 0.40, 0.40)")
    print("Channel weights        : (0.60, 0.40)")
    print("Rule                   : 1x image per mode; no fold-level QFE repeat")

    summaries = {}

    for mode in MODES:
        print("\n" + "=" * 70)
        print("MODE:", mode)
        print("=" * 70)

        result = run_mode(
            mode,
            records
        )

        rows = result["results"]
        processed_rows = [
            r for r in rows
            if r["status"] == "processed"
        ]

        qtimes = [
            r["qfe_time_sec"]
            for r in processed_rows
        ]

        summaries[mode] = {
            "processed": result["processed"],
            "skipped": result["skipped"],
            "failed": result["failed"],
            "qfe_mean_sec": float(np.mean(qtimes)) if qtimes else None,
            "qfe_total_sec": float(np.sum(qtimes)) if qtimes else None,
            "peak_rss_max_mb": max(
                (
                    r["peak_rss_mb"]
                    for r in rows
                    if "peak_rss_mb" in r
                ),
                default=0
            ),
        }

        print(
            f"processed={result['processed']} "
            f"skipped={result['skipped']} "
            f"failed={result['failed']}"
        )

    with open(
        QFE_DIR / "runtime_summary_all.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "labels": labels,
                "modes": MODES,
                "qfe_input_size": list(QFE_SIZE),
                "patch_size": PATCH_SIZE,
                "n_qubits": N_QUBITS,
                "backend": "default.qubit",
                "workers": NUM_WORKERS,
                "adaptive_configuration": {
                    "name": "C1",
                    "cross_weights": [0.20, 0.40, 0.40],
                    "channel_weights": [0.60, 0.40],
                },
                "processing_rule": (
                    "each unique image once per mode; "
                    "folds do not repeat QFE"
                ),
                "summary": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nSELESAI")
    print("Output:", QFE_DIR)


if __name__ == "__main__":
    main()