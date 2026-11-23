from pathlib import Path
import hashlib
import importlib.util
import sys

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "intra_adaptive_cross_final_model.keras"
QFE_PATH = BASE_DIR / "adaptive_meat_rgb_qconv_Versi2_QFE_5MODE_UNIQUE.py"
MODE = "intra_adaptive_cross"
CLASS_NAMES = ["Beef", "Horse_Meat", "Pork"]
QFE_SHAPE = (64, 64, 6)
RESIZE_SIZE = (448, 448)
QFE_SIZE = (128, 128)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

st.set_page_config(page_title="HQC-CNN Meat Classifier", page_icon="🥩", layout="centered")

@st.cache_resource(show_spinner="Memuat model CNN...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH.name}")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    if tuple(model.input_shape[1:]) != QFE_SHAPE:
        raise ValueError(f"Input model adalah {model.input_shape}; seharusnya (None, 64, 64, 6).")
    if int(model.output_shape[-1]) != len(CLASS_NAMES):
        raise ValueError(f"Output model tidak sesuai: {model.output_shape}")
    return model

@st.cache_resource(show_spinner="Memuat rangkaian Quantum Feature Extraction...")
def load_qfe():
    if not QFE_PATH.exists():
        raise FileNotFoundError(f"Source QFE tidak ditemukan: {QFE_PATH.name}")
    spec = importlib.util.spec_from_file_location("meat_qfe_source", str(QFE_PATH))
    if spec is None or spec.loader is None:
        raise ImportError("Tidak dapat mengimpor source QFE.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "qconv"):
        raise AttributeError("Fungsi qconv() tidak ditemukan pada source QFE.")
    return module

def preprocess_uploaded_image(uploaded_file, apply_clahe=True):
    """Preprocessing inference dengan pilihan CLAHE.

    Kedua kondisi tetap menjalankan resize 448x448, konversi RGB,
    resize 128x128, dan normalisasi. Jika apply_clahe=False, tahap CLAHE
    dilewati karena pengguna menyatakan gambar sudah melalui CLAHE.
    """
    raw = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("File gambar tidak dapat dibaca.")

    # Proses resize tetap dilakukan pada kedua kondisi.
    resized_bgr = cv2.resize(bgr, RESIZE_SIZE, interpolation=cv2.INTER_AREA)

    if apply_clahe:
        # CLAHE hanya diterapkan pada kanal L di ruang warna LAB.
        lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID,
        )
        l_clahe = clahe.apply(l_channel)

        # Kanal A dan B dipertahankan; hanya kanal L yang berubah.
        lab_processed = cv2.merge((l_clahe, a_channel, b_channel))
        processed_bgr = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2BGR)
        processing_label = "CLAHE diterapkan pada kanal L"
    else:
        # Gambar dianggap sudah melalui CLAHE; tidak dilakukan CLAHE ulang.
        processed_bgr = resized_bgr
        processing_label = "CLAHE dilewati (gambar dianggap sudah di-CLAHE)"

    # Tahap setelah CLAHE sama untuk kedua kondisi.
    processed_rgb_448 = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    rgb_128 = cv2.resize(
        processed_rgb_448, QFE_SIZE, interpolation=cv2.INTER_AREA
    )
    rgb_128 = rgb_128.astype(np.float64) / 255.0

    return processed_rgb_448, rgb_128, processing_label

def run_prediction(uploaded_file, apply_clahe=True):
    model = load_model()
    qfe = load_qfe()
    processed_rgb_448, image_128, processing_label = preprocess_uploaded_image(
        uploaded_file, apply_clahe=apply_clahe
    )

    # Gunakan QFE asli; tidak mengubah circuit, bobot C1, atau measurement.
    feature_map = np.asarray(qfe.qconv(image_128, mode=MODE), dtype=np.float32)
    if feature_map.shape != QFE_SHAPE:
        raise ValueError(f"Output QFE {feature_map.shape}, seharusnya {QFE_SHAPE}.")

    probabilities = np.asarray(model.predict(feature_map[None, ...], verbose=0)[0])
    if probabilities.shape != (3,):
        raise ValueError(f"Output prediksi tidak sesuai: {probabilities.shape}")
    index = int(np.argmax(probabilities))
    return processed_rgb_448, feature_map, probabilities, CLASS_NAMES[index], processing_label

st.title("🥩 HQC-CNN Meat Classification")
st.caption("Hybrid Quantum-Classical CNN · Intra Adaptive Cross (C1)")
st.markdown(
    "Unggah gambar daging dengan ukuran apa pun. Tentukan apakah CLAHE perlu "
    "diterapkan atau sudah pernah dilakukan, lalu aplikasi melanjutkan ke QFE "
    "12-qubit dan klasifikasi CNN."
)

with st.expander("Konfigurasi preprocessing penelitian", expanded=False):
    st.write("Resize: 448 × 448 piksel menggunakan OpenCV INTER_AREA.")
    st.write("Ruang warna: BGR → LAB.")
    st.write("Jika checkbox aktif: CLAHE hanya pada kanal L; kanal A dan B tidak dikenai CLAHE. Jika checkbox nonaktif: CLAHE dilewati.")
    st.write("clipLimit = 2.0; tileGridSize = (8, 8).")
    st.write("Setelah pemilihan CLAHE: konversi ke RGB, resize 128 × 128, normalisasi [0, 1].")
    st.write("QFE: mode intra_adaptive_cross, 12 qubit, output 64 × 64 × 6.")

uploaded = st.file_uploader(
    "Unggah gambar daging (JPG, JPEG, atau PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    try:
        uploaded_bytes = uploaded.getvalue()
        raw_preview = np.frombuffer(uploaded_bytes, dtype=np.uint8)
        decoded_preview = cv2.imdecode(raw_preview, cv2.IMREAD_COLOR)
        if decoded_preview is None:
            raise ValueError("File gambar tidak dapat dibaca.")
        original_rgb = cv2.cvtColor(decoded_preview, cv2.COLOR_BGR2RGB)
        st.image(
            original_rgb,
            caption=f"Gambar input · {decoded_preview.shape[1]} × {decoded_preview.shape[0]}",
            use_column_width=True,
        )
    except Exception as exc:
        st.error(f"Gagal membaca gambar: {exc}")
        st.stop()

    # Key berdasarkan isi gambar agar pilihan kembali ke default (aktif)
    # ketika pengguna mengunggah gambar yang berbeda.
    image_key = hashlib.sha256(uploaded_bytes).hexdigest()[:16]
    apply_clahe = st.checkbox(
        "Gambar belum melalui CLAHE (centang untuk menerapkan CLAHE)",
        value=True,
        key=f"apply_clahe_{image_key}",
        help=(
            "Biarkan tercentang jika gambar belum di-CLAHE. Hapus centang "
            "jika gambar sudah di-CLAHE agar CLAHE tidak diterapkan dua kali. "
            "Status CLAHE dipilih oleh pengguna; aplikasi tidak menebaknya otomatis."
        ),
    )

    if apply_clahe:
        st.info(
            "Checkbox aktif: gambar dianggap belum melalui CLAHE. Gambar di-resize "
            "ke 448×448, lalu CLAHE diterapkan hanya pada kanal L (LAB)."
        )
        spinner_text = (
            "Resize 448×448, menerapkan CLAHE pada kanal L, menghitung QFE, "
            "dan menjalankan CNN. Proses dapat memerlukan waktu..."
        )
    else:
        st.info(
            "Checkbox nonaktif: gambar dianggap sudah melalui CLAHE. Tahap CLAHE "
            "akan dilewati; resize, QFE, dan CNN tetap dijalankan."
        )
        spinner_text = (
            "Resize 448×448, melewati CLAHE, menghitung QFE, dan menjalankan CNN. "
            "Proses dapat memerlukan waktu..."
        )

    if st.button("Klasifikasikan gambar", type="primary", use_container_width=True):
        try:
            with st.spinner(spinner_text):
                processed_preview, fmap, probs, predicted, processing_label = run_prediction(
                    uploaded, apply_clahe=apply_clahe
                )
            st.image(
                processed_preview,
                caption=f"Citra 448×448 setelah resize · {processing_label}",
                use_column_width=True,
            )
            confidence = float(np.max(probs) * 100.0)
            st.success(f"Hasil klasifikasi: **{predicted}**")
            col1, col2 = st.columns(2)
            col1.metric("Kelas prediksi", predicted)
            col2.metric("Confidence", f"{confidence:.2f}%")
            st.subheader("Probabilitas setiap kelas")
            st.bar_chart({name: [float(probs[i] * 100.0)] for i, name in enumerate(CLASS_NAMES)})
            st.caption(f"Mode: {MODE} · Bentuk feature map: {fmap.shape}")
        except Exception as exc:
            st.error(f"Prediksi gagal: {exc}")
            st.exception(exc)

with st.expander("Informasi pipeline"):
    st.write("1. Input gambar asli dengan ukuran bebas.")
    st.write("2. Resize gambar ke 448 × 448 menggunakan INTER_AREA.")
    st.write("3. Jika checkbox aktif, BGR → LAB lalu CLAHE clipLimit=2.0 dan tileGridSize=(8,8) hanya pada kanal L; kanal A dan B dipertahankan. Jika checkbox nonaktif, CLAHE dilewati.")
    st.write("4. Konversi ke RGB, resize 128 × 128, dan normalisasi [0,1].")
    st.write("5. QFE asli menjalankan mode intra_adaptive_cross dengan 12 qubit.")
    st.write("6. Feature map 64 × 64 × 6 dimasukkan ke model CNN.")
    st.write("7. CNN menghasilkan probabilitas Beef, Horse_Meat, dan Pork.")
