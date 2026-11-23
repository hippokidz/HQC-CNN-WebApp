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

st.set_page_config(
    page_title="HQC-CNN Meat Classifier",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS hanya mengatur tampilan. Tidak mengubah preprocessing, QFE, atau CNN.
st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .block-container {
        max-width: 1500px;
        padding-top: 1.15rem !important;
        padding-bottom: .55rem;
        padding-left: clamp(.65rem, 2vw, 1.6rem);
        padding-right: clamp(.65rem, 2vw, 1.6rem);
        overflow: visible !important;
      }
      /* Header uses div/span, not Streamlit's H1 widget.
         No fixed height, clipping, negative margin, or hidden overflow. */
      .hqc-header {
        width: 100%;
        min-height: 76px;
        height: auto;
        box-sizing: border-box;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 12px 10px 8px;
        margin: 0 auto;
        overflow: visible !important;
        text-align: center;
      }
      .hqc-header-icon {
        display: inline-flex;
        flex: 0 0 auto;
        align-items: center;
        justify-content: center;
        font-size: clamp(1.65rem, 2.8vw, 2.25rem);
        line-height: 1.25;
        padding: 2px 0;
        overflow: visible;
      }
      .hqc-header-title {
        display: inline-block;
        min-width: 0;
        max-width: 100%;
        color: #31333f;
        font-size: clamp(1.4rem, 2.6vw, 2.15rem);
        font-weight: 800;
        line-height: 1.28;
        padding: 3px 0 5px;
        margin: 0;
        overflow: visible;
        overflow-wrap: normal;
        word-break: normal;
      }
      .hqc-subtitle {
        display: block;
        width: 100%;
        box-sizing: border-box;
        color: #888d98;
        text-align: center;
        font-size: clamp(.75rem, 1.1vw, .95rem);
        line-height: 1.45;
        padding: 2px 8px 5px;
        margin: 0 auto .45rem auto;
        overflow: visible !important;
      }
      [data-testid="stMarkdownContainer"] {overflow: visible !important;}
      h2, h3 {margin-top: .2rem !important; margin-bottom: .35rem !important;}
      [data-testid="stFileUploader"] {margin-bottom: .15rem;}
      [data-testid="stFileUploaderDropzone"] {padding: .35rem .55rem !important;}
      [data-testid="stImage"] {margin: .08rem 0 .15rem 0;}
      [data-testid="stImage"] img {
        max-height: 175px !important;
        width: auto !important;
        max-width: 100% !important;
        object-fit: contain !important;
      }
      [data-testid="stAlert"] {padding: .4rem .65rem !important; margin: .2rem 0 !important;}
      [data-testid="stMetric"] {padding: .25rem .4rem !important;}
      [data-testid="stCheckbox"] {margin: .12rem 0 !important;}
      div.stButton > button {min-height: 2.35rem; font-weight: 650;}
      [data-testid="stHorizontalBlock"] {gap: clamp(.6rem, 1.4vw, 1.2rem); align-items: flex-start;}
      @media (max-width: 768px) {
        .block-container {padding-top: .8rem !important; padding-bottom: .4rem;}
        .hqc-header {
          min-height: 64px;
          gap: 7px;
          padding: 10px 5px 6px;
        }
        .hqc-header-icon {
          font-size: clamp(1.25rem, 5vw, 1.65rem);
          line-height: 1.3;
        }
        .hqc-header-title {
          font-size: clamp(1.05rem, 4.6vw, 1.45rem);
          line-height: 1.3;
          padding: 3px 0 5px;
          overflow-wrap: anywhere;
        }
        .hqc-subtitle {
          font-size: .72rem;
          line-height: 1.4;
          margin-bottom: .35rem;
        }
        h2, h3 {font-size: 1rem !important;}
        [data-testid="stImage"] img {max-height: 115px !important;}
        [data-testid="stFileUploaderDropzone"] {padding: .25rem .4rem !important;}
        div.stButton > button {width: 100%;}
      }
      @media (max-height: 800px) and (min-width: 769px) {
        .block-container {padding-top: .85rem !important; padding-bottom: .3rem;}
        .hqc-header {
          min-height: 68px;
          padding-top: 10px;
          padding-bottom: 6px;
        }
        .hqc-header-title {
          font-size: clamp(1.3rem, 2.3vw, 1.9rem);
          line-height: 1.3;
        }
        .hqc-subtitle {margin-bottom: .25rem;}
        [data-testid="stImage"] img {max-height: 145px !important;}
        [data-testid="stFileUploaderDropzone"] {padding: .25rem .4rem !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Header rendered as a natural-height HTML flex row.
# Avoids Streamlit H1's heading wrapper, which was clipped in the deployed layout.
st.markdown(
    """
    <div class="hqc-header">
      <span class="hqc-header-icon" aria-hidden="true">🥩</span>
      <span class="hqc-header-title">HQC-CNN Meat Classification</span>
    </div>
    <div class="hqc-subtitle">
      Hybrid Quantum-Classical CNN With Intra Adaptive Cross Channel Entanglement
    </div>
    """,
    unsafe_allow_html=True,
)

if "hqc_prediction" not in st.session_state:
    st.session_state.hqc_prediction = None
if "hqc_prediction_signature" not in st.session_state:
    st.session_state.hqc_prediction_signature = None

left, right = st.columns([1, 1.12], gap="medium")

with left:
    st.subheader("Input gambar")
    uploaded = st.file_uploader(
        "Pilih gambar daging",
        type=["jpg", "jpeg", "png"],
        help="Unggah gambar daging JPG, JPEG, atau PNG dengan ukuran apa pun.",
    )

    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        raw_preview = np.frombuffer(uploaded_bytes, dtype=np.uint8)
        decoded_preview = cv2.imdecode(raw_preview, cv2.IMREAD_COLOR)
        if decoded_preview is None:
            st.error("File gambar tidak dapat dibaca.")
            st.stop()

        original_rgb = cv2.cvtColor(decoded_preview, cv2.COLOR_BGR2RGB)
        st.image(
            original_rgb,
            caption=f"Gambar input · {decoded_preview.shape[1]} × {decoded_preview.shape[0]}",
            use_container_width=True,
        )

        image_key = hashlib.sha256(uploaded_bytes).hexdigest()[:16]
        apply_clahe = st.checkbox(
            "Gambar belum melalui CLAHE (centang untuk menerapkan CLAHE)",
            value=True,
            key=f"apply_clahe_{image_key}",
            help=(
                "Centang jika gambar belum melalui CLAHE. Hilangkan centang jika "
                "gambar sudah melalui CLAHE agar tidak diproses dua kali."
            ),
        )

        # Signature includes image bytes AND CLAHE selection. This prevents
        # showing a previous prediction after the user changes the checkbox.
        prediction_signature = hashlib.sha256(
            uploaded_bytes + f"|clahe={int(apply_clahe)}".encode("utf-8")
        ).hexdigest()
        if st.session_state.hqc_prediction_signature != prediction_signature:
            st.session_state.hqc_prediction = None
            st.session_state.hqc_prediction_signature = None

        if apply_clahe:
            st.caption("CLAHE aktif: CLAHE diterapkan hanya pada kanal L (LAB).")
            spinner_text = (
                "Resize 448×448, CLAHE kanal L, Quantum Feature Extraction, "
                "dan klasifikasi CNN sedang berjalan..."
            )
        else:
            st.caption("CLAHE nonaktif: tahap CLAHE dilewati (gambar dianggap sudah di-CLAHE).")
            spinner_text = (
                "Resize 448×448, CLAHE dilewati, Quantum Feature Extraction, "
                "dan klasifikasi CNN sedang berjalan..."
            )

        if st.button("Klasifikasikan gambar", type="primary", use_container_width=True):
            try:
                with st.spinner(spinner_text):
                    result = run_prediction(uploaded, apply_clahe=apply_clahe)
                st.session_state.hqc_prediction = result
                st.session_state.hqc_prediction_signature = prediction_signature
            except Exception as exc:
                st.session_state.hqc_prediction = None
                st.session_state.hqc_prediction_signature = None
                st.error(f"Prediksi gagal: {exc}")
                st.exception(exc)
    else:
        st.info("Unggah gambar untuk memulai klasifikasi.")
        st.session_state.hqc_prediction = None
        st.session_state.hqc_prediction_signature = None

with right:
    st.subheader("Hasil klasifikasi")
    result = st.session_state.hqc_prediction
    if result is not None:
        processed_preview, fmap, probs, predicted, processing_label = result
        st.image(
            processed_preview,
            caption=f"Citra setelah preprocessing · {processing_label}",
            use_container_width=True,
        )
        confidence = float(np.max(probs) * 100.0)
        st.success(f"Hasil klasifikasi: **{predicted}**")
        metric_cols = st.columns(2)
        metric_cols[0].metric("Kelas prediksi", predicted)
        metric_cols[1].metric("Confidence", f"{confidence:.2f}%")
        st.markdown("**Probabilitas setiap kelas**")
        for name, probability in zip(CLASS_NAMES, probs):
            value = float(probability)
            st.progress(
                max(0.0, min(1.0, value)),
                text=f"{name}: {value * 100:.2f}%",
            )
        st.caption(f"Mode: {MODE} · Feature map: {fmap.shape}")
    else:
        st.info("Hasil prediksi akan muncul di sini setelah klasifikasi selesai.")

with st.expander("Detail konfigurasi penelitian"):
    st.write("Resize: 448 × 448 piksel dengan OpenCV INTER_AREA.")
    st.write("Jika checkbox aktif: BGR → LAB, CLAHE hanya pada kanal L, clipLimit=2.0, tileGridSize=(8,8); kanal A dan B dipertahankan.")
    st.write("Jika checkbox nonaktif: CLAHE dilewati karena gambar dianggap sudah diproses.")
    st.write("Setelah itu: RGB → 128 × 128 → normalisasi [0,1].")
    st.write("QFE asli: PennyLane default.qubit, 12 qubit, patch 2 × 2, mode intra_adaptive_cross.")
    st.write("CNN: input feature map 64 × 64 × 6, output Beef, Horse_Meat, Pork.")
