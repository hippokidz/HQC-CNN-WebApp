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

# Responsive, compact layout. Columns become a vertical stack on narrow screens.
st.markdown(
    """
    <style>
      /* Clean, compact page: designed to fit common desktop viewports. */
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {
        max-width: 1440px;
        padding: .35rem 1rem .45rem 1rem;
      }
      .app-hero {
        width: 100%;
        text-align: center;
        margin: 0 auto .45rem auto;
        padding: .1rem .2rem .2rem .2rem;
      }
      .app-hero-title {
        width: 100%;
        margin: 0 auto;
        text-align: center;
        font-size: clamp(1.35rem, 2.5vw, 2.05rem);
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: .005em;
        overflow-wrap: anywhere;
      }
      .app-hero-subtitle {
        width: 100%;
        margin: .25rem auto 0 auto;
        text-align: center;
        color: #8a8f98;
        font-size: clamp(.74rem, 1vw, .92rem);
        line-height: 1.25;
        overflow-wrap: anywhere;
      }
      h2, h3 {font-size: 1.12rem !important; margin: .15rem 0 .25rem 0 !important;}
      [data-testid="stCaptionContainer"] {margin: .05rem 0 .15rem 0;}
      [data-testid="stFileUploader"] {padding: 0 !important; margin: 0 0 .15rem 0 !important;}
      [data-testid="stFileUploaderDropzone"] {padding: .35rem .55rem !important; min-height: unset !important;}
      [data-testid="stFileUploaderDropzoneInstructions"] svg {width: 28px !important; height: 28px !important;}
      [data-testid="stImage"] {margin: .05rem 0 .15rem 0;}
      [data-testid="stImage"] img {
        display: block;
        max-height: 150px !important;
        max-width: 100% !important;
        width: auto !important;
        object-fit: contain !important;
      }
      [data-testid="stCheckbox"] {margin: .05rem 0 .05rem 0;}
      [data-testid="stCheckbox"] label {font-size: .9rem !important;}
      [data-testid="stAlert"] {padding: .3rem .55rem !important; margin: .15rem 0 !important;}
      [data-testid="stMetric"] {padding: .25rem .4rem !important;}
      [data-testid="stProgress"] {margin: .05rem 0 .18rem 0;}
      div.stButton > button {min-height: 2.25rem; padding: .35rem .7rem; font-weight: 650;}
      [data-testid="stHorizontalBlock"] {gap: .8rem; align-items: flex-start;}
      [data-testid="stExpander"] {margin-top: .25rem;}
      [data-testid="stExpander"] details summary {padding-top: .25rem; padding-bottom: .25rem;}
      /* Short desktop/laptop windows get a more compact preview. */
      @media (max-height: 800px) and (min-width: 769px) {
        .block-container {padding-top: .2rem; padding-bottom: .25rem;}
        .app-hero {margin-bottom: .25rem; padding-bottom: .05rem;}
        .app-hero-subtitle {margin-top: .15rem;}
        [data-testid="stImage"] img {max-height: 125px !important;}
        [data-testid="stFileUploaderDropzone"] {padding: .25rem .45rem !important;}
        [data-testid="stAlert"] {padding: .2rem .45rem !important;}
      }
      /* Phones: smaller preview and tighter spacing. The panels stack naturally. */
      @media (max-width: 768px) {
        .block-container {padding: .25rem .55rem .35rem .55rem;}
        .app-hero {margin-bottom: .3rem; padding-bottom: .1rem;}
        .app-hero-title {font-size: clamp(1.18rem, 5.2vw, 1.55rem); line-height: 1.12;}
        .app-hero-subtitle {font-size: .72rem; margin-top: .15rem; line-height: 1.2;}
        h2, h3 {font-size: 1rem !important; margin: .1rem 0 .18rem 0 !important;}
        [data-testid="stImage"] img {max-height: 105px !important;}
        [data-testid="stFileUploaderDropzone"] {padding: .25rem .4rem !important;}
        [data-testid="stFileUploaderDropzoneInstructions"] svg {width: 22px !important; height: 22px !important;}
        [data-testid="stCheckbox"] label {font-size: .82rem !important;}
        [data-testid="stHorizontalBlock"] {gap: .45rem;}
        div.stButton > button {width: 100%; min-height: 2.15rem;}
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
        raise ValueError(
            f"Input model adalah {model.input_shape}; seharusnya (None, 64, 64, 6)."
        )
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
    raw = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("File gambar tidak dapat dibaca.")

    resized_bgr = cv2.resize(bgr, RESIZE_SIZE, interpolation=cv2.INTER_AREA)

    if apply_clahe:
        lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID,
        )
        l_clahe = clahe.apply(l_channel)
        lab_processed = cv2.merge((l_clahe, a_channel, b_channel))
        processed_bgr = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2BGR)
        processing_label = "CLAHE diterapkan pada kanal L"
    else:
        processed_bgr = resized_bgr
        processing_label = "CLAHE dilewati (gambar dianggap sudah di-CLAHE)"

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
    # Original PennyLane QFE is retained without changing circuit or C1 weights.
    feature_map = np.asarray(qfe.qconv(image_128, mode=MODE), dtype=np.float32)
    if feature_map.shape != QFE_SHAPE:
        raise ValueError(f"Output QFE {feature_map.shape}, seharusnya {QFE_SHAPE}.")
    probabilities = np.asarray(
        model.predict(feature_map[None, ...], verbose=0)[0]
    )
    if probabilities.shape != (3,):
        raise ValueError(f"Output prediksi tidak sesuai: {probabilities.shape}")
    index = int(np.argmax(probabilities))
    return (
        processed_rgb_448,
        feature_map,
        probabilities,
        CLASS_NAMES[index],
        processing_label,
    )


# Keep result available across Streamlit reruns and clear it when a new image is selected.
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_image_key" not in st.session_state:
    st.session_state.prediction_image_key = None

st.markdown(
    """
    <div class="app-hero">
      <div class="app-hero-title">🥩 HQC-CNN Meat Classification</div>
      <div class="app-hero-subtitle">Hybrid Quantum-Classical CNN With Intra Adaptive Cross Channel Entanglement</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main responsive workspace: side-by-side on desktop, stacked on mobile.
left, right = st.columns([0.92, 1.35], gap="medium")

with left:
    st.subheader("Input gambar")
    uploaded = st.file_uploader(
        "Pilih gambar daging",
        type=["jpg", "jpeg", "png"],
        help="Format JPG, JPEG, atau PNG. Ukuran gambar bebas.",
    )

    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        image_key = hashlib.sha256(uploaded_bytes).hexdigest()[:16]
        if st.session_state.prediction_image_key != image_key:
            st.session_state.prediction_result = None
            st.session_state.prediction_image_key = image_key

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

        apply_clahe = st.checkbox(
            "Gambar belum melalui CLAHE (terapkan CLAHE)",
            value=True,
            key=f"apply_clahe_{image_key}",
            help=(
                "Centang jika gambar belum di-CLAHE. Hilangkan centang jika gambar "
                "sudah di-CLAHE agar CLAHE tidak diterapkan dua kali."
            ),
        )
        if apply_clahe:
            st.caption("CLAHE aktif: hanya kanal L pada ruang warna LAB yang diproses.")
            spinner_text = "Resize, CLAHE kanal L, QFE 12-qubit, dan klasifikasi CNN..."
        else:
            st.caption("CLAHE nonaktif: tahap CLAHE dilewati.")
            spinner_text = "Resize, QFE 12-qubit, dan klasifikasi CNN..."

        if st.button(
            "🔍 Klasifikasikan gambar",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(spinner_text):
                    result = run_prediction(uploaded, apply_clahe=apply_clahe)
                st.session_state.prediction_result = result
            except Exception as exc:
                st.session_state.prediction_result = None
                st.error(f"Prediksi gagal: {exc}")
                st.exception(exc)
    else:
        st.info("Unggah gambar untuk memulai klasifikasi.")

with right:
    st.subheader("Hasil klasifikasi")
    result = st.session_state.prediction_result
    if result is not None:
        processed_preview, fmap, probs, predicted, processing_label = result
        st.image(
            processed_preview,
            caption=f"Hasil preprocessing · {processing_label}",
            use_container_width=True,
        )
        confidence = float(np.max(probs) * 100.0)
        st.success(f"Prediksi: **{predicted}**")
        metric_cols = st.columns(2)
        metric_cols[0].metric("Kelas", predicted)
        metric_cols[1].metric("Confidence", f"{confidence:.2f}%")
        st.markdown("**Probabilitas kelas**")
        for name, probability in zip(CLASS_NAMES, probs):
            value = float(probability)
            st.progress(max(0.0, min(1.0, value)), text=f"{name}: {value * 100:.2f}%")
        st.caption(f"Mode: {MODE} · Feature map: {fmap.shape}")
    else:
        st.caption("Hasil akan tampil di sini setelah proses klasifikasi selesai.")

with st.expander("Detail konfigurasi penelitian"):
    st.write("Preprocessing: resize 448 × 448, OpenCV INTER_AREA.")
    st.write("CLAHE: ruang warna LAB, hanya kanal L; clipLimit=2.0; tileGridSize=(8,8).")
    st.write("Jika checkbox tidak aktif, CLAHE dilewati karena gambar dianggap sudah di-CLAHE.")
    st.write("QFE: PennyLane default.qubit, 12 qubit, patch 2 × 2, mode intra_adaptive_cross.")
    st.write("Input CNN: feature map 64 × 64 × 6; kelas: Beef, Horse_Meat, Pork.")
