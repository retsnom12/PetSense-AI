"""Streamlit interface for multimodal PetSense-AI inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st

from src.audio_features import extract_audio_features
from src.vision_metrics import FacialExpressionAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "petsense_audio_classifier.joblib"

st.set_page_config(page_title="PetSense-AI", page_icon="🐾", layout="wide")
st.title("PetSense-AI")
st.caption("Multimodal pet vocalization and facial expression classifier")


def render_audio_tab() -> None:
    """Render vocalization upload, inference, and probability results."""
    uploaded_audio = st.file_uploader(
        "Upload a pet vocalization",
        type=["wav", "mp3"],
        key="audio_uploader",
    )
    if uploaded_audio is None:
        st.info("Upload a WAV or MP3 file to classify a pet vocalization.")
        return

    st.audio(uploaded_audio)
    if not MODEL_PATH.is_file():
        st.warning(
            "No trained model was found. Add labeled audio to data/<class>/ "
            "and run `python -m src.train` before using this tab."
        )
        return

    temporary_path = PROJECT_ROOT / f".petsense_{uploaded_audio.name}"
    try:
        temporary_path.write_bytes(uploaded_audio.getvalue())
        features = extract_audio_features(str(temporary_path)).reshape(1, -1)
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        st.subheader(f"Predicted class: {prediction}")
        st.write("Class probabilities")
        for class_name, probability in zip(model.classes_, probabilities):
            st.progress(float(probability), text=f"{class_name}: {probability:.1%}")
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        st.error(f"Audio classification failed: {error}")
    finally:
        temporary_path.unlink(missing_ok=True)


def render_vision_tab() -> None:
    """Render facial image analysis and annotated output."""
    uploaded_image = st.file_uploader(
        "Upload a pet facial image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )
    if uploaded_image is None:
        st.info("Upload a JPG, JPEG, or PNG image to analyze facial metrics.")
        return

    image_bytes = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("The uploaded file could not be decoded as an image.")
        return

    try:
        analyzer = FacialExpressionAnalyzer()
        metrics = analyzer.analyze(image)
    except (RuntimeError, ValueError) as error:
        st.error(f"Facial analysis failed: {error}")
        return

    annotated_image = image.copy()
    if metrics["face_detected"]:
        x, y, width, height = metrics["bbox"]
        cv2.rectangle(annotated_image, (x, y), (x + width, y + height), (0, 200, 0), 2)
    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Analyzed image")
    if not metrics["face_detected"]:
        st.warning("No frontal animal face was detected in the uploaded image.")

    aspect_ratio, tension_index = st.columns(2)
    aspect_ratio.metric("Facial Aspect Ratio", f"{metrics['aspect_ratio']:.3f}")
    tension_index.metric(
        "Whisker/Muzzle Tension Index",
        f"{metrics['whisker_tension_proxy']:.3f}",
    )


vocalization_tab, facial_metrics_tab = st.tabs(
    ["Vocalization Classifier", "Facial Expression Metrics"]
)
with vocalization_tab:
    render_audio_tab()
with facial_metrics_tab:
    render_vision_tab()

