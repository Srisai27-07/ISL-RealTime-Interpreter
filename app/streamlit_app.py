"""
Streamlit Web Application: Vaani-ISL Real-Time Interpreter.
Designed for judge presentations, browser interaction, and accessibility demonstration.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
import os
import yaml
import cv2
import numpy as np
import streamlit as st

sys.path.append(".")

from src.vision.landmark_extractor import ISLLandmarkExtractor
from src.models.temporal_model import load_checkpoint
from src.pipeline.real_time_predictor import RealTimePredictor
from src.utils.text_to_speech import TextToSpeechEngine
from src.utils.visualizer import HUDVisualizer


# Page Configuration
st.set_page_config(
    page_title="Vaani-ISL: Real-Time Indian Sign Language Interpreter",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern hackathon presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background-color: #1E1E2E;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #4ECDC4;
        margin-bottom: 1rem;
    }
    .sentence-box {
        font-size: 1.5rem;
        font-weight: 600;
        color: #F8F9FA;
        background: #2B2D42;
        padding: 1rem;
        border-radius: 8px;
        min-height: 3.5rem;
        border: 1px solid #4A4E69;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    st.markdown('<div class="main-title">🤟 Vaani-ISL: Real-Time Interpreter</div>', unsafe_allow_html=True)
    st.caption("AI-Powered Dual-Hand Indian Sign Language Translator | Hackathon Edition")

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ System Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.50, 0.95, float(config["sequence"]["prediction_threshold"]), 0.05)
        camera_idx = st.number_input("Camera Index", value=0, min_value=0, max_value=3)
        tts_enabled = st.checkbox("Enable Text-to-Speech Audio", value=True)

        st.divider()
        st.subheader("📚 Active ISL Vocabulary")
        for cls in config["classes"]:
            st.markdown(f"- **{cls}**")

        st.divider()
        st.info("💡 **Hackathon Tip:** Make sure your hands are clearly visible within the frame. ISL uses both hands!")

    col_video, col_info = st.columns([3, 2])

    with col_info:
        st.subheader("📝 Live Transcription")
        sentence_placeholder = st.empty()
        sentence_placeholder.markdown('<div class="sentence-box">[Perform signs to construct sentence...]</div>', unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns(2)
        clear_clicked = col_btn1.button("🗑️ Clear Transcript", use_container_width=True)
        speak_clicked = col_btn2.button("🔊 Speak Sentence", use_container_width=True)

        st.divider()
        st.subheader("🎯 Recognition Telemetry")
        current_sign_metric = st.empty()
        current_conf_metric = st.empty()
        status_metric = st.empty()

        current_sign_metric.metric("Current Sign", "None")
        current_conf_metric.metric("Model Confidence", "0%")
        status_metric.success("System Ready")

    with col_video:
        st.subheader("📷 Live Camera Stream")
        video_placeholder = st.empty()
        run_camera = st.toggle("Start Camera Feed", value=False)

    weights_path = config["model"]["weights_path"]
    model_ready = os.path.exists(weights_path)

    if not model_ready:
        st.warning(f"⚠️ Model weights not found at `{weights_path}`. Run `python scripts/03_train_model.py --generate-mock` to generate weights for demo testing.")

    if run_camera:
        if not model_ready:
            st.error("Cannot run live camera inference without model weights. Please run training script first.")
            return

        model, class_names = load_checkpoint(weights_path)
        tts = TextToSpeechEngine(enabled=tts_enabled)
        extractor = ISLLandmarkExtractor()
        visualizer = HUDVisualizer()

        predictor = RealTimePredictor(
            model=model,
            class_names=class_names,
            sequence_length=config["sequence"]["sequence_length"],
            prediction_threshold=conf_threshold,
            debounce_frames=config["sequence"]["debounce_frames"],
            cooldown_seconds=config["sequence"]["cooldown_seconds"],
            on_word_committed=lambda word: tts.speak(word) if tts_enabled else None
        )

        cap = cv2.VideoCapture(int(camera_idx))

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            features, results = extractor.process_frame(frame)
            annotated = extractor.draw_landmarks(frame, results)

            pred_class, confidence, sentence = predictor.update(features)

            # Update Streamlit UI components
            if pred_class:
                current_sign_metric.metric("Current Sign", pred_class)
                current_conf_metric.metric("Model Confidence", f"{int(confidence * 100)}%")
            else:
                current_sign_metric.metric("Current Sign", "Waiting for gesture...")
                current_conf_metric.metric("Model Confidence", "0%")

            sentence_text = predictor.get_sentence_text()
            if sentence_text:
                sentence_placeholder.markdown(f'<div class="sentence-box">{sentence_text}</div>', unsafe_allow_html=True)

            # Render frame to Streamlit
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        extractor.close()
        tts.stop()


if __name__ == "__main__":
    main()
