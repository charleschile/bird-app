import io
import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st

from audio_utils import denoise_audio
from model_utils import load_model, predict_bird


# ---------------- Page Layout ----------------
st.set_page_config(
    page_title="🐦 Bird Audio Classifier",
    page_icon="🐦",
    layout="wide",
)


# ---------------- Header ----------------
st.markdown(
    """
    <div style="text-align:center; padding: 10px 0;">
        <h1 style="font-size:42px;">🐦 Bird Sound Denoising & Classification</h1>
        <p style="font-size:18px; color:#666;">
        Upload a bird audio clip, and the system will perform <b>noise reduction</b> and classify it using powerful EfficientNet-based models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- Scan Model Directory ----------------
MODEL_DIR = "models"
available_models = [
    f for f in os.listdir(MODEL_DIR)
    if f.endswith(".pth")
]

if not available_models:
    st.error("❌ No .pth models found in the models/ directory.")
    st.stop()


# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

selected_model_file = st.sidebar.selectbox(
    "Select Model Checkpoint",
    available_models,
)

show_waveform = st.sidebar.checkbox("Show Waveform", value=True)
show_spec = st.sidebar.checkbox("Show Spectrogram", value=False)
show_prob_table = st.sidebar.checkbox("Show Probability Table", value=True)

# Load model
model_path = os.path.join(MODEL_DIR, selected_model_file)
model = load_model(model_path)


# ---------------- File Upload ----------------
uploaded_file = st.file_uploader(
    "🎤 Upload Bird Audio (OGG / WAV / MP3)",
    type=["ogg", "wav", "mp3"],
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Original Audio")
        st.audio(uploaded_file)

    # Load audio
    y, sr = librosa.load(uploaded_file, sr=None)

    st.info(f"Sample Rate: {sr} Hz | Duration: {len(y)/sr:.2f} seconds")

    if show_waveform or show_spec:
        fig, ax = plt.subplots(2 if show_spec else 1, 1, figsize=(10, 5))

        # Waveform
        if show_waveform:
            if show_spec:
                ax_wave = ax[0]
            else:
                ax_wave = ax

            ax_wave.plot(y, color="black")
            ax_wave.set_title("Waveform (Original)")

        # Spectrogram
        if show_spec:
            import librosa.display
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
            ax[1].set_title("Spectrogram (Original)")

        st.pyplot(fig)


    # # ---------------- Process Button ----------------
    # if st.button("🚀 Run Noise Reduction & Classification"):
    #     with st.spinner("Running noise reduction and model inference..."):
    #         denoised = denoise_audio(y, sr)

    #         # Convert to WAV bytes
    #         wav_bytes = io.BytesIO()
    #         sf.write(wav_bytes, denoised, sr, format="WAV")
    #         wav_bytes.seek(0)

    #         # Prediction
    #         pred_label, prob, prob_dict = predict_bird(model, denoised, sr)

    #     # ---------------- Results ----------------
    #     st.success(f"🎯 Prediction: **{pred_label}** (Confidence {prob:.2f})")

    #     col1, col2 = st.columns([1, 1])

    #     with col1:
    #         st.subheader("🎧 Denoised Audio")
    #         st.audio(wav_bytes)

    #     # Plot denoised waveform/spectrogram
    #     if show_waveform or show_spec:
    #         fig, ax = plt.subplots(2 if show_spec else 1, 1, figsize=(10, 5))

    #         if show_waveform:
    #             if show_spec:
    #                 ax_wave = ax[0]
    #             else:
    #                 ax_wave = ax

    #             ax_wave.plot(denoised, color="green")
    #             ax_wave.set_title("Waveform (Denoised)")

    #         if show_spec:
    #             D = librosa.amplitude_to_db(np.abs(librosa.stft(denoised)), ref=np.max)
    #             import librosa.display
    #             librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    #             ax[1].set_title("Spectrogram (Denoised)")

    #         st.pyplot(fig)

    #     # Probability Table
    #     if show_prob_table and prob_dict:
    #         st.subheader("📊 Probability Distribution")
    #         st.table({
    #             "Class": list(prob_dict.keys()),
    #             "Probability": [f"{v:.3f}" for v in prob_dict.values()],
    #         })

# ---------------- Step 1: Noise Reduction Button ----------------
if st.button("🔧 Run Noise Reduction"):
    with st.spinner("Running noise reduction..."):
        denoised = denoise_audio(y, sr)

        # Store in session so we can use it for classification later
        st.session_state["denoised_audio"] = denoised

        # Convert to WAV bytes
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, denoised, sr, format="WAV")
        wav_bytes.seek(0)

    st.success("Noise reduction completed.")

    # Show denoised audio player
    st.subheader("🎧 Denoised Audio")
    st.audio(wav_bytes)

    # Show waveform/spectrogram
    if show_waveform or show_spec:
        fig, ax = plt.subplots(2 if show_spec else 1, 1, figsize=(10, 5))

        # Waveform
        if show_waveform:
            if show_spec:
                ax_wave = ax[0]
            else:
                ax_wave = ax
            ax_wave.plot(denoised, color="green")
            ax_wave.set_title("Waveform (Denoised)")

        # Spectrogram
        if show_spec:
            import librosa.display
            D = librosa.amplitude_to_db(np.abs(librosa.stft(denoised)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
            ax[1].set_title("Spectrogram (Denoised)")

        st.pyplot(fig)


# ---------------- Step 2: Classification Button ----------------
if "denoised_audio" in st.session_state:

    st.markdown("---")
    st.subheader("📌 Step 2: Classification")

    if st.button("🚀 Run Classification"):
        denoised = st.session_state["denoised_audio"]

        with st.spinner("Running model inference..."):
            pred_label, prob, prob_dict = predict_bird(model, denoised, sr)

        # Prediction result
        st.success(f"🎯 Prediction: **{pred_label}** (Confidence {prob:.2f})")

        # Probability Table
        if show_prob_table and prob_dict:
            st.subheader("📊 Probability Distribution")
            st.table({
                "Class": list(prob_dict.keys()),
                "Probability": [f"{v:.3f}" for v in prob_dict.values()],
            })

else:
    st.info("👆 Please upload an audio file to begin.")
