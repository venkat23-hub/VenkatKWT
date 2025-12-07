# app.py
import streamlit as st
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os

# Force torchaudio to use soundfile backend → no TorchCodec/FFmpeg error!
torchaudio.set_audio_backend("soundfile")

# Your project files
from model import KeywordTransformer
from preprocessing import AudioPreprocessor
from config import label_dict, N_MELS, FIXED_TIME_DIM

# ========================
# Model & Preprocessor Setup
# ========================
IDX_TO_LABEL = {v: k for k, v in label_dict.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model_and_preprocessor():
    model_path = "best_finetuned_from_npy.pth"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = KeywordTransformer(
        image_size=(N_MELS, FIXED_TIME_DIM),
        patch_size=(40, 10),
        num_classes=len(label_dict),
        dim=160,
        depth=6,
        heads=8,
        mlp_dim=256,
        dropout=0.4
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=40,
        n_fft=400,
        hop_length=160,
        fixed_time=101,
        target_duration=1.0
    )

    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# ========================
# Beautiful UI (Same as Flask)
# ========================
st.markdown("""
<div style="background: linear-gradient(135deg, #0052A3, #003d7a);
            color: white; padding: 40px; text-align: center;
            border-radius: 16px; margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,82,163,0.3);">
    <h1 style="margin:0; font-size:38px; font-weight:700;">IIIT Sricity</h1>
    <h2 style="margin:15px 0 0; opacity:0.95; font-size:22px;">
        Personal Keyword Spotting System
    </h2>
    <p style="margin:10px 0 0; font-size:16px; opacity:0.9;">
        Your Voice • Your Wake Word • 100% Private
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Choose Input Method")

col1, col2 = st.columns(2)

# Upload Audio
with col1:
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'ogg', 'webm', 'm4a'],
        help="Supports WAV, MP3, OGG, WebM, M4A"
    )

# Record from Microphone
with col2:
    st.markdown("**OR** Record Live")
    recorded_audio = st.audio_input("Click mic and say your keyword")

# Save audio to temporary file
audio_path = None
if uploaded_file is not None:
    audio_path = f"temp_upload_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded: {uploaded_file.name}")

elif recorded_audio is not None:
    audio_path = "temp_recorded.wav"
    with open(audio_path, "wb") as f:
        f.write(recorded_audio)
    st.success("Recording saved!")

# ========================
# Predict Button & Result
# ========================
if audio_path and st.button("Predict Keyword", type="primary", use_container_width=True):
    with st.spinner("Analyzing your voice..."):
        try:
            # Same preprocessing as your Flask app
            tensor, mel_spec = preprocessor.preprocess(audio_path)
            tensor = tensor.to(device)

            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probs, dim=1)
                keyword = IDX_TO_LABEL[predicted_idx.item()]
                confidence_percent = round(confidence.item() * 100, 2)

            # === SUCCESS ===
            st.success("**Prediction Successful!**")

            colA, colB = st.columns(2)
            with colA:
                st.metric("**Detected Keyword**", value=keyword.upper())
            with colB:
                st.metric("**Confidence**", value=f"{confidence_percent}%")

            # === Mel-Spectrogram ===
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax, format='%+2.0f dB')
            ax.set_title('Mel-Spectrogram', fontsize=14, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(audio_path):
                os.remove(audio_path)

else:
    if not uploaded_file and not recorded_audio:
        st.info("Please upload an audio file or record your voice to get started.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#666;'>"
    "Made with ❤️ by IIIT Sricity BTP Student | Personal Wake Word Detector"
    "</p>",
    unsafe_allow_html=True
)