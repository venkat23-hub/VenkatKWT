# app.py — FINAL VERSION (Recording + Upload Both Work 100%)
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

# Your files
from model import KeywordTransformer
from preprocessing import AudioPreprocessor
from config import label_dict, N_MELS, FIXED_TIME_DIM

# ========================
# Model & Preprocessor
# ========================
IDX_TO_LABEL = {v: k for k, v in label_dict.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model_path = "best_finetuned_from_npy.pth"
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        st.stop()

    model = KeywordTransformer(
        image_size=(N_MELS, FIXED_TIME_DIM),
        patch_size=(40, 10),
        num_classes=len(label_dict),
        dim=160, depth=6, heads=8, mlp_dim=256, dropout=0.4
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

preprocessor = AudioPreprocessor(
    sample_rate=16000,
    n_mels=40,
    n_fft=400,
    hop_length=160,
    fixed_time=101,
    target_duration=1.0
)

# ========================
# UI
# ========================
st.markdown("""
<div style="background: linear-gradient(135deg, #0052A3, #003d7a);
            color: white; padding: 40px; text-align: center;
            border-radius: 16px; margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,82,163,0.4);">
    <h1 style="margin:0; font-size:40px; font-weight:700;">IIIT Sricity</h1>
    <h2 style="margin:15px 0 0; opacity:0.95; font-size:22px;">
        Personal Keyword Spotting System
    </h2>
    <p style="margin:10px 0 0; font-size:17px; opacity:0.9;">
        Your Voice • Your Wake Word • 100% Private
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Input Audio")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'ogg', 'webm', 'm4a']
    )

with col2:
    st.markdown("**OR** Record Live")
    recorded_audio = st.audio_input("Click mic and say your keyword")

# ========================
# Save Audio Correctly (This is the fix!)
# ========================
audio_path = None

if uploaded_file is not None:
    audio_path = f"temp_upload_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getvalue())  # ← .getvalue() for UploadedFile
    st.success(f"Uploaded: {uploaded_file.name}")

elif recorded_audio is not None:
    audio_path = "temp_recorded.wav"
    with open(audio_path, "wb") as f:
        f.write(recorded_audio.getvalue())  # ← This was missing! Now fixed
    st.success("Recording saved!")

# ========================
# Predict
# ========================
if audio_path and st.button("Predict Keyword", type="primary", use_container_width=True):
    with st.spinner("Analyzing your voice..."):
        try:
            tensor, mel_spec = preprocessor.preprocess(audio_path)
            tensor = tensor.to(device)

            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                conf, idx = torch.max(probs, dim=1)
                keyword = IDX_TO_LABEL[idx.item()]
                confidence = round(conf.item() * 100, 2)

            st.success("**Prediction Complete!**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("**Detected Keyword**", keyword.upper())
            with c2:
                st.metric("**Confidence**", f"{confidence}%")

            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax, format='%+2.0f dB')
            ax.set_title('Mel-Spectrogram')
            plt.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

else:
    st.info("Upload a file or record your voice to begin.")

st.markdown("---")
st.caption("Made with love by IIIT Sricity BTP Student")
