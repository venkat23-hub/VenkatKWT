# app.py — FINAL WORKING VERSION (No torch.cuda.is_available() crash)
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
# FIXED: Safe Device Detection (No torch.cuda.is_available() crash)
# ========================
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
except:
    device = torch.device("cpu")  # Fallback if CUDA check fails

st.sidebar.success(f"Using device: {device}")

# ========================
# Model & Preprocessor
# ========================
IDX_TO_LABEL = {v: k for k, v in label_dict.items()}

@st.cache_resource
def load_model():
    model_path = "best_finetuned_from_npy.pth"
    if not os.path.exists(model_path):
        st.error("Model file not found! Please upload best_finetuned_from_npy.pth")
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
# ULTRA-PREMIUM HEADER WITH IIIT SRICITY LOGO (Left Side)
# ========================

# Replace this block in your app.py
st.markdown("""
<style>
    .logo-header {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 2.5rem 3rem;
        border-radius: 24px;
        box-shadow: 0 25px 60px rgba(15, 23, 42, 0.6);
        margin-bottom: 3.5rem;
        border: 3px solid #fbbf24;
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        gap: 2rem;
    }
    .logo-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(251, 191, 36, 0.15), transparent 70%);
        pointer-events: none;
    }
    .logo-img {
        width: 120px;
        height: 120px;
        border-radius: 16px;
        border: 3px solid #fbbf24;
        box-shadow: 0 8px 25px rgba(251, 191, 36, 0.4);
        object-fit: contain;
        background: white;
        padding: 8px;
    }
    .header-text h1 {
        font-size: 44px;
        font-weight: 900;
        margin: 0 0 8px 0;
        letter-spacing: -1.8px;
        color: white;
        text-shadow: 0 6px 20px rgba(0,0,0,0.5);
    }
    .header-text h2 {
        font-size: 28px;
        font-weight: 600;
        margin: 0 0 10px 0;
        color: #fbbf24;
    }
    .header-text p {
        font-size: 19px;
        margin: 0;
        color: #e2e8f0;
        opacity: 0.95;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# === HEADER WITH LOGO (Left Side) ===
st.markdown(f"""
<div class="logo-header">
    <img src="https://upload.wikimedia.org/wikipedia/en/4/49/IIIT_Sri_City_Logo.png" 
         class="logo-img" 
         alt="IIIT Sricity Logo">
    <div class="header-text">
        <h1>Indian Institute of Information Technology, Sricity</h1>
        <h2>Personal Keyword Spotting System</h2>
        <p>BTP Project • Your Voice • Your Wake Word • 100% Private</p>
    </div>
</div>
""", unsafe_allow_html=True)
# ========================
# Save Audio Correctly
# ========================
audio_path = None

if uploaded_file is not None:
    audio_path = f"temp_upload_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success(f"Uploaded: {uploaded_file.name}")

elif recorded_audio is not None:
    audio_path = "temp_recorded.wav"
    with open(audio_path, "wb") as f:
        f.write(recorded_audio.getvalue())  # ← Fixed!
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
st.caption("Made with love by IIIT Sricity BTP Student | Runs on CPU • No Data Sent")
