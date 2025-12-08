# app.py — FINAL PREMIUM VERSION (Deploy & Impress Everyone!)
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

# Your project files
from model import KeywordTransformer
from preprocessing import AudioPreprocessor
from config import label_dict, N_MELS, FIXED_TIME_DIM

# ========================
# Safe Device (No CUDA crash)
# ========================
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except:
    device = torch.device("cpu")

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
# ULTRA-PREMIUM UI — Navy & Gold with IIIT Sricity Logo
# ========================
st.set_page_config(page_title="IIIT Sricity • Keyword Spotting", page_icon="microphone", layout="centered")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 2.8rem 3.5rem;
        border-radius: 28px;
        box-shadow: 0 25px 70px rgba(15, 23, 42, 0.7);
        margin-bottom: 3.5rem;
        border: 3px solid #fbbf24;
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        gap: 2.5rem;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 15% 50%, rgba(251, 191, 36, 0.2), transparent 70%);
        pointer-events: none;
    }
    .logo-img {
        width: 130px;
        height: 130px;
        border-radius: 20px;
        border: 4px solid #fbbf24;
        box-shadow: 0 10px 30px rgba(251, 191, 36, 0.5);
        object-fit: contain;
        background: white;
        padding: 10px;
    }
    .header-text h1 {
        font-size: 48px;
        font-weight: 900;
        margin: 0 0 10px 0;
        letter-spacing: -2px;
        color: white;
        text-shadow: 0 6px 20px rgba(0,0,0,0.5);
    }
    .header-text h2 {
        font-size: 30px;
        font-weight: 600;
        margin: 0 0 12px 0;
        color: #fbbf24;
    }
    .header-text p {
        font-size: 20px;
        margin: 0;
        color: #e2e8f0;
        opacity: 0.95;
        font-weight: 400;
    }
    .input-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 28px;
        padding: 3rem 2.5rem;
        border: 1px solid rgba(251, 191, 36, 0.4);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        transition: all 0.4s;
        text-align: center;
    }
    .input-container:hover {
        transform: translateY(-12px);
        box-shadow: 0 30px 70px rgba(251, 191, 36, 0.3);
        border-color: #fbbf24;
    }
    .input-title {
        color: #fbbf24;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .input-desc {
        color: #cbd5e1;
        font-size: 16px;
        margin: 1rem 0 2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
        color: #0f172a !important;
        border: none !important;
        border-radius: 20px !important;
        height: 70px !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 35px rgba(251, 191, 36, 0.5) !important;
        transition: all 0.4s !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .stButton>button:hover {
        transform: translateY(-6px) !important;
        box-shadow: 0 20px 50px rgba(251, 191, 36, 0.7) !important;
    }
</style>
""", unsafe_allow_html=True)

# HEADER WITH LOGO
st.markdown(f"""
<div class="main-header">
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/9f/IIIT_Sri_City_Logo.png" 
         class="logo-img" alt="IIIT Sricity">
    <div class="header-text">
        <h1>Indian Institute of Information Technology, Sricity</h1>
        <h2>Personal Keyword Spotting System</h2>
        <p>BTP Project • Your Voice • Your Wake Word • 100% Private</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("<h3 style='text-align: center; color: #fbbf24; margin: 3rem 0 2rem; font-size: 34px; font-weight: 700;'>Input Audio</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="input-container">
        <div class="input-title">Upload Audio File</div>
        <div class="input-desc">Supported: WAV, MP3, OGG, WEBM, M4A</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['wav', 'mp3', 'ogg', 'webm', 'm4a'], label_visibility="collapsed")

with col2:
    st.markdown("""
    <div class="input-container">
        <div class="input-title">Record Live</div>
        <div class="input-desc">Click the mic and say your keyword</div>
    </div>
    """, unsafe_allow_html=True)
    recorded_audio = st.audio_input("", label_visibility="collapsed")

# ========================
# Save Audio
# ========================
audio_path = None

if uploaded_file is not None:
    audio_path = f"temp_upload_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success("File uploaded successfully!")

elif recorded_audio is not None:
    audio_path = "temp_recorded.wav"
    with open(audio_path, "wb") as f:
        f.write(recorded_audio.getvalue())
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

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:15px; margin:2rem 0;'>"
    "Made with passion by IIIT Sricity BTP Student | Runs on CPU • No Data Sent • 100% Private"
    "</p>",
    unsafe_allow_html=True
)
