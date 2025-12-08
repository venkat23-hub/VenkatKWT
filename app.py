# app.py — FINAL 100% RESPONSIVE (Mobile + Desktop)
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
# Device
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
preprocessor = AudioPreprocessor(sample_rate=16000, n_mels=40, n_fft=400,
                                hop_length=160, fixed_time=101, target_duration=1.0)

# ========================
# FULLY RESPONSIVE ULTRA-PREMIUM UI
# ========================
st.set_page_config(page_title="IIIT Sricity • Wake Word", page_icon="microphone", layout="centered")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.6);
        margin: 1rem 0 2.5rem;
        border: 2px solid #fbbf24;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        justify-content: center;
        text-align: center;
    }
    @media (max-width: 768px) {
        .main-header {
            flex-direction: column;
            padding: 2rem 1rem;
            gap: 1rem;
        }
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 15% 50%, rgba(251, 191, 36, 0.2), transparent 70%);
        pointer-events: none;
    }
    .logo-img {
        width: 100px;
        height: 100px;
        border-radius: 16px;
        border: 4px solid #fbbf24;
        box-shadow: 0 8px 25px rgba(251, 191, 36, 0.5);
        object-fit: contain;
        background: white;
        padding: 8px;
        flex-shrink: 0;
    }
    @media (max-width: 768px) {
        .logo-img { width: 80px; height: 80px; }
    }
    .header-text h1 {
        font-size: 36px;
        font-weight: 900;
        margin: 0;
        letter-spacing: -1.5px;
        color: white;
        text-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    @media (max-width: 768px) {
        .header-text h1 { font-size: 28px; }
    }
    .header-text h2 {
        font-size: 24px;
        font-weight: 600;
        margin: 8px 0;
        color: #fbbf24;
    }
    @media (max-width: 768px) {
        .header-text h2 { font-size: 20px; }
    }
    .header-text p {
        font-size: 16px;
        margin: 0;
        color: #e2e8f0;
        opacity: 0.9;
    }
    .input-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin: 2rem 0;
    }
    @media (max-width: 768px) {
        .input-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
    }
    .input-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        border: 1px solid rgba(251, 191, 36, 0.4);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        text-align: center;
        transition: all 0.4s;
    }
    .input-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 60px rgba(251, 191, 36, 0.3);
        border-color: #fbbf24;
    }
    .input-title {
        color: #fbbf24;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    @media (max-width: 768px) {
        .input-title { font-size: 22px; }
    }
    .input-desc {
        color: #cbd5e1;
        font-size: 15px;
        margin: 1rem 0 2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
        color: #0f172a !important;
        border: none !important;
        border-radius: 18px !important;
        height: 64px !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 30px rgba(251, 191, 36, 0.5) !important;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header with Logo (Responsive)
st.markdown(f"""
<div class="main-header">
    <img src="https://upload.wikimedia.org/wikipedia/en/4/49/IIIT_Sri_City_Logo.png" 
         class="logo-img" alt="IIIT Sricity">
    <div class="header-text">
        <h1>Indian Institute of Information Technology, Sricity</h1>
        <h2>Personal Keyword Spotting System</h2>
        <p>BTP Project • Your Voice • Your Wake Word • 100% Private</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Responsive Input Cards
st.markdown("<h3 style='text-align: center; color: #fbbf24; margin: 2rem 0 1.5rem; font-size: 30px; font-weight: 700;'>Input Audio</h3>", unsafe_allow_html=True)

st.markdown('<div class="input-grid">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="input-card">
        <div class="input-title">Upload Audio File</div>
        <div class="input-desc">WAV • MP3 • OGG • WEBM • M4A</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['wav','mp3','ogg','webm','m4a'], label_visibility="collapsed")

with col2:
    st.markdown("""
    <div class="input-card">
        <div class="input-title">Record Live</div>
        <div class="input-desc">Click mic and say your keyword</div>
    </div>
    """, unsafe_allow_html=True)
    recorded_audio = st.audio_input("", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ========================
# Save & Predict (Same as before)
# ========================
audio_path = None
if uploaded_file:
    audio_path = f"temp_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success("File uploaded!")

elif recorded_audio:
    audio_path = "temp_recorded.wav"
    with open(audio_path, "wb") as f:
        f.write(recorded_audio.getvalue())
    st.success("Recording saved!")

if audio_path and st.button("Predict Keyword", type="primary", use_container_width=True):
    with st.spinner("Analyzing..."):
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
            with c1: st.metric("**Keyword**", keyword.upper())
            with c2: st.metric("**Confidence**", f"{confidence}%")

            fig, ax = plt.subplots(figsize=(10,4))
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
    st.info("Upload or record audio to begin")

st.markdown("---")
st.caption("Made by IIIT Sricity BTP Students and logo is taken from wikipedia, Thank you")
