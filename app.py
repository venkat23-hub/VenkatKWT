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
# ULTRA-PREMIUM UI — Navy & Gold Theme (Final)
# ========================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 4rem 2rem;
        border-radius: 24px;
        text-align: center;
        color: white;
        box-shadow: 0 25px 60px rgba(15, 23, 42, 0.6);
        margin-bottom: 3.5rem;
        border: 3px solid #fbbf24;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 30% 30%, rgba(251, 191, 36, 0.15), transparent 60%);
        pointer-events: none;
    }
    .main-header h1 {
        font-size: 52px;
        font-weight: 900;
        margin: 0 0 12px 0;
        letter-spacing: -2px;
        text-shadow: 0 6px 20px rgba(0,0,0,0.5);
    }
    .main-header h2 {
        font-size: 30px;
        font-weight: 600;
        margin: 0 0 15px 0;
        color: #fbbf24;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 20px;
        margin: 0;
        opacity: 0.92;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    .input-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        border: 1px solid rgba(251, 191, 36, 0.3);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        transition: all 0.4s;
    }
    .input-container:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 60px rgba(0,0,0,0.4);
        border-color: #fbbf24;
    }
    .input-title {
        color: #fbbf24;
        text-align: center;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-shadow: 0 3px 10px rgba(0,0,0,0.4);
    }
    .input-desc {
        color: #e2e8f0;
        text-align: center;
        font-size: 16px;
        margin: 1rem 0 2rem;
        opacity: 0.9;
    }

    .stButton>button {
        background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
        color: #0f172a !important;
        border: none !important;
        border-radius: 18px !important;
        height: 70px !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 30px rgba(251, 191, 36, 0.5) !important;
        transition: all 0.4s !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-6px) !important;
        box-shadow: 0 20px 50px rgba(251, 191, 36, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# Premium Header
st.markdown("""
<div class="main-header">
    <h1>Indian Institute of Information Technology, Sricity</h1>
    <h2>Personal Keyword Spotting System</h2>
    <p>BTP Project</p>
</div>
""", unsafe_allow_html=True)

# Single Premium Input Section (Exactly like your mockup)
st.markdown("""
<h3 style='text-align: center; color: #fbbf24; margin-bottom: 3rem; font-size: 32px; font-weight: 700;'>
    Input Audio
</h3>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="input-container">
        <div class="input-title">Upload Audio File</div>
        <div class="input-desc">Drag & drop or click to browse<br>WAV, MP3, OGG, WEBM, M4A</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['wav', 'mp3', 'ogg', 'webm', 'm4a'], label_visibility="collapsed")

with col2:
    st.markdown("""
    <div class="input-container">
        <div class="input-title">Record Live</div>
        <div class="input-desc">Click the microphone and say your keyword</div>
    </div>
    """, unsafe_allow_html=True)
    recorded_audio = st.audio_input("", label_visibility="collapsed")
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
