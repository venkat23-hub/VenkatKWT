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
# BEAUTIFUL & PROFESSIONAL UI
# ========================

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0052A3, #003d7a);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 40px rgba(0, 82, 163, 0.4);
        margin-bottom: 2.5rem;
        border: 3px solid rgba(255,255,255,0.1);
    }
    .main-header h1 {
        font-size: 46px;
        font-weight: 800;
        margin: 0 0 12px 0;
        letter-spacing: -1px;
        text-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .main-header h2 {
        font-size: 26px;
        font-weight: 500;
        margin: 0 0 10px 0;
        opacity: 0.95;
    }
    .main-header p {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 16px;
    }
    .input-box {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        height: 100%;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0052A3, #003d7a) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        height: 56px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 20px rgba(0,82,163,0.3) !important;
        transition: all 0.3s !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(0,82,163,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>Indian Institute of Information Technology, Sricity</h1>
    <h2>Personal Keyword Spotting System</h2>
    <p>BTP Project • Your Voice-Powered Wake Word Detector</p>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("<h3 style='text-align: center; color: #333; margin-bottom: 2rem;'>Input Audio</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="input-box">
        <h4 style="text-align:center; color:#0052A3; margin-top:0;">Upload Audio File</h4>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "",  # Empty label for clean look
        type=['wav', 'mp3', 'ogg', 'webm', 'm4a'],
        label_visibility="collapsed"
    )

with col2:
    st.markdown("""
    <div class="input-box">
        <h4 style="text-align:center; color:#0052A3; margin-top:0;">Record Live</h4>
        <p style="text-align:center; color:#666; font-size:14px; margin-bottom:20px;">
            Click the mic and say your keyword
        </p>
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
