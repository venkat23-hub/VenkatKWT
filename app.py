import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import os

# Your project files (same as Flask)
from model import KeywordTransformer
from preprocessing import AudioPreprocessor
from config import label_dict, N_MELS, FIXED_TIME_DIM

# === SAME AS FLASK ===
IDX_TO_LABEL = {v: k for k, v in label_dict.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model & preprocessor (cached like Flask global)
@st.cache_resource
def load_model_and_preprocessor():
    model_path = "best_finetuned_from_npy.pth"
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
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

    preprocessor = AudioPreprocessor(
        sample_rate=16000, n_mels=40, n_fft=400,
        hop_length=160, fixed_time=101, target_duration=1.0
    )

    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# === BEAUTIFUL TITLE (Same as Flask) ===
st.markdown("""
<div style="background: linear-gradient(135deg, #0052A3, #003d7a); 
            color: white; padding: 35px; text-align: center; 
            border-radius: 16px; margin-bottom: 30px; 
            box-shadow: 0 8px 32px rgba(0,82,163,0.3);">
    <h1 style="margin:0; font-size:36px; font-weight:700;">IIIT Sricity</h1>
    <h2 style="margin:12px 0 0; opacity:0.95; font-size:20px;">
        Personal Keyword Spotting System • Your Voice, Your Wake Word
    </h2>
</div>
""", unsafe_allow_html=True)

st.markdown("### Choose Input Method")

col1, col2 = st.columns(2)

# === UPLOAD (Same as Flask /upload route) ===
with col1:
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'ogg', 'webm', 'm4a'],
        help="Max 10MB"
    )

# === RECORD (FIXED: Use st.audio_input — new official widget) ===
with col2:
    st.markdown("**OR** Record Live")
    recorded_audio = st.audio_input("Click and say your keyword")  # ← FIXED: st.audio_input

# Get audio path
audio_path = None
if uploaded_file:
    audio_path = f"temp_upload_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
elif recorded_audio:
    audio_path = "temp_recorded.wav"
    with open(audio_path, "wb") as f:
        f.write(recorded_audio)

# === PREDICT BUTTON (Same as Flask /predict) ===
if audio_path:
    if st.button("Predict Keyword", type="primary", use_container_width=True):
        with st.spinner("Processing audio..."):
            try:
                # SAME preprocessing as Flask
                tensor, mel_spec = preprocessor.preprocess(audio_path)
                tensor = tensor.to(device)

                with torch.no_grad():
                    output = model(tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                    keyword = IDX_TO_LABEL[idx.item()]
                    confidence = round(conf.item() * 100, 2)

                # SUCCESS RESULT
                st.success("**Prediction Complete!**")
                colA, colB = st.columns(2)
                with colA:
                    st.metric("Detected Keyword", keyword.upper())
                with colB:
                    st.metric("Confidence", f"{confidence}%")

                # SPECTROGRAM (Same as Flask)
                fig, ax = plt.subplots(figsize=(10, 4))
                img = ax.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(img, ax=ax, format='%+2.0f dB')
                plt.title('Mel-Spectrogram')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

else:
    st.info("Please upload an audio file or record your voice to predict.")

# Footer
st.markdown("---")
st.markdown("**Your personal wake word detector** • Trained only on your voice • 100% private")