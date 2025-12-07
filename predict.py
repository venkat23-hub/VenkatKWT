import os
import torch
import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt

from model import KeywordTransformer
from config import label_dict, N_MELS, FIXED_TIME_DIM

# === Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# === Load Model ===
MODEL_PATH = "D:\WebApp_KWT_BTP\best_model.pth"

model = KeywordTransformer(
    image_size=(40, 101),   # Must match training
    patch_size=(40, 10),
    num_classes=len(label_dict),
    dim=160,
    depth=6,
    heads=8,
    mlp_dim=256,
    dropout=0.4
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"[INFO] Loaded model from {MODEL_PATH}")

# === Helper: Extract Mel-Spectrogram ===
def extract_spectrogram(audio, sr=16000, n_mels=N_MELS, fixed_size=FIXED_TIME_DIM):
    spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, fmax=sr//2
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Pad or crop to fixed time dimension
    if spectrogram.shape[1] < fixed_size:
        pad_width = fixed_size - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0,0), (0,pad_width)), mode="constant")
    elif spectrogram.shape[1] > fixed_size:
        spectrogram = spectrogram[:, :fixed_size]

    return spectrogram

# === Record Audio from Microphone ===
def record_and_predict(duration=2, sr=16000):
    print(f"\n🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()

    # Normalize
    audio = audio / np.max(np.abs(audio))

    # Extract spectrogram
    spectrogram = extract_spectrogram(audio, sr=sr)

    # Prepare tensor
    spectrogram_tensor = torch.tensor(spectrogram).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    # Prediction
    with torch.no_grad():
        outputs = model(spectrogram_tensor)
        _, predicted = torch.max(outputs, 1)

    pred_idx = predicted.item()
    pred_word = list(label_dict.keys())[list(label_dict.values()).index(pred_idx)]

    print(f"✅ Predicted Word: {pred_word}")
    return pred_word

# === Run Interactive Loop ===
while True:
    cmd = input("\nPress ENTER to record (or type 'q' to quit): ")
    if cmd.lower() == "q":
        break
    predicted_word = record_and_predict(duration=2)
