"""
Audio Preprocessing Pipeline - Librosa Version (No Native Lib Errors)
Matches your training preprocessing exactly
"""

import torch
import numpy as np
from io import BytesIO
import librosa  # Pure Python — works on Streamlit Cloud without FFmpeg/libsndfile

class AudioPreprocessor:
    """
    Preprocess audio files for keyword spotting model
    Uses exact same parameters as your training: sample_rate=16000, n_mels=40, n_fft=400, hop_length=160
    """
    
    def __init__(
        self,
        sample_rate=16000,
        n_mels=40,
        n_fft=400,
        hop_length=160,
        fixed_time=101,  # Fixed time frames matching training
        target_duration=1.0  # 1 second audio
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fixed_time = fixed_time
        self.target_duration = target_duration
        self.target_length = int(sample_rate * target_duration)
        
        # Initialize mel spectrogram params (Librosa uses these directly)
        self.mel_params = {
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'fmin': 0.0,
            'fmax': None,
            'power': 2.0,
        }

    def load_audio(self, audio_path_or_bytes):
        """Load audio from file path or bytes using Librosa (no native libs needed)"""
        try:
            if isinstance(audio_path_or_bytes, bytes):
                # Load from bytes
                audio_data, sr = librosa.load(BytesIO(audio_path_or_bytes), sr=self.sample_rate)
                waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
            else:
                # Load from file path
                audio_data, sr = librosa.load(audio_path_or_bytes, sr=self.sample_rate)
                waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
            
            return waveform, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")

    def resample_audio(self, waveform, orig_sr):
        """Resample to 16 kHz (Librosa already handles this in load)"""
        # Since Librosa load resamples, this is a no-op now
        return waveform

    def convert_to_mono(self, waveform):
        """Convert stereo to mono (Librosa loads as mono by default)"""
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def trim_or_pad(self, waveform):
        """Trim or pad to exactly 1 second (16000 samples)"""
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
        else:
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def compute_mel_spectrogram(self, waveform):
        """Compute mel spectrogram and convert to log scale using Librosa"""
        # Convert waveform to numpy for Librosa
        audio_np = waveform.squeeze(0).numpy()
        
        # Compute mel spectrogram with exact training params
        mel_spec = librosa.feature.melspectrogram(
            y=audio_np,
            sr=self.sample_rate,
            **self.mel_params
        )
        # Power to amplitude (power=2.0 → sqrt for amplitude)
        mel_spec = np.sqrt(mel_spec)
        # Log transform
        mel_spec_db = np.log(mel_spec + 1e-6)
        
        return mel_spec_db

    def pad_or_crop_time(self, mel_spec_db):
        """Pad or crop to FIXED_TIME = 101 frames"""
        if mel_spec_db.shape[1] < self.fixed_time:
            pad_w = self.fixed_time - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_w)), mode="constant")
        elif mel_spec_db.shape[1] > self.fixed_time:
            mel_spec_db = mel_spec_db[:, :self.fixed_time]
        
        return mel_spec_db

    def to_tensor(self, mel_spec_db):
        """Convert to PyTorch tensor with shape (1, 1, 40, 101)"""
        tensor = torch.from_numpy(mel_spec_db).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        return tensor

    def preprocess(self, audio_path_or_bytes):
        """
        Complete preprocessing pipeline matching your training exactly
        
        Process:
        1. Load audio with Librosa (resamples to 16kHz, mono)
        2. Trim/pad to 1 second (16000 samples)
        3. Compute mel-spectrogram (n_fft=400, hop_length=160, n_mels=40, power=2.0)
        4. Log transform: log(sqrt(mel) + 1e-6)
        5. Pad/crop to 101 time frames
        6. Convert to tensor (1, 1, 40, 101)
        
        Returns:
            tuple: (tensor, mel_spec_np) - tensor for model, mel_spec for visualization
        """
        # Step 1: Load audio (Librosa handles resampling/mono)
        waveform, sr = self.load_audio(audio_path_or_bytes)
        
        # Step 2: Trim/pad to 1 second
        waveform = self.trim_or_pad(waveform)
        
        # Step 3: Compute mel-spectrogram
        mel_spec_db = self.compute_mel_spectrogram(waveform)
        
        # Step 4: Pad/crop to fixed time
        mel_spec_db = self.pad_or_crop_time(mel_spec_db)
        
        # Step 5: Convert to tensor
        tensor = self.to_tensor(mel_spec_db)
        
        return tensor, mel_spec_db
