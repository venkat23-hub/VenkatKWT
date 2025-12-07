"""
Audio Preprocessing Pipeline - Exact Same as Your Testing Code
Matches your training preprocessing exactly
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from io import BytesIO
import soundfile as sf


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
        
        # Initialize mel spectrogram transform with exact training parameters
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=None,
            power=2.0,
        )

    def load_audio(self, audio_path_or_bytes):
        """Load audio from file path or bytes"""
        if isinstance(audio_path_or_bytes, bytes):
            # Load from bytes
            audio_data, sr = sf.read(BytesIO(audio_path_or_bytes))
            waveform = torch.from_numpy(audio_data).float()
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
        else:
            # Load from file path
            waveform, sr = torchaudio.load(audio_path_or_bytes)
        
        return waveform, sr

    def resample_audio(self, waveform, orig_sr):
        """Resample to 16 kHz"""
        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.sample_rate)
        return waveform

    def convert_to_mono(self, waveform):
        """Convert stereo to mono"""
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
        """Compute mel spectrogram and convert to log scale"""
        mel_spec = self.mel_transform(waveform).squeeze(0)
        mel_spec_db = torch.log(mel_spec + 1e-6)
        return mel_spec_db.numpy()

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
        1. Load audio
        2. Resample to 16 kHz
        3. Convert stereo to mono
        4. Trim/pad to 1 second (16000 samples)
        5. Compute mel-spectrogram (n_fft=400, hop_length=160, n_mels=40, power=2.0)
        6. Log transform: log(mel + 1e-6)
        7. Pad/crop to 101 time frames
        8. Convert to tensor (1, 1, 40, 101)
        
        Returns:
            tuple: (tensor, mel_spec_np) - tensor for model, mel_spec for visualization
        """
        # Step 1: Load audio
        waveform, sr = self.load_audio(audio_path_or_bytes)
        
        # Step 2: Resample to 16 kHz
        waveform = self.resample_audio(waveform, sr)
        
        # Step 3: Convert to mono
        waveform = self.convert_to_mono(waveform)
        
        # Step 4: Trim/pad to 1 second
        waveform = self.trim_or_pad(waveform)
        
        # Step 5: Compute mel-spectrogram
        mel_spec_db = self.compute_mel_spectrogram(waveform)
        
        # Step 6: (Already done in compute_mel_spectrogram with log transform)
        
        # Step 7: Pad/crop to fixed time
        mel_spec_db = self.pad_or_crop_time(mel_spec_db)
        
        # Step 8: Convert to tensor
        tensor = self.to_tensor(mel_spec_db)
        
        return tensor, mel_spec_db
