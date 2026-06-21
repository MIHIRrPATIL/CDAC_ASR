import numpy as np
import torch
import scipy.fftpack
from pathlib import Path

class AudioPreprocessor:
    def __init__(self, sr=16000):
        self.sr = sr
        # Load Silero VAD
        import os
        hub_dir = torch.hub.get_dir()
        local_repo_path = os.path.join(hub_dir, "snakers4_silero-vad_master")
        
        if os.path.exists(local_repo_path):
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir=local_repo_path,
                source='local',
                model='silero_vad',
                trust_repo=True,
                onnx=False
            )
        else:
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                source='github',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
        (self.get_speech_timestamps, _, self.read_audio, self.vad_iterator, self.collect_chunks) = self.vad_utils

    def apply_fft_filter(self, audio, noise_reduction_factor=0.02):
        """
        Robust Spectral Subtraction using FFT to reduce background hiss.
        Estimates the noise floor from the quietest 10% region of the audio (minimum statistics).
        """
        # Convert to frequency domain
        audio_fft = scipy.fftpack.fft(audio)
        audio_mag = np.abs(audio_fft)
        audio_phase = np.angle(audio_fft)

        # Safely estimate noise spectrum from the quietest chunk to prevent speech distortion
        n_chunks = 10
        chunk_size = len(audio) // n_chunks
        if chunk_size > 100:
            min_rms = float('inf')
            min_chunk_idx = 0
            for k in range(n_chunks):
                chunk = audio[k * chunk_size : (k + 1) * chunk_size]
                rms = np.sqrt(np.mean(chunk**2))
                if rms < min_rms:
                    min_rms = rms
                    min_chunk_idx = k
            
            # Extract noise from the quietest window
            noise_chunk = audio[min_chunk_idx * chunk_size : (min_chunk_idx + 1) * chunk_size]
            noise_fft = scipy.fftpack.fft(noise_chunk, n=len(audio))
            noise_estimate = np.abs(noise_fft)
        else:
            noise_estimate = np.mean(audio_mag) * 0.05
        
        # Subtract noise
        audio_mag_cleaned = np.maximum(audio_mag - (noise_estimate * noise_reduction_factor), 0)
        
        # Convert back to time domain
        audio_cleaned_fft = audio_mag_cleaned * np.exp(1j * audio_phase)
        audio_cleaned = scipy.fftpack.ifft(audio_cleaned_fft).real
        
        return audio_cleaned.astype(np.float32)

    def trim_silence_vad(self, audio):
        """
        Uses Silero VAD to find speech boundaries and crop the audio.
        """
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio)
        else:
            audio_tensor = audio

        # Ensure audio is 1D
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.squeeze()

        speech_timestamps = self.get_speech_timestamps(
            audio_tensor, 
            self.vad_model, 
            sampling_rate=self.sr
        )
        
        if not speech_timestamps:
            return audio # Return original if no speech found
            
        # Collect all speech chunks
        cleaned_audio = self.collect_chunks(speech_timestamps, audio_tensor)
        return cleaned_audio.numpy()

    def preprocess(self, audio, apply_filter=True, apply_vad=True):
        """
        Full pipeline: Filter -> VAD Trim
        """
        processed_audio = audio
        
        if apply_filter:
            processed_audio = self.apply_fft_filter(processed_audio)
            
        if apply_vad:
            processed_audio = self.trim_silence_vad(processed_audio)
            
        return processed_audio

if __name__ == "__main__":
    # Test with dummy data
    preprocessor = AudioPreprocessor()
    dummy_audio = np.random.uniform(-1, 1, 16000 * 2) # 2 seconds of noise
    out = preprocessor.preprocess(dummy_audio)
    print(f"Original size: {len(dummy_audio)}, Processed size: {len(out)}")
