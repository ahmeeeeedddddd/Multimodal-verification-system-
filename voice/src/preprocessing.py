"""Audio preprocessing: load, trim silence, fix length, normalize.

The same pipeline is used for dataset files and for live microphone
audio, so training and inference always see identical inputs.
"""
import numpy as np
import librosa
import soundfile as sf

from config import SAMPLE_RATE, N_SAMPLES, TOP_DB


def load_audio(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a wav file as mono float32 at the target sample rate."""
    audio, file_sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:                       # stereo -> mono
        audio = audio.mean(axis=1)
    if file_sr != sr:                        # resample if needed
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio.astype(np.float32)


def trim_silence(audio: np.ndarray, top_db: float = TOP_DB) -> np.ndarray:
    """Remove leading/trailing silence. Keep the original if all is silent."""
    if audio.size == 0:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if trimmed.size > 0 else audio


def fix_length(audio: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Zero-pad (at the end) or crop so the clip has exactly n_samples."""
    if audio.size >= n_samples:
        return audio[:n_samples]
    return np.pad(audio, (0, n_samples - audio.size))


def normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Remove DC offset and peak-normalize to the range [-1, 1]."""
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    return (audio / (peak + eps)).astype(np.float32)


def preprocess(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Full pipeline: trim silence -> normalize -> fix length.

    Normalizing BEFORE padding keeps the zero padding exactly zero,
    so the padded part does not distort the energy features.
    """
    audio = trim_silence(audio)
    audio = normalize(audio)
    audio = fix_length(audio, int(sr * (N_SAMPLES / SAMPLE_RATE)))
    return audio


def preprocess_file(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convenience wrapper: load a file and run the full pipeline."""
    return preprocess(load_audio(path, sr), sr)