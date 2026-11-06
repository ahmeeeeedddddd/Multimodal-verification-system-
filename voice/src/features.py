"""Voice feature extraction (the interface used by fusion and the pipeline).

Feature vector layout (fixed length, 1-D float32):
    [ MFCC mean (13) | MFCC std (13) | ZCR mean, ZCR std | RMS mean, RMS std ]
    = 13 + 13 + 2 + 2 = 30 dimensions

The same function is used for training, live microphone input and fusion,
so every consumer sees exactly the same features.
"""
import numpy as np
import librosa

from config import SAMPLE_RATE, N_MFCC

# Frame settings (25 ms window, 10 ms hop at 8 kHz)
N_FFT = 256
HOP_LENGTH = 80

VOICE_FEATURE_DIM = N_MFCC * 2 + 2 + 2   # 30


def _frame_stats(x: np.ndarray) -> np.ndarray:
    """Return [mean, std] of a (1, n_frames) or (n_frames,) array."""
    x = np.asarray(x).ravel()
    return np.array([x.mean(), x.std()], dtype=np.float32)


def extract_voice_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract a fixed-length feature vector from PREPROCESSED audio.

    Args:
        audio: 1-D float array, already trimmed / normalized / fixed length
               (output of preprocessing.preprocess).
        sr:    sample rate of the audio.

    Returns:
        1-D float32 array of length VOICE_FEATURE_DIM.
    """
    audio = np.asarray(audio, dtype=np.float32)

    # MFCC: (n_mfcc, n_frames) -> mean and std over time for each coefficient
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=40,
    )
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # Zero-crossing rate (how often the signal changes sign)
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=N_FFT, hop_length=HOP_LENGTH
    )

    # RMS energy per frame
    rms = librosa.feature.rms(
        y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH
    )

    features = np.concatenate([
        mfcc_mean, mfcc_std, _frame_stats(zcr), _frame_stats(rms),
    ]).astype(np.float32)

    assert features.shape[0] == VOICE_FEATURE_DIM
    return features