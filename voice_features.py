import os
import glob
import numpy as np
import librosa

def extract_voice_features(y, sr=16000, n_mfcc=13):
    if len(y) < sr // 4:
        y = np.pad(y, (0, sr // 4 - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    zcr_mean = np.mean(zcr, keepdims=True).ravel()
    zcr_std = np.std(zcr, keepdims=True).ravel()
    rms_mean = np.mean(rms, keepdims=True).ravel()
    rms_std = np.std(rms, keepdims=True).ravel()

    return np.hstack([mfcc_mean, mfcc_std, zcr_mean, zcr_std, rms_mean, rms_std])

def load_audio_file(file_path, sr=16000, duration=2.0):
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    return y

def get_voice_dataset(data_dir=None, n_speakers=40, samples_per_speaker=10, sr=16000):
    if data_dir and os.path.exists(data_dir):
        speaker_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if len(speaker_dirs) > 0:
            features, labels = [], []
            for spk_idx, spk in enumerate(speaker_dirs):
                files = glob.glob(os.path.join(data_dir, spk, "*.wav"))
                for f in files:
                    y = load_audio_file(f, sr=sr)
                    feat = extract_voice_features(y, sr=sr)
                    features.append(feat)
                    labels.append(spk_idx)
            return np.array(features), np.array(labels)

    features, labels = [], []
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    np.random.seed(42)

    for spk_id in range(n_speakers):
        base_f0 = 100.0 + (spk_id * 3.5)
        f1 = 500.0 + (spk_id * 10.0)
        f2 = 1500.0 + (spk_id * 15.0)

        for _ in range(samples_per_speaker):
            f0 = base_f0 + np.random.uniform(-4, 4)
            signal = 0.5 * np.sin(2 * np.pi * f0 * t)
            signal += 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
            signal += 0.2 * np.sin(2 * np.pi * f1 * t)
            signal += 0.1 * np.sin(2 * np.pi * f2 * t)
            noise = np.random.normal(0, 0.02, len(t))
            y = (signal + noise).astype(np.float32)

            feat = extract_voice_features(y, sr=sr)
            features.append(feat)
            labels.append(spk_id)

    return np.array(features), np.array(labels)
