"""Sanity check for the voice feature extractor."""
import numpy as np

from config import FSDD_DIR
from preprocessing import preprocess_file
from features import extract_voice_features, VOICE_FEATURE_DIM

for name in ["7_george_32.wav", "7_jackson_32.wav", "3_theo_5.wav"]:
    audio = preprocess_file(FSDD_DIR / name)
    feats = extract_voice_features(audio)
    print(f"{name:20s} shape={feats.shape} dtype={feats.dtype} "
          f"nan={np.isnan(feats).any()}")
    assert feats.shape == (VOICE_FEATURE_DIM,)
    assert not np.isnan(feats).any()

print(f"\nVOICE_FEATURE_DIM = {VOICE_FEATURE_DIM}")
print("First 5 features of last file:", np.round(feats[:5], 3))
print("All checks passed.")