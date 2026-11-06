"""Sanity check + visual comparison for the preprocessing pipeline."""
import matplotlib.pyplot as plt
import numpy as np

from config import FSDD_DIR, N_SAMPLES
from preprocessing import load_audio, preprocess

sample = FSDD_DIR / "7_george_32.wav"
raw = load_audio(sample)
clean = preprocess(raw)

print(f"Raw length   : {len(raw)} samples")
print(f"Clean length : {len(clean)} samples (expected {N_SAMPLES})")
print(f"Clean range  : [{clean.min():.3f}, {clean.max():.3f}]")
assert len(clean) == N_SAMPLES, "Length fix failed"
assert np.max(np.abs(clean)) <= 1.0 + 1e-6, "Normalization failed"
print("All checks passed.")

fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].plot(raw)
axes[0].set_title("Raw")
axes[1].plot(clean)
axes[1].set_title("Preprocessed (trim + normalize + fix length)")
plt.tight_layout()
plt.show()