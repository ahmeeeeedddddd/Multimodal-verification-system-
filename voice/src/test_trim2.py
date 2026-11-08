"""Compare speech length after trimming at several top_db values."""
import random

import librosa
import numpy as np

from config import DATA_DIR, FSDD_DIR, SAMPLE_RATE
from preprocessing import load_audio

random.seed(0)
mine = random.sample(sorted((DATA_DIR / "team_voice").glob("*_basmala_*.wav")), 30)
fsdd = random.sample(sorted(FSDD_DIR.glob("*.wav")), 30)

print("top_db | basmala mean (min-max) | FSDD mean (min-max)")
for db in (25, 20, 18, 15, 12):
    row = []
    for files in (mine, fsdd):
        lens = [len(librosa.effects.trim(load_audio(f, SAMPLE_RATE), top_db=db)[0])
                / SAMPLE_RATE for f in files]
        row.append(f"{np.mean(lens):.2f} ({min(lens):.2f}-{max(lens):.2f})")
    print(f"{db:6d} | {row[0]:24s} | {row[1]}")