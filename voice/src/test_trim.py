# voice/src/test_trim.py
import librosa
import numpy as np
from config import DATA_DIR, SAMPLE_RATE
from preprocessing import load_audio

f = sorted((DATA_DIR / "team_voice").glob("*_basmala_*.wav"))[0]
audio = load_audio(f, SAMPLE_RATE)
for db in (25, 20, 15, 10):
    t, _ = librosa.effects.trim(audio, top_db=db)
    print(f"top_db={db}: speech = {len(t)/SAMPLE_RATE:.2f} s")