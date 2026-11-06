"""Quick quality check for recorded clips: duration, peak level, speech length."""
import sys

import librosa
import numpy as np

from config import DATA_DIR, SAMPLE_RATE
from preprocessing import load_audio

name = sys.argv[1] if len(sys.argv) > 1 else "basmala"
files = sorted((DATA_DIR / "team_voice").glob(f"*_{name}_*.wav"))
print(f"{len(files)} files for '{name}'")

bad = []
speech_secs = []
for f in files:
    audio = load_audio(f, SAMPLE_RATE)
    peak = float(np.max(np.abs(audio)))
    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    speech = len(trimmed) / SAMPLE_RATE
    speech_secs.append(speech)
    # flag clips that are very quiet or where almost no speech was detected
    if peak < 0.05 or speech < 0.15:
        bad.append((f.name, round(peak, 3), round(speech, 2)))

print(f"Speech length (sec): min={min(speech_secs):.2f} "
      f"mean={np.mean(speech_secs):.2f} max={max(speech_secs):.2f}")
print(f"Suspicious clips: {len(bad)}")
for b in bad:
    print("  ", b)