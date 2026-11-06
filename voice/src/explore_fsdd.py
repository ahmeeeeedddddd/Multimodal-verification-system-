"""Quick exploration of the FSDD dataset: counts, speakers, durations."""
from collections import Counter

import numpy as np
import soundfile as sf

from config import FSDD_DIR

# Collect all wav files (sorted so the output is deterministic)
files = sorted(FSDD_DIR.glob("*.wav"))
print(f"Total files: {len(files)}")

speakers = Counter()
digits = Counter()
durations = []
sample_rates = set()

for f in files:
    # File name format: {digit}_{speaker}_{index}.wav
    digit, speaker, _ = f.stem.split("_")
    speakers[speaker] += 1
    digits[digit] += 1

    audio, sr = sf.read(f)
    sample_rates.add(sr)
    durations.append(len(audio) / sr)

print(f"\nSpeakers ({len(speakers)}):")
for name, count in sorted(speakers.items()):
    print(f"  {name:10s}: {count} files")

print(f"\nDigits: {dict(sorted(digits.items()))}")
print(f"\nSample rate(s): {sample_rates}")
print(
    f"Duration (sec): min={min(durations):.2f}  "
    f"max={max(durations):.2f}  mean={np.mean(durations):.2f}"
)