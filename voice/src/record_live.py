"""Record a LIVE test session (e.g. on the demo laptop) into data/live_test.

Files use the FSDD naming (digit_name_index.wav) and are kept separate from
data/team_voice so they can be used as an honest validation/test session.

Usage:
    python record_live.py --name basmala --reps 10
"""
import argparse
import random
import time

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import DATA_DIR, SAMPLE_RATE

OUT_DIR = DATA_DIR / "live_test"
REC_SR = 16000
REC_SECONDS = 1.5
WORDS = ["zero", "one", "two", "three", "four",
         "five", "six", "seven", "eight", "nine"]


def next_index(name: str, digit: int) -> int:
    """Next free index so repeated runs never overwrite old files."""
    idx = [int(f.stem.split("_")[2])
           for f in OUT_DIR.glob(f"{digit}_{name}_*.wav")]
    return max(idx) + 1 if idx else 0


def record_one() -> np.ndarray:
    audio = sd.rec(int(REC_SECONDS * REC_SR), samplerate=REC_SR,
                   channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def is_good(audio: np.ndarray) -> bool:
    peak = float(np.max(np.abs(audio)))
    if peak < 0.02:
        print("   too quiet - speak louder. Retrying...")
        return False
    clipped = int(np.sum(np.abs(audio) >= 0.999))
    if clipped > 20:
        print(f"   clipped ({clipped} samples) - lower mic volume. Retrying...")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--reps", type=int, default=10,
                        help="NEW repetitions per digit in this run")
    args = parser.parse_args()

    if "_" in args.name or not args.name.islower():
        raise SystemExit("Name must be lowercase with no underscore.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plan = []
    for _ in range(args.reps):
        digits = list(range(10))
        random.shuffle(digits)
        plan.extend(digits)

    print(f"Recording {len(plan)} LIVE clips for '{args.name}'.")
    input("Use the same laptop/position as the demo. Press Enter to start...")

    for n, digit in enumerate(plan, 1):
        while True:
            print(f"[{n}/{len(plan)}] Next: {digit} ({WORDS[digit]})")
            time.sleep(0.5)
            print("   >>> SPEAK")
            audio = record_one()
            if is_good(audio):
                audio = librosa.resample(audio, orig_sr=REC_SR,
                                         target_sr=SAMPLE_RATE)
                path = OUT_DIR / f"{digit}_{args.name}_{next_index(args.name, digit)}.wav"
                sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
                break

    print(f"\nDone. Saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()