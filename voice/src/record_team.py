"""Record team voice samples in the same naming format as FSDD.

Output files:  data/team_voice/{digit}_{name}_{index}.wav  (8 kHz, mono, 16-bit)

Usage:
    python record_team.py --name ahmed --reps 30
"""
import argparse
import random
import time

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import DATA_DIR, SAMPLE_RATE

OUT_DIR = DATA_DIR / "team_voice"
REC_SR = 16000          # most microphones handle 16 kHz natively
REC_SECONDS = 1.5       # recording window per digit
WORDS = ["zero", "one", "two", "three", "four",
         "five", "six", "seven", "eight", "nine"]


def next_index(name: str, digit: int) -> int:
    """Continue numbering after existing files (supports multiple sessions)."""
    indices = []
    for f in OUT_DIR.glob(f"{digit}_{name}_*.wav"):
        indices.append(int(f.stem.split("_")[2]))
    return max(indices) + 1 if indices else 0


def record_one() -> np.ndarray:
    """Record one clip from the default microphone (mono float32)."""
    audio = sd.rec(int(REC_SECONDS * REC_SR), samplerate=REC_SR,
                   channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def is_good(audio: np.ndarray) -> bool:
    """Reject clips that are too quiet or truly clipped."""
    peak = float(np.max(np.abs(audio)))
    if peak < 0.02:
        print("   too quiet - speak louder or move closer. Retrying...")
        return False
    # Count samples at (almost) full scale: a few = fine, many = real clipping
    clipped = int(np.sum(np.abs(audio) >= 0.999))
    if clipped > 20:
        print(f"   clipped ({clipped} samples) - lower mic volume. Retrying...")
        return False
    return True


def save_clip(audio: np.ndarray, name: str, digit: int) -> None:
    """Resample to the project sample rate and save as 16-bit wav."""
    audio = librosa.resample(audio, orig_sr=REC_SR, target_sr=SAMPLE_RATE)
    idx = next_index(name, digit)
    path = OUT_DIR / f"{digit}_{name}_{idx}.wav"
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="lowercase, no underscores, e.g. ahmed")
    parser.add_argument("--reps", type=int, default=30,
                        help="repetitions per digit")
    args = parser.parse_args()

    if "_" in args.name or not args.name.islower():
        raise SystemExit("Name must be lowercase and contain no underscore.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Shuffle digits inside each round so the order is not predictable
    plan = []
    for _ in range(args.reps):
        digits = list(range(10))
        random.shuffle(digits)
        plan.extend(digits)

    print(f"Recording {len(plan)} clips for '{args.name}'.")
    print("Say the digit in ENGLISH when you see 'SPEAK'. Quiet room please.")
    input("Press Enter to start...")

    done = 0
    for digit in plan:
        while True:
            print(f"[{done + 1}/{len(plan)}] Next: {digit} ({WORDS[digit]})")
            time.sleep(0.5)
            print("   >>> SPEAK")
            audio = record_one()
            if is_good(audio):
                save_clip(audio, args.name, digit)
                break
        done += 1

    print(f"\nDone. Files saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()