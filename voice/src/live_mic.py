"""Live microphone verification (Bonus 2, voice part).

Records one clip from the microphone and runs the SAME preprocessing and
feature code used in training (VoiceVerifier.verify).

Usage:
    python live_mic.py --claim basmala
    python live_mic.py --claim basmala --model rf --rounds 5
"""
import argparse
import time

import librosa
import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE
from verify_voice import VoiceVerifier

REC_SR = 16000        # record at 16 kHz, then resample (same as record_team.py)
REC_SECONDS = 1.5


def record_clip() -> np.ndarray:
    """Record one clip from the default microphone, resampled to SAMPLE_RATE."""
    audio = sd.rec(int(REC_SECONDS * REC_SR), samplerate=REC_SR,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0]
    return librosa.resample(audio, orig_sr=REC_SR, target_sr=SAMPLE_RATE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True, help="claimed identity")
    parser.add_argument("--model", default="svm", choices=["svm", "rf"])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    verifier = VoiceVerifier(args.model)
    print(f"Enrolled: {verifier.enrolled_ids}")
    print(f"Claiming: {args.claim} | threshold = {verifier.threshold:.3f}\n")

    for i in range(args.rounds):
        input(f"[{i + 1}/{args.rounds}] Press Enter, then say a digit in English...")
        time.sleep(0.3)
        print("   >>> SPEAK")
        audio = record_clip()

        if float(np.max(np.abs(audio))) < 0.02:
            print("   too quiet, try again.\n")
            continue

        r = verifier.verify(audio, args.claim)
        print(f"   probability = {r['probability']:.3f} -> "
              f"{'ACCEPT' if r['accepted'] else 'REJECT'}\n")


if __name__ == "__main__":
    main()