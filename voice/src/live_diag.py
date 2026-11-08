"""Diagnose live microphone verification: show the full probability vector
and save every live clip (FSDD naming) for later analysis/enrollment.

Usage:
    python live_diag.py --claim basmala --rounds 10
"""
import argparse
import time

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import DATA_DIR, SAMPLE_RATE
from verify_voice import VoiceVerifier

REC_SR = 16000
REC_SECONDS = 1.5
OUT_DIR = DATA_DIR / "live_test"


def record_clip() -> np.ndarray:
    """Record one clip, resampled to the project sample rate."""
    audio = sd.rec(int(REC_SECONDS * REC_SR), samplerate=REC_SR,
                   channels=1, dtype="float32")
    sd.wait()
    return librosa.resample(audio[:, 0], orig_sr=REC_SR, target_sr=SAMPLE_RATE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True)
    parser.add_argument("--model", default="svm", choices=["svm", "rf"])
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v = VoiceVerifier(args.model)

    for i in range(args.rounds):
        digit = input(f"[{i + 1}/{args.rounds}] Type the digit you will say, "
                      f"then Enter: ").strip()
        time.sleep(0.5)
        print("   >>> SPEAK")
        audio = record_clip()

        # Save the raw live clip in FSDD naming (index continues automatically)
        n = len(list(OUT_DIR.glob(f"{digit}_{args.claim}_*.wav")))
        sf.write(str(OUT_DIR / f"{digit}_{args.claim}_{n}.wav"),
                 audio, SAMPLE_RATE, subtype="PCM_16")

        # Full probability vector over all enrolled identities
        feats = v.features(audio).reshape(1, -1)
        proba = v.model.predict_proba(feats)[0]
        order = np.argsort(proba)[::-1]
        print("   probabilities:",
              "  ".join(f"{v.classes[j]}={proba[j]:.2f}" for j in order))
        print(f"   predicted = {v.classes[order[0]]} | claimed '{args.claim}' "
              f"p = {proba[v.classes.index(args.claim)]:.3f}\n")


if __name__ == "__main__":
    main()