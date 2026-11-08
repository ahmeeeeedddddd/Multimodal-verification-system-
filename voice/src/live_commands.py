"""Live voice commands gated by speaker verification (Task 4).

Flow per round:
  1) Speaker says a DIGIT        -> VoiceVerifier checks claimed identity.
  2) Speaker says a COMMAND word -> CommandRecognizer + permission executor.

Usage:
    python live_commands.py --claim basmala
"""
import argparse
import time

import librosa
import numpy as np
import sounddevice as sd

from commands import CommandRecognizer, CMD_SR
from config import SAMPLE_RATE
from permissions import execute
from verify_voice import VoiceVerifier

REC_SR = 16000
REC_SECONDS = 1.5


def record_raw() -> np.ndarray:
    """Record one clip from the default mic at 16 kHz (mono float32)."""
    audio = sd.rec(int(REC_SECONDS * REC_SR), samplerate=REC_SR,
                   channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def prompt_and_record(msg: str) -> np.ndarray:
    input(msg)
    time.sleep(0.3)
    print("   >>> SPEAK")
    return record_raw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True, help="claimed identity")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    verifier = VoiceVerifier("svm")
    recognizer = CommandRecognizer(min_confidence=0.5)
    print(f"Claiming '{args.claim}' | verify threshold = {verifier.threshold:.3f}")
    print("Commands: on, off, go, stop, yes (unlock), no (delete)\n")

    for i in range(args.rounds):
        print(f"--- Round {i + 1}/{args.rounds} ---")

        # Step 1: identity check with a spoken digit (8 kHz pipeline)
        digit_audio = prompt_and_record("Press Enter, then say a DIGIT (e.g. five)... ")
        digit_8k = librosa.resample(digit_audio, orig_sr=REC_SR,
                                    target_sr=SAMPLE_RATE)
        r = verifier.verify(digit_8k, args.claim)
        verified = bool(r["accepted"])
        print(f"   identity p={r['probability']:.3f} -> "
              f"{'VERIFIED' if verified else 'NOT VERIFIED'}")

        # Step 2: spoken command (16 kHz pipeline)
        cmd_audio = prompt_and_record("Press Enter, then say a COMMAND word... ")
        word, conf = recognizer.recognize(cmd_audio)
        print(f"   heard: {word} (confidence {conf:.2f})")

        result = execute(word, args.claim if verified else None, verified)
        print(f"   => {result.message}\n")


if __name__ == "__main__":
    main()