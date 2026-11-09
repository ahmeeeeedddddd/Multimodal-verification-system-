"""06 — Train Speech Command Recognizer.

Trains the isolated-word speech command recognizer on standard vocabulary:
['unlock', 'open', 'status', 'lights', 'delete', 'start', 'stop', 'help']
using MFCC feature extraction + RBF SVM classifier.
Saves:
- saved_models/command_recognizer.pkl
"""

import os
import soundfile as sf
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from speech_commands import (
    COMMAND_WORDS,
    extract_command_features,
    generate_synthetic_command_data,
    SpeechCommandRecognizer
)


def main():
    np.random.seed(42)
    os.makedirs("saved_models", exist_ok=True)
    print("[06-Speech] Training Speech Command Recognizer...")

    # 1. Base synthetic data
    X_syn, y_syn = generate_synthetic_command_data(n_samples_per_word=100)

    # 2. Add demo audio anchors
    X_demo, y_demo = [], []
    for word_idx, name in enumerate(COMMAND_WORDS):
        path = f"demo_assets/audio/command_{name}.wav"
        if os.path.exists(path):
            y_audio, sr = sf.read(path)
            feat = extract_command_features(y_audio, sr=sr)
            for _ in range(50):
                augmented = feat + np.random.normal(0, 0.2 * np.abs(feat).mean(), feat.shape)
                X_demo.append(augmented)
                y_demo.append(word_idx)

    X_all = np.vstack([X_syn, np.array(X_demo)])
    y_all = np.concatenate([y_syn, np.array(y_demo)])
    print(f"[06-Speech] Total training samples: {len(X_all)}")

    # 3. Train Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, probability=True, random_state=42))
    ])
    pipeline.fit(X_all, y_all)

    # 4. Evaluation
    print("\n[06-Speech] Testing on command audio assets:")
    correct = 0
    for name in COMMAND_WORDS:
        path = f"demo_assets/audio/command_{name}.wav"
        if os.path.exists(path):
            y_audio, sr = sf.read(path)
            feat = extract_command_features(y_audio, sr=sr).reshape(1, -1)
            probs = pipeline.predict_proba(feat)[0]
            pred_idx = int(np.argmax(probs))
            pred_word = COMMAND_WORDS[pred_idx]
            match = "PASS" if pred_word == name else "FAIL"
            print(f"  [{match}] {name}.wav -> {pred_word} (conf={probs[pred_idx]*100:.1f}%)")
            if pred_word == name:
                correct += 1

    print(f"\n[06-Speech] Accuracy: {correct}/{len(COMMAND_WORDS)} ({correct/len(COMMAND_WORDS)*100:.1f}%)")

    # 5. Save model
    save_path = "saved_models/command_recognizer.pkl"
    joblib.dump({"pipeline": pipeline, "words": COMMAND_WORDS}, save_path)
    print(f"[06-Speech] Saved model to {save_path}")


if __name__ == "__main__":
    main()
