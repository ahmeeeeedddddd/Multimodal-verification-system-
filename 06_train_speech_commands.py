import os
import soundfile as sf
import numpy as np
import joblib
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from speech_commands import (
    COMMAND_WORDS,
    extract_command_features,
    generate_synthetic_command_data
)


def augment_audio(y, sr=16000):
    """Generate audio variations using classical acoustic transformations."""
    augmented = []
    # 1. Volume scaling
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        augmented.append(y * scale)
    # 2. Additive background noise
    for noise_level in [0.005, 0.01, 0.02, 0.03]:
        noise = np.random.normal(0, noise_level, len(y))
        augmented.append(y + noise)
    # 3. Time-shift / padding variations
    for shift in [-1600, -800, 800, 1600]:
        augmented.append(np.roll(y, shift))
    return augmented


def main():
    np.random.seed(42)
    os.makedirs("saved_models", exist_ok=True)
    print("[06-Speech] Training Speech Command Recognizer on Real + Acoustic Augmentations...")

    X_train, y_train = [], []

    # 1. Load each real command audio asset and heavily augment it
    for word_idx, name in enumerate(COMMAND_WORDS):
        path = f"demo_assets/audio/command_{name}.wav"
        if os.path.exists(path):
            y_audio, sr = sf.read(path)
            augs = augment_audio(y_audio, sr=sr)
            for a in augs:
                feat = extract_command_features(a, sr=sr)
                X_train.append(feat)
                y_train.append(word_idx)
                # Feature-space jitter
                for _ in range(8):
                    jitter = feat + np.random.normal(0, 0.08 * np.abs(feat).clip(min=1e-3))
                    X_train.append(jitter)
                    y_train.append(word_idx)

    # 2. Add synthetic acoustic data
    X_syn, y_syn = generate_synthetic_command_data(n_samples_per_word=50)
    X_all = np.vstack([np.array(X_train), X_syn])
    y_all = np.concatenate([np.array(y_train), y_syn])
    print(f"[06-Speech] Total training samples: {len(X_all)}")

    # 3. Train Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, probability=True, random_state=42))
    ])
    pipeline.fit(X_all, y_all)

    # 4. Evaluation
    print("\n[06-Speech] Testing on real command audio assets:")
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
