"""Isolated-word command recognizer (Task 4).

Words are recognised with time-aware MFCC features + ZCR + RMS and an SVM.
Runs at 16 kHz (Speech Commands native rate), independent of the 8 kHz
speaker-verification pipeline.

An extra "unknown" class (other words from Speech Commands) lets the
recognizer reject words that are not commands instead of forcing one of
the six command words.
"""
import warnings

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from config import DATA_DIR, MODELS_DIR, RANDOM_STATE

warnings.filterwarnings("ignore", category=FutureWarning)

CMD_SR = 16000
CMD_SAMPLES = CMD_SR            # fixed 1 second
CMD_WORDS = ["on", "off", "go", "stop", "yes", "no"]
UNKNOWN_WORDS = ["cat", "dog", "bed", "bird", "happy", "house", "tree", "wow"]
UNKNOWN_LABEL = "unknown"
MAX_UNKNOWN_PER_WORD = 300      # ~2400 total, similar size to one command class
CMD_DIR = DATA_DIR / "speech_commands"
N_MFCC = 13
N_FFT, HOP = 512, 160           # 32 ms window, 10 ms hop at 16 kHz
N_SEGMENTS = 4                  # temporal segments per word


def preprocess_cmd(audio: np.ndarray) -> np.ndarray:
    """Trim silence, remove DC, peak-normalize, pad/crop to 1 second."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size:
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        audio = trimmed if trimmed.size else audio
    audio = audio - audio.mean()
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    if audio.size >= CMD_SAMPLES:
        return audio[:CMD_SAMPLES].astype(np.float32)
    return np.pad(audio, (0, CMD_SAMPLES - audio.size)).astype(np.float32)


def cmd_features(audio: np.ndarray) -> np.ndarray:
    """Time-aware features: MFCC mean per temporal segment + delta + ZCR/RMS.

    Dimension = 13*4 (segment means) + 13 (std) + 13 (delta mean)
                + 4 (ZCR/RMS stats) = 82.
    """
    y = preprocess_cmd(audio)
    mfcc = librosa.feature.mfcc(y=y, sr=CMD_SR, n_mfcc=N_MFCC, n_fft=N_FFT,
                                hop_length=HOP, n_mels=40)
    delta = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT,
                                             hop_length=HOP).ravel()
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP).ravel()

    segments = np.array_split(mfcc, N_SEGMENTS, axis=1)
    seg_means = np.concatenate([s.mean(axis=1) for s in segments])

    return np.concatenate([
        seg_means, mfcc.std(axis=1), np.abs(delta).mean(axis=1),
        [zcr.mean(), zcr.std(), rms.mean(), rms.std()],
    ]).astype(np.float32)


def _read_list(name: str) -> set:
    """Read an official split list (paths like 'yes/abc_nohash_0.wav')."""
    path = CMD_DIR / name
    return set(path.read_text().split())


def load_dataset():
    """Load clips per word, split by the OFFICIAL lists (speaker-disjoint).

    Command words keep their own label; the extra words are all labelled
    "unknown" (capped per word to keep the classes balanced).

    Returns dict: split -> (X, y) for 'train', 'val', 'test'.
    """
    val_set = _read_list("validation_list.txt")
    test_set = _read_list("testing_list.txt")
    data = {s: ([], []) for s in ("train", "val", "test")}

    # (folder name, label, max files or None)
    jobs = [(w, w, None) for w in CMD_WORDS] + \
           [(w, UNKNOWN_LABEL, MAX_UNKNOWN_PER_WORD) for w in UNKNOWN_WORDS]

    for word, label, cap in jobs:
        folder = CMD_DIR / word
        if not folder.exists():
            raise SystemExit(f"Missing folder: {folder}")
        files = sorted(folder.glob("*.wav"))
        if cap is not None:
            files = files[:cap]
        for f in tqdm(files, desc=f"Extracting '{word}' -> {label}"):
            rel = f"{word}/{f.name}"
            split = "val" if rel in val_set else "test" if rel in test_set else "train"
            audio, sr = sf.read(str(f), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != CMD_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=CMD_SR)
            data[split][0].append(cmd_features(audio))
            data[split][1].append(label)

    return {s: (np.vstack(X), np.array(y)) for s, (X, y) in data.items()}


def train():
    data = load_dataset()
    X_tr, y_tr = data["train"]
    X_va, y_va = data["val"]
    X_te, y_te = data["test"]
    print(f"train={len(y_tr)} val={len(y_va)} test={len(y_te)}")

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(kernel="rbf", probability=True,
                                 random_state=RANDOM_STATE))])
    grid = {"clf__C": [10, 50, 100], "clf__gamma": ["scale", 0.005]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
    gs.fit(X_tr, y_tr)
    model = gs.best_estimator_
    print("Best params:", gs.best_params_, f"| CV acc = {gs.best_score_:.4f}")
    print(f"VAL accuracy: {model.score(X_va, y_va):.4f}")

    pred = model.predict(X_te)
    print("\nTEST report:\n", classification_report(y_te, pred, digits=3))
    print("Confusion matrix (rows=true, cols=pred), order:", list(model.classes_))
    print(confusion_matrix(y_te, pred, labels=model.classes_))

    joblib.dump({"model": model, "words": list(model.classes_)},
                MODELS_DIR / "command_svm.joblib")
    print("Saved ->", MODELS_DIR / "command_svm.joblib")


class CommandRecognizer:
    """Load the trained model and classify a 16 kHz waveform."""

    def __init__(self, min_confidence: float = 0.5):
        bundle = joblib.load(MODELS_DIR / "command_svm.joblib")
        self.model = bundle["model"]
        self.words = list(bundle["words"])
        self.min_confidence = min_confidence

    def recognize(self, audio_16k: np.ndarray):
        """Return (word or None, confidence).

        None = predicted "unknown" (not a command) or confidence too low.
        """
        proba = self.model.predict_proba(cmd_features(audio_16k).reshape(1, -1))[0]
        i = int(np.argmax(proba))
        conf = float(proba[i])
        word = self.words[i]
        if word == UNKNOWN_LABEL or conf < self.min_confidence:
            return None, conf
        return word, conf


if __name__ == "__main__":
    train()