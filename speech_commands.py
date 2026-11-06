"""Task 4 / Bonus 1: Verification-Based Speech Command System.

Provides:
- Isolated-word speech command recognizer using MFCC features + SVM / KNN classifier.
- Role-based access control (RBAC) permission table.
- Command executor that gates protected commands based on identity and verification status.
- Live microphone recording support.
"""

import os
import joblib
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

COMMAND_WORDS = [
    "unlock",
    "open",
    "status",
    "lights",
    "delete",
    "start",
    "stop",
    "help"
]

# Permission table mapping roles to allowed commands
ROLE_PERMISSIONS = {
    "admin": ["unlock", "open", "status", "lights", "delete", "start", "stop", "help"],
    "user": ["status", "lights", "start", "stop", "help"],
    "guest": ["help"],
    "unverified": []
}

# User to role mapping (default mock directory for enrolled subjects)
USER_ROLES = {
    "User_28": "admin",
    "User_29": "user",
    "User_30": "user",
    "User_31": "user",
    "User_32": "admin",
    "User_33": "user",
    "User_34": "guest",
    # numeric mapped IDs
    28: "admin",
    29: "user",
    30: "user",
    31: "user",
    32: "admin",
    33: "user",
    34: "guest"
}


def extract_command_features(y, sr=16000, n_mfcc=13):
    """Extract MFCC statistics for speech command recognition."""
    if len(y) < sr // 4:
        y = np.pad(y, (0, sr // 4 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    return np.hstack([mfcc_mean, mfcc_std, [zcr, rms]])


def generate_synthetic_command_data(n_samples_per_word=25, sr=16000, duration=1.0, seed=42):
    """Generate reproducible acoustic feature variations for standard command vocabulary."""
    np.random.seed(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    X, y = [], []

    # Characteristic formant frequencies for each command word
    word_freqs = {
        "unlock": (320, 1100, 2400),
        "open": (450, 950, 2200),
        "status": (280, 1400, 2600),
        "lights": (400, 1700, 2800),
        "delete": (350, 1500, 2500),
        "start": (300, 1200, 2300),
        "stop": (480, 850, 2100),
        "help": (500, 1300, 2400)
    }

    for word_idx, word in enumerate(COMMAND_WORDS):
        f1_base, f2_base, f3_base = word_freqs[word]
        for _ in range(n_samples_per_word):
            f1 = f1_base + np.random.normal(0, 15)
            f2 = f2_base + np.random.normal(0, 30)
            f3 = f3_base + np.random.normal(0, 45)

            # Signal synthesis with amplitude envelope
            env = np.hanning(len(t))
            sig = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.sin(2 * np.pi * f3 * t)
            sig = (sig * env + np.random.normal(0, 0.03, len(t))).astype(np.float32)

            feat = extract_command_features(sig, sr=sr)
            X.append(feat)
            y.append(word_idx)

    return np.array(X), np.array(y)


class SpeechCommandRecognizer:
    """Trained isolated-word speech command recognizer."""

    def __init__(self, model_type="svm"):
        self.model_type = model_type
        self.words = list(COMMAND_WORDS)
        self.pipeline = None

    def fit(self, X, y):
        if self.model_type == "svm":
            clf = SVC(kernel="rbf", C=5.0, probability=True, random_state=42)
        else:
            clf = KNeighborsClassifier(n_neighbors=3)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
        self.pipeline.fit(X, y)
        return self

    def predict(self, audio, sr=16000):
        """Predict command word and confidence from raw audio waveform."""
        feat = extract_command_features(audio, sr=sr).reshape(1, -1)
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            probs = self.pipeline.predict_proba(feat)[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
        else:
            pred_idx = int(self.pipeline.predict(feat)[0])
            conf = 1.0

        return self.words[pred_idx], conf

    def save(self, filepath="saved_models/command_recognizer.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "words": self.words}, filepath)

    @classmethod
    def load(cls, filepath="saved_models/command_recognizer.pkl"):
        recognizer = cls()
        bundle = joblib.load(filepath)
        recognizer.pipeline = bundle["pipeline"]
        recognizer.words = bundle["words"]
        return recognizer


def get_trained_recognizer():
    """Load or train a command recognizer."""
    save_path = os.path.join("saved_models", "command_recognizer.pkl")
    if os.path.exists(save_path):
        try:
            return SpeechCommandRecognizer.load(save_path)
        except Exception:
            pass

    X, y = generate_synthetic_command_data()
    rec = SpeechCommandRecognizer(model_type="svm")
    rec.fit(X, y)
    rec.save(save_path)
    return rec


class CommandExecutor:
    """Permission-gated command executor tied to multimodal verification."""

    def __init__(self, permissions=None, user_roles=None):
        self.permissions = permissions or ROLE_PERMISSIONS
        self.user_roles = user_roles or USER_ROLES

    def get_role(self, identity):
        return self.user_roles.get(identity, "guest")

    def execute(self, command: str, identity, is_verified: bool):
        """Execute command only if identity is verified and holds appropriate role permission.

        Returns:
            dict with {
                'command': command,
                'identity': identity,
                'role': role,
                'verified': is_verified,
                'status': 'SUCCESS' | 'PERMISSION_DENIED' | 'UNVERIFIED',
                'action_taken': str
            }
        """
        command = command.lower().strip()
        role = self.get_role(identity) if is_verified else "unverified"
        allowed_commands = self.permissions.get(role, [])

        if not is_verified:
            return {
                "command": command,
                "identity": identity,
                "role": "unverified",
                "verified": False,
                "status": "UNVERIFIED",
                "action_taken": f"ACCESS DENIED: Biometric verification failed for identity '{identity}'. Protected command '{command}' cannot be executed."
            }

        if command not in allowed_commands:
            return {
                "command": command,
                "identity": identity,
                "role": role,
                "verified": True,
                "status": "PERMISSION_DENIED",
                "action_taken": f"PERMISSION DENIED: Identity '{identity}' has role '{role}'. Command '{command}' requires elevated privileges."
            }

        # Successful execution
        action_descriptions = {
            "unlock": "Secure door lock electronically DISENGAGED.",
            "open": "Main access gateway OPENED.",
            "status": "All systems operational: Modalities active, Biometric threshold OK.",
            "lights": "Facility lighting system toggled ON.",
            "delete": "Sensitive record successfully REMOVED from security log.",
            "start": "Surveillance & monitoring daemon STARTED.",
            "stop": "Surveillance & monitoring daemon STOPPED.",
            "help": "Available commands for your role: " + ", ".join(allowed_commands)
        }

        desc = action_descriptions.get(command, f"Action for command '{command}' executed.")

        return {
            "command": command,
            "identity": identity,
            "role": role,
            "verified": True,
            "status": "SUCCESS",
            "action_taken": f"SUCCESS [Role: {role.upper()}]: {desc}"
        }


def record_live_command(duration=1.5, sr=16000):
    """Record a voice command from the microphone using sounddevice."""
    try:
        import sounddevice as sd
        print(f"[*] Recording for {duration} seconds... Speak command now!")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print(f"[-] Could not record microphone audio: {e}")
        return None

