"""Dataset loading, speaker split and trial generation for voice verification.

Design:
  * ENROLLED speakers  -> the system is trained on them (train/val/test split
    of their recordings, stratified by spoken digit). This includes the FSDD
    speakers listed below AND every team member found in data/team_voice.
  * UNKNOWN speakers   -> never seen in training; used only to build
    "unknown impostor" trials in val/test.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import DATA_DIR, FSDD_DIR, RANDOM_STATE
from preprocessing import preprocess_file
from features import extract_voice_features
LIVE_DIR = DATA_DIR / "live_test"

TEAM_DIR = DATA_DIR / "team_voice"

FSDD_ENROLLED = ["george", "jackson", "lucas", "nicolas"]
UNKNOWN_SPEAKERS = ["theo", "yweweler"]


def team_speakers():
    """Discover team members from file names in data/team_voice."""
    if not TEAM_DIR.exists():
        return []
    return sorted({f.stem.split("_")[1] for f in TEAM_DIR.glob("*.wav")})


def enrolled_speakers():
    """All enrolled identities: FSDD enrolled + team members."""
    return FSDD_ENROLLED + team_speakers()


@dataclass
class VoiceDataset:
    """Feature matrix + labels for a set of recordings."""
    X: np.ndarray          # (n, VOICE_FEATURE_DIM)
    speaker: np.ndarray    # (n,) speaker name per recording
    digit: np.ndarray      # (n,) spoken digit per recording


def extract_all(speakers, folders=None) -> VoiceDataset:
    """Preprocess + extract features for recordings of the given speakers.

    folders: which folders to read (default: FSDD + team_voice).
    """
    if folders is None:
        folders = [FSDD_DIR] + ([TEAM_DIR] if TEAM_DIR.exists() else [])
    X, spk, dig = [], [], []
    files = sorted(f for folder in folders for f in folder.glob("*.wav")
                   if f.stem.split("_")[1] in speakers)
    for f in tqdm(files, desc=f"Extracting {len(speakers)} speakers"):
        digit, speaker, _ = f.stem.split("_")
        X.append(extract_voice_features(preprocess_file(f)))
        spk.append(speaker)
        dig.append(int(digit))
    return VoiceDataset(np.vstack(X), np.array(spk), np.array(dig))


def split_by_session(fsdd_ids, team_ids):
    """Session-aware split.

    * FSDD speakers: random stratified train/val/test (as before).
    * Team speakers: team_voice -> train only;
                     live_test  -> split 50/50 into val and test.
    Returns (train, val, test) VoiceDatasets over ALL enrolled speakers.
    """
    fsdd = extract_all(fsdd_ids, folders=[FSDD_DIR])
    f_train, f_val, f_test = split_enrolled(fsdd)

    t_train = extract_all(team_ids, folders=[TEAM_DIR])
    live = extract_all(team_ids, folders=[LIVE_DIR])

    # split live recordings, stratified by (speaker, digit) when possible
    idx = np.arange(len(live.X))
    strata = np.array([f"{s}_{d}" for s, d in zip(live.speaker, live.digit)])
    idx_val, idx_test = train_test_split(
        idx, test_size=0.5, stratify=strata, random_state=RANDOM_STATE)
    l_val, l_test = subset(live, idx_val), subset(live, idx_test)

    def merge(a, b):
        return VoiceDataset(np.vstack([a.X, b.X]),
                            np.concatenate([a.speaker, b.speaker]),
                            np.concatenate([a.digit, b.digit]))

    return (merge(f_train, t_train), merge(f_val, l_val),
            merge(f_test, l_test))


def subset(ds: VoiceDataset, idx) -> VoiceDataset:
    return VoiceDataset(ds.X[idx], ds.speaker[idx], ds.digit[idx])


def split_enrolled(ds: VoiceDataset, val_size=0.2, test_size=0.2):
    """Split enrolled-speaker recordings into train/val/test.

    Stratified by (speaker, digit) so every split contains every digit
    for every speaker in equal proportion.
    """
    strata = np.array([f"{s}_{d}" for s, d in zip(ds.speaker, ds.digit)])
    idx = np.arange(len(ds.X))

    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=strata, random_state=RANDOM_STATE)
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=rel_val, stratify=strata[idx_trainval],
        random_state=RANDOM_STATE)

    return subset(ds, idx_train), subset(ds, idx_val), subset(ds, idx_test)


def split_unknown(ds: VoiceDataset):
    """Split unknown-speaker recordings 50/50 into val and test pools."""
    strata = np.array([f"{s}_{d}" for s, d in zip(ds.speaker, ds.digit)])
    idx = np.arange(len(ds.X))
    idx_val, idx_test = train_test_split(
        idx, test_size=0.5, stratify=strata, random_state=RANDOM_STATE)
    return subset(ds, idx_val), subset(ds, idx_test)


def make_trials(enrolled: VoiceDataset, unknown: VoiceDataset,
                seed: int = RANDOM_STATE):
    """Build verification trials: (feature index, claimed_id, label, type).

    Every recording of an enrolled speaker produces:
      * 1 genuine trial          (claims its own identity)
      * 1 known-impostor trial   (claims a random OTHER enrolled identity)
    Every recording of an unknown speaker produces:
      * 1 unknown-impostor trial (claims a random enrolled identity)

    Returns:
        X       (n_trials, dim) features
        claimed (n_trials,)     claimed speaker name
        y       (n_trials,)     1 = genuine, 0 = impostor
        kind    (n_trials,)     'genuine' | 'known_impostor' | 'unknown_impostor'
    """
    rng = np.random.default_rng(seed)
    ids = sorted(set(enrolled.speaker))

    X, claimed, y, kind = [], [], [], []

    for x, spk in zip(enrolled.X, enrolled.speaker):
        X.append(x); claimed.append(spk); y.append(1); kind.append("genuine")

        other = rng.choice([i for i in ids if i != spk])
        X.append(x); claimed.append(other); y.append(0); kind.append("known_impostor")

    for x in unknown.X:
        X.append(x); claimed.append(rng.choice(ids)); y.append(0)
        kind.append("unknown_impostor")

    return np.vstack(X), np.array(claimed), np.array(y), np.array(kind)