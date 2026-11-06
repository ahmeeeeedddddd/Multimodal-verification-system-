"""SOCOFing dataset loader, subject split, and verification trial generator.

SOCOFing filename convention:
    <subject_id>__<gender>_Left_<finger>_finger.BMP
    <subject_id>__<gender>_Right_<finger>_finger.BMP

    e.g.  1__M_Left_index_finger.BMP

We load the REAL subset only (600 subjects × up to 10 fingers each).

Design
------
* ENROLLED subjects  → system trained on them (train / val / test split,
                        stratified by finger).
* UNKNOWN subjects   → never seen in training; used for unknown-impostor
                        trials in val / test sets.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    REAL_DIR, N_ENROLLED, N_UNKNOWN,
    RANDOM_STATE, VAL_SIZE, TEST_SIZE,
)
from preprocessing import preprocess_file
from features import extract_fingerprint_features

# ------------------------------------------------------------------
# Filename parsing
# ------------------------------------------------------------------
_FNAME_RE = re.compile(
    r"^(\d+)__([MF])_(Left|Right)_(\w+)_finger\.BMP$",
    re.IGNORECASE,
)


def parse_filename(fname: str) -> dict | None:
    """Parse a SOCOFing filename into its components.

    Returns a dict with keys: subject_id, gender, hand, finger.
    Returns None if the filename doesn't match the pattern.
    """
    m = _FNAME_RE.match(Path(fname).name)
    if m is None:
        return None
    return dict(
        subject_id=int(m.group(1)),
        gender=m.group(2),
        hand=m.group(3),
        finger=m.group(4).lower(),
    )


# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------
@dataclass
class FingerprintDataset:
    """Feature matrix + metadata for a set of fingerprint images."""
    X:          np.ndarray   # (n, feature_dim)  float32
    subject:    np.ndarray   # (n,)               int
    finger:     np.ndarray   # (n,)               str  e.g. "index"
    hand:       np.ndarray   # (n,)               str  "Left" / "Right"


def _subset(ds: FingerprintDataset, idx) -> FingerprintDataset:
    return FingerprintDataset(
        ds.X[idx], ds.subject[idx], ds.finger[idx], ds.hand[idx]
    )


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def _select_subjects(all_ids: list[int], n_enrolled: int,
                     n_unknown: int) -> tuple[list[int], list[int]]:
    """Pick enrolled and unknown subject IDs deterministically."""
    rng = np.random.default_rng(RANDOM_STATE)
    chosen = rng.choice(all_ids, size=n_enrolled + n_unknown, replace=False)
    enrolled = sorted(chosen[:n_enrolled].tolist())
    unknown  = sorted(chosen[n_enrolled:].tolist())
    return enrolled, unknown


def load_dataset(subject_ids: list[int]) -> FingerprintDataset:
    """Preprocess + extract features for every image of the given subjects.

    Args:
        subject_ids: list of integer subject IDs to load.

    Returns:
        FingerprintDataset with one row per image.
    """
    id_set = set(subject_ids)
    paths = [
        p for p in sorted(REAL_DIR.glob("*.BMP"))
        if (meta := parse_filename(p.name)) and meta["subject_id"] in id_set
    ]
    if not paths:
        raise FileNotFoundError(
            f"No .BMP images found in {REAL_DIR} for subjects {subject_ids[:5]}…\n"
            "Run  python download_dataset.py  first to fetch the SOCOFing data."
        )

    X, subjects, fingers, hands = [], [], [], []
    for p in tqdm(paths, desc="Extracting fingerprint features"):
        meta = parse_filename(p.name)
        img  = preprocess_file(p)
        X.append(extract_fingerprint_features(img))
        subjects.append(meta["subject_id"])
        fingers.append(meta["finger"])
        hands.append(meta["hand"])

    return FingerprintDataset(
        X=np.vstack(X).astype(np.float32),
        subject=np.array(subjects, dtype=int),
        finger=np.array(fingers),
        hand=np.array(hands),
    )


def get_all_subject_ids() -> list[int]:
    """Scan REAL_DIR and return a sorted list of all unique subject IDs."""
    ids = set()
    for p in REAL_DIR.glob("*.BMP"):
        meta = parse_filename(p.name)
        if meta:
            ids.add(meta["subject_id"])
    return sorted(ids)


# ------------------------------------------------------------------
# Train / val / test split
# ------------------------------------------------------------------
def split_enrolled(ds: FingerprintDataset, val_size=VAL_SIZE, test_size=TEST_SIZE):
    """Split enrolled-subject data into train / val / test.

    Stratified by subject ID so each split contains images from every enrolled
    subject.  SOCOFing Real has exactly one image per (subject, finger) pair,
    so stratifying by (subject × finger) would leave each stratum with only 1
    member and make sklearn's StratifiedShuffleSplit fail.

    Returns (train, val, test) FingerprintDataset triples.
    """
    strata = ds.subject          # stratify by subject only
    idx    = np.arange(len(ds.X))

    idx_tv, idx_test = train_test_split(
        idx, test_size=test_size, stratify=strata, random_state=RANDOM_STATE)
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=rel_val, stratify=strata[idx_tv],
        random_state=RANDOM_STATE)

    return _subset(ds, idx_train), _subset(ds, idx_val), _subset(ds, idx_test)


def split_unknown(ds: FingerprintDataset):
    """Split unknown-subject data 50/50 into val and test pools.

    Stratified by subject ID (see split_enrolled for why we don't use finger).
    """
    strata = ds.subject
    idx    = np.arange(len(ds.X))
    idx_v, idx_t = train_test_split(
        idx, test_size=0.5, stratify=strata, random_state=RANDOM_STATE)
    return _subset(ds, idx_v), _subset(ds, idx_t)


# ------------------------------------------------------------------
# Verification trial generation
# ------------------------------------------------------------------
def make_trials(enrolled: FingerprintDataset, unknown: FingerprintDataset,
                seed: int = RANDOM_STATE):
    """Build verification trials.

    For each enrolled sample:
        * 1 genuine trial  → claims its own subject ID
        * 1 known-impostor → claims a random different enrolled ID

    For each unknown sample:
        * 1 unknown-impostor trial → claims a random enrolled ID

    Returns:
        X       (n_trials, dim)   feature matrix
        claimed (n_trials,)       claimed subject IDs (int)
        y       (n_trials,)       1 = genuine, 0 = impostor
        kind    (n_trials,)       "genuine" | "known_impostor" | "unknown_impostor"
    """
    rng = np.random.default_rng(seed)
    ids = sorted(set(enrolled.subject.tolist()))

    X, claimed, y, kind = [], [], [], []

    for x, subj in zip(enrolled.X, enrolled.subject):
        # genuine
        X.append(x); claimed.append(subj); y.append(1); kind.append("genuine")
        # known impostor
        others = [i for i in ids if i != subj]
        X.append(x); claimed.append(int(rng.choice(others)))
        y.append(0); kind.append("known_impostor")

    for x in unknown.X:
        X.append(x); claimed.append(int(rng.choice(ids)))
        y.append(0); kind.append("unknown_impostor")

    return (
        np.vstack(X),
        np.array(claimed, dtype=int),
        np.array(y, dtype=int),
        np.array(kind),
    )
