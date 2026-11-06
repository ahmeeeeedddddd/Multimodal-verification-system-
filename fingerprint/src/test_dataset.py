"""Smoke-tests for dataset loading and trial generation.

These tests use SYNTHETIC DATA so they run without the real SOCOFing images.
They validate the shapes, label balance, and trial-generation logic.

Run from fingerprint/src/:
    python test_dataset.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from dataset import FingerprintDataset, _subset, split_enrolled, split_unknown, make_trials


# ------------------------------------------------------------------
# Synthetic dataset factory
# ------------------------------------------------------------------
def _make_synthetic(n_subjects=10, n_fingers=5, feat_dim=50,
                    seed=0) -> FingerprintDataset:
    """Create a fake FingerprintDataset for unit testing."""
    rng     = np.random.default_rng(seed)
    fingers = ["index", "middle", "ring", "little", "thumb"]
    X, subjects, finger_labels, hands = [], [], [], []
    for subj in range(1, n_subjects + 1):
        for fi, f in enumerate(fingers[:n_fingers]):
            X.append(rng.random(feat_dim).astype(np.float32))
            subjects.append(subj)
            finger_labels.append(f)
            hands.append("Left" if fi % 2 == 0 else "Right")
    return FingerprintDataset(
        X=np.vstack(X),
        subject=np.array(subjects, dtype=int),
        finger=np.array(finger_labels),
        hand=np.array(hands),
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
def test_split_sizes():
    """Split sizes should sum to the total and be non-empty."""
    ds = _make_synthetic(n_subjects=20, n_fingers=5)
    train, val, test = split_enrolled(ds, val_size=0.2, test_size=0.2)
    total = len(ds.X)
    split_sum = len(train.X) + len(val.X) + len(test.X)
    assert split_sum == total, f"Split sum {split_sum} ≠ total {total}"
    assert len(train.X) > 0 and len(val.X) > 0 and len(test.X) > 0
    print(f"  [PASS] split_sizes  (train={len(train.X)}, "
          f"val={len(val.X)}, test={len(test.X)})")


def test_trial_labels():
    """Trials must have genuine label=1 and impostor label=0."""
    enrolled = _make_synthetic(n_subjects=10, n_fingers=5, seed=0)
    unknown  = _make_synthetic(n_subjects=4,  n_fingers=3, seed=99)
    # offset unknown subject IDs so they don't overlap
    unknown.subject += 1000

    _, _, y, kind = make_trials(enrolled, unknown)
    assert all(y[kind == "genuine"] == 1),        "Genuine trials not labelled 1"
    assert all(y[kind == "known_impostor"] == 0),  "Known impostors not labelled 0"
    assert all(y[kind == "unknown_impostor"] == 0),"Unknown impostors not labelled 0"
    print("  [PASS] trial_labels")


def test_trial_counts():
    """Each enrolled sample → 2 trials; each unknown → 1 trial."""
    enrolled = _make_synthetic(n_subjects=5, n_fingers=4, seed=1)
    unknown  = _make_synthetic(n_subjects=2, n_fingers=3, seed=2)
    unknown.subject += 500

    X, cl, y, kind = make_trials(enrolled, unknown)
    n_enrolled = len(enrolled.X)
    n_unknown  = len(unknown.X)
    assert len(X) == 2 * n_enrolled + n_unknown, (
        f"Expected {2*n_enrolled+n_unknown} trials, got {len(X)}"
    )
    print(f"  [PASS] trial_counts  "
          f"({n_enrolled} enrolled → {2*n_enrolled} trials; "
          f"{n_unknown} unknown → {n_unknown} trials)")


def test_subset():
    """_subset must preserve relative order and shapes."""
    ds  = _make_synthetic(n_subjects=4, n_fingers=3)
    idx = np.array([0, 2, 4])
    sub = _subset(ds, idx)
    assert len(sub.X) == 3
    assert np.array_equal(sub.X, ds.X[idx])
    print("  [PASS] subset")


if __name__ == "__main__":
    print("Testing dataset utilities (synthetic data) …")
    test_subset()
    test_split_sizes()
    test_trial_labels()
    test_trial_counts()
    print("All dataset tests passed.")

