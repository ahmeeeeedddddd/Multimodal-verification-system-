"""Train and tune fingerprint verification models (SVM + Random Forest).

Approach: closed-set multiclass classifier over ENROLLED subjects.
genuine_probability(claimed) = P(claimed | fingerprint features).

Usage:
    cd fingerprint/src
    python train_fingerprint.py

Outputs (saved to fingerprint/models/):
    fingerprint_svm.joblib
    fingerprint_rf.joblib
Each bundle contains: {model, threshold, classes, feature_dim}.
"""
import sys
import warnings
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

from config import MODELS_DIR, RANDOM_STATE, N_ENROLLED, N_UNKNOWN
from dataset import (
    get_all_subject_ids, _select_subjects,
    load_dataset, split_enrolled, split_unknown, make_trials,
)
from metrics import compute_eer, auc, far_frr, full_report
from features import feature_dim as get_feature_dim


# ------------------------------------------------------------------
# Scoring helpers
# ------------------------------------------------------------------
def genuine_scores(model: Pipeline, X: np.ndarray,
                   claimed: np.ndarray) -> np.ndarray:
    """P(claimed subject | features) for every trial."""
    proba   = model.predict_proba(X)
    classes = list(model.classes_)
    # Map each claimed subject to its column index; default to 0 for unknowns
    cols = np.array([
        classes.index(c) if c in classes else 0
        for c in claimed
    ])
    return proba[np.arange(len(X)), cols]


# ------------------------------------------------------------------
# Model definitions
# ------------------------------------------------------------------
def build_models() -> dict:
    """Return {name: (pipeline, param_grid)} for each model family."""

    # ── SVM ────────────────────────────────────────────────────────
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", probability=True,
                       random_state=RANDOM_STATE)),
    ])
    svm_grid = {
        "clf__C":     [1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.05],
    }

    # ── Random Forest ──────────────────────────────────────────────
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_grid = {
        "clf__n_estimators":   [200, 400],
        "clf__max_depth":      [None, 20],
        "clf__min_samples_leaf": [1, 3],
    }

    return {"svm": (svm, svm_grid), "rf": (rf, rf_grid)}


# ------------------------------------------------------------------
# Main training routine
# ------------------------------------------------------------------
def main():
    # ── 1. Load data ───────────────────────────────────────────────
    print("Loading subject IDs from SOCOFing …")
    all_ids = get_all_subject_ids()
    if len(all_ids) < N_ENROLLED + N_UNKNOWN:
        sys.exit(
            f"Need at least {N_ENROLLED + N_UNKNOWN} subjects, "
            f"found {len(all_ids)}.  Run download_dataset.py first."
        )
    enrolled_ids, unknown_ids = _select_subjects(all_ids, N_ENROLLED, N_UNKNOWN)
    print(f"Enrolled subjects : {len(enrolled_ids)}  "
          f"(first 5: {enrolled_ids[:5]})")
    print(f"Unknown  subjects : {len(unknown_ids)}")

    print("\nLoading enrolled dataset …")
    enrolled_ds = load_dataset(enrolled_ids)
    print(f"  {len(enrolled_ds.X)} images, "
          f"feature dim = {enrolled_ds.X.shape[1]}")

    print("\nLoading unknown dataset …")
    unknown_ds  = load_dataset(unknown_ids)

    # ── 2. Split ───────────────────────────────────────────────────
    train, val, test   = split_enrolled(enrolled_ds)
    unk_val, unk_test  = split_unknown(unknown_ds)

    print(f"\nSplit sizes  train={len(train.X)}  "
          f"val={len(val.X)}  test={len(test.X)}")

    # ── 3. Train / tune ────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, (pipe, grid) in build_models().items():
        print(f"\n{'='*60}")
        print(f"  Tuning {name.upper()}")
        print(f"{'='*60}")

        gs = GridSearchCV(
            pipe, grid, cv=cv, scoring="accuracy",
            n_jobs=-1, verbose=1,
        )
        gs.fit(train.X, train.subject)
        model = gs.best_estimator_

        print(f"\nBest params : {gs.best_params_}")
        print(f"CV accuracy : {gs.best_score_:.4f}")

        # ── 4. Validation: select threshold ───────────────────────
        X_v, cl_v, y_v, kind_v = make_trials(val, unk_val)
        s_v = genuine_scores(model, X_v, cl_v)
        eer_v, thr = compute_eer(y_v, s_v)
        full_report(y_v, s_v, thr, label=f"{name.upper()} VAL")

        # ── 5. Test: evaluate at val threshold ────────────────────
        X_t, cl_t, y_t, kind_t = make_trials(test, unk_test)
        s_t = genuine_scores(model, X_t, cl_t)
        full_report(y_t, s_t, thr, label=f"{name.upper()} TEST")

        for k in ("known_impostor", "unknown_impostor"):
            mask = kind_t == k
            if mask.any():
                accepted = (s_t[mask] >= thr).mean()
                print(f"  accepted {k:20s}: {accepted:.4f}")

        # ── 6. Save ───────────────────────────────────────────────
        bundle = {
            "model":       model,
            "threshold":   thr,
            "classes":     list(model.classes_),
            "feature_dim": get_feature_dim(),
        }
        out = MODELS_DIR / f"fingerprint_{name}.joblib"
        joblib.dump(bundle, out)
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()

