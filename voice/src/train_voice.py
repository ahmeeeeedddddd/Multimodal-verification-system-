"""Train and tune the voice verification models (SVM + Random Forest).

Approach: one closed-set multiclass classifier over the ENROLLED speakers.
genuine_probability(claimed) = P(claimed | audio).

Split strategy (session-aware):
  * FSDD speakers : random stratified train/val/test.
  * Team speakers : team_voice -> train; live_test -> val/test (50/50).
  * Unknown speakers (never trained on) are used only for impostor trials.
"""
import warnings

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import FSDD_DIR, MODELS_DIR, RANDOM_STATE
from dataset import (FSDD_ENROLLED, team_speakers, UNKNOWN_SPEAKERS,
                     extract_all, split_by_session, split_unknown,
                     make_trials)
from metrics_basic import compute_eer, auc, far_frr

warnings.filterwarnings("ignore", category=FutureWarning)


def genuine_scores(model, X, claimed):
    """P(claimed identity | audio) for every trial."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    cols = np.array([classes.index(c) for c in claimed])
    return proba[np.arange(len(X)), cols]


def build_models():
    """Return {name: (pipeline, param_grid)}."""
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True,
                    random_state=RANDOM_STATE)),
    ])
    svm_grid = {"clf__C": [1, 10, 100],
                "clf__gamma": ["scale", 0.01, 0.05]}

    rf = Pipeline([
        ("scaler", StandardScaler()),   # harmless for trees, keeps one interface
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_grid = {"clf__n_estimators": [200, 400],
               "clf__max_depth": [None, 15],
               "clf__min_samples_leaf": [1, 3]}

    return {"svm": (svm, svm_grid), "rf": (rf, rf_grid)}


def main():
    # ---- data ----
    team_ids = team_speakers()
    print("Enrolled identities:", FSDD_ENROLLED + team_ids)
    train, val, test = split_by_session(FSDD_ENROLLED, team_ids)
    unknown = extract_all(UNKNOWN_SPEAKERS, folders=[FSDD_DIR])
    unk_val, unk_test = split_unknown(unknown)
    print(f"train={len(train.X)} val={len(val.X)} test={len(test.X)}")

    # ---- tune on TRAIN only (stratified CV) ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, (pipe, grid) in build_models().items():
        print(f"\n=== Tuning {name.upper()} ===")
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy",
                          n_jobs=-1, verbose=1)
        gs.fit(train.X, train.speaker)
        model = gs.best_estimator_
        print("Best params :", gs.best_params_)
        print(f"CV accuracy : {gs.best_score_:.4f}")

        # ---- evaluate verification on VAL trials (threshold selection) ----
        X, claimed, y, kind = make_trials(val, unk_val)
        s = genuine_scores(model, X, claimed)
        eer, thr = compute_eer(y, s)
        far, frr = far_frr(y, s, thr)
        print(f"VAL  : AUC={auc(y, s):.4f}  EER={eer:.4f}  "
              f"thr@EER={thr:.3f}  FAR={far:.4f}  FRR={frr:.4f}")

        # ---- final check on TEST using the val threshold ----
        Xt, claimed_t, yt, kind_t = make_trials(test, unk_test)
        st = genuine_scores(model, Xt, claimed_t)
        far_t, frr_t = far_frr(yt, st, thr)
        print(f"TEST : AUC={auc(yt, st):.4f}  EER={compute_eer(yt, st)[0]:.4f}  "
              f"FAR={far_t:.4f}  FRR={frr_t:.4f}  (val threshold)")
        for k in ("known_impostor", "unknown_impostor"):
            m = kind_t == k
            print(f"       accepted {k:16s}: "
                  f"{(st[m] >= thr).mean():.4f}")

        # ---- save ----
        joblib.dump({"model": model, "threshold": thr,
                     "classes": list(model.classes_)},
                    MODELS_DIR / f"voice_{name}.joblib")
        print(f"Saved -> {MODELS_DIR / f'voice_{name}.joblib'}")


if __name__ == "__main__":
    main()