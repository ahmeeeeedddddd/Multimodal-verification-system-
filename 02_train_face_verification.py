"""02 — Train Stage 1 Face Verification Model.

Loads Olivetti (baseline 40 subjects) + enrolled team member faces.
Extracts HOG + LBP + PCA representations and trains an RBF SVM + StandardScaler pipeline.
Saves:
- saved_models/face_verifier_team.pkl (with all subjects + team members)
- saved_models/face_verifier.pkl
"""

import os
import cv2
import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from face_features import preprocess_face, FaceFeatureExtractor
from metrics import compute_metrics, compute_eer

TEAM_FACES_DIR = Path("data/team_faces")


def load_team_faces():
    """Load team faces using robust face preprocessing."""
    u8_list, norm_list, labels, names = [], [], [], []
    name_map = {}
    label_id = 1000

    for person_dir in sorted(TEAM_FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        name_map[label_id] = name

        img_files = sorted(
            list(person_dir.glob("*.jpg")) +
            list(person_dir.glob("*.jpeg")) +
            list(person_dir.glob("*.png"))
        )

        valid_count = 0
        for f in img_files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            u8, norm = preprocess_face(img)
            if u8 is None:
                continue

            u8_list.append(u8)
            norm_list.append(norm)
            labels.append(label_id)
            names.append(name)
            valid_count += 1

        print(f"  [02-Face] Loaded {valid_count}/{len(img_files)} faces for '{name}' (label={label_id})")
        label_id += 1

    return u8_list, norm_list, np.array(labels), names, name_map


def prepare_trials(features, labels, enrolled_ids, impostor_ids, enroll_ratio=0.6):
    """Build difference-vector pairs for verification."""
    templates = {}
    for uid in enrolled_ids:
        idxs = np.where(labels == uid)[0]
        if len(idxs) == 0:
            continue
        n_enroll = max(3, int(len(idxs) * enroll_ratio))
        enroll_idxs = idxs[:n_enroll]
        templates[uid] = np.mean(features[enroll_idxs], axis=0)

    X_pairs, y_pairs = [], []
    for uid in enrolled_ids:
        if uid not in templates:
            continue
        t = templates[uid]
        idxs = np.where(labels == uid)[0]
        n_enroll = max(3, int(len(idxs) * enroll_ratio))
        query_idxs = idxs[n_enroll:] if len(idxs) > n_enroll else idxs

        for idx in query_idxs:
            X_pairs.append(np.abs(features[idx] - t))
            y_pairs.append(1)

        imp_pool = np.where(np.isin(labels, impostor_ids) & (labels != uid))[0]
        if len(imp_pool) > 0:
            n_imp = min(len(query_idxs) * 3, len(imp_pool))
            chosen = np.random.choice(imp_pool, size=n_imp, replace=False)
            for idx in chosen:
                X_pairs.append(np.abs(features[idx] - t))
                y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs), templates


def main():
    np.random.seed(42)
    os.makedirs("saved_models", exist_ok=True)

    # 1. Olivetti Baseline
    print("[02-Face] Loading Olivetti faces dataset...")
    raw = fetch_olivetti_faces()
    ol_u8 = []
    ol_norm = []
    for img in raw.images:
        u, n = preprocess_face(img)
        ol_u8.append(u)
        ol_norm.append(n)
    ol_labels = raw.target

    # 2. Team Faces
    print("[02-Face] Loading team faces...")
    team_u8, team_norm, team_labels, team_names, name_map = load_team_faces()

    all_u8 = np.array(ol_u8 + team_u8)
    all_norm = np.array(ol_norm + team_norm)
    all_labels = np.concatenate([ol_labels, team_labels])

    print(f"[02-Face] Total samples: {len(all_labels)} (Olivetti={len(ol_labels)}, Team={len(team_labels)})")

    # 3. PCA Feature Extractor
    extractor = FaceFeatureExtractor(n_pca_components=40)
    extractor.fit(all_norm[all_labels < 28])
    all_features = np.array([extractor.extract(u, n) for u, n in zip(all_u8, all_norm)])
    print(f"[02-Face] Feature dimensionality: {all_features.shape[1]}")

    # 4. Splits & Trials
    train_ids = np.concatenate([np.arange(0, 28), np.unique(team_labels)])
    test_known = np.concatenate([np.arange(28, 35), np.unique(team_labels)])
    test_unknown = np.arange(35, 40)
    test_impostors = np.concatenate([test_known, test_unknown])

    X_train, y_train, train_templates = prepare_trials(
        all_features, all_labels, enrolled_ids=train_ids, impostor_ids=train_ids
    )
    X_test, y_test, test_templates = prepare_trials(
        all_features, all_labels, enrolled_ids=test_known, impostor_ids=test_impostors
    )

    print(f"[02-Face] Train pairs: {len(X_train)}, Test pairs: {len(X_test)}")

    # 5. Model: SVM (RBF) with StandardScaler
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=5.0, probability=True, random_state=42))
    ])

    print("\n--- Training SVM (RBF) with StandardScaler ---")
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    eer, opt_thresh = compute_eer(y_test, test_probs)
    metrics = compute_metrics(y_test, test_probs, threshold=opt_thresh)

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  FAR:       {metrics['far']:.4f}")
    print(f"  FRR:       {metrics['frr']:.4f}")
    print(f"  EER:       {eer:.4f} (optimal threshold={opt_thresh:.4f})")
    print(f"  AUC:       {metrics['auc']:.4f}")

    # Full templates for all enrolled subjects
    full_templates = {}
    for uid in np.unique(all_labels):
        idxs = np.where(all_labels == uid)[0]
        full_templates[uid] = np.mean(all_features[idxs], axis=0)

    bundle = {
        "model": model,
        "model_name": "SVM (RBF) with Scaler",
        "threshold": opt_thresh,
        "extractor": extractor,
        "templates": full_templates,
        "label_to_name": name_map
    }

    joblib.dump(bundle, "saved_models/face_verifier_team.pkl")
    joblib.dump(bundle, "saved_models/face_verifier.pkl")
    print(f"\n[02-Face] Saved models to saved_models/face_verifier_team.pkl and face_verifier.pkl")


if __name__ == "__main__":
    main()
