"""
Bonus 3 — Retrain face verifier including team member faces.

Loads Olivetti (baseline) + team faces from data/team_faces/<name>/*.jpg
Saves updated saved_models/face_verifier.pkl with team members enrolled.
"""
import os
import cv2
import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from face_features import preprocess_face, FaceFeatureExtractor
from metrics import compute_metrics, compute_eer

TEAM_FACES_DIR = Path("data/team_faces")


def load_team_faces():
    """Load team face images and assign integer label IDs starting from 1000."""
    images, labels, names = [], [], []
    label_id = 1000
    for person_dir in sorted(TEAM_FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        img_files = sorted(
            list(person_dir.glob("*.jpg")) +
            list(person_dir.glob("*.jpeg")) +
            list(person_dir.glob("*.png"))
        )
        for f in img_files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"  WARNING: could not read {f}, skipping")
                continue
            images.append(img)
            labels.append(label_id)
            names.append(name)
        print(f"  Loaded {len(img_files)} images for '{name}' (label={label_id})")
        label_id += 1
    return images, np.array(labels), names


def prepare_trials(features, labels, enrolled_ids, impostor_ids, enroll_count=5):
    """Build difference-vector trials for binary verification."""
    templates = {}
    for uid in enrolled_ids:
        idxs = np.where(labels == uid)[0][:enroll_count]
        if len(idxs) == 0:
            continue
        templates[uid] = np.mean(features[idxs], axis=0)

    X_pairs, y_pairs = [], []
    for uid in enrolled_ids:
        if uid not in templates:
            continue
        t = templates[uid]
        query_idxs = np.where(labels == uid)[0][enroll_count:]
        if len(query_idxs) == 0:
            continue
        for idx in query_idxs:
            X_pairs.append(np.abs(features[idx] - t))
            y_pairs.append(1)

        imp_pool = np.where(np.isin(labels, impostor_ids))[0]
        if len(imp_pool) == 0:
            continue
        n_imp = min(len(query_idxs), len(imp_pool))
        chosen = np.random.choice(imp_pool, size=n_imp, replace=False)
        for idx in chosen:
            X_pairs.append(np.abs(features[idx] - t))
            y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs), templates


def main():
    np.random.seed(42)
    os.makedirs("saved_models", exist_ok=True)

    # --- Load Olivetti ---
    print("[Bonus3-Face] Loading Olivetti dataset...")
    raw = fetch_olivetti_faces()
    ol_images_raw, ol_labels = raw.images, raw.target

    print("[Bonus3-Face] Preprocessing Olivetti images...")
    ol_u8, ol_norm = [], []
    for img in ol_images_raw:
        u8, norm = preprocess_face(img)
        ol_u8.append(u8)
        ol_norm.append(norm)

    # --- Load team faces ---
    print("[Bonus3-Face] Loading team face images...")
    team_imgs_raw, team_labels, team_names = load_team_faces()

    print("[Bonus3-Face] Preprocessing team images...")
    team_u8, team_norm = [], []
    for img in team_imgs_raw:
        u8, norm = preprocess_face(img)
        team_u8.append(u8)
        team_norm.append(norm)

    # --- Combine ---
    all_u8   = np.array(ol_u8 + team_u8)
    all_norm = np.array(ol_norm + team_norm)
    all_labels = np.concatenate([ol_labels, team_labels])

    print(f"[Bonus3-Face] Total samples: {len(all_labels)} "
          f"(Olivetti={len(ol_labels)}, Team={len(team_labels)})")

    # --- Fit PCA on training split (Olivetti ids 0-27 + all team faces) ---
    ol_train_mask = ol_labels < 28
    team_mask = np.ones(len(team_labels), dtype=bool)   # all team faces in PCA fit
    pca_fit_mask = np.concatenate([ol_train_mask, team_mask])

    extractor = FaceFeatureExtractor(n_pca_components=40)
    extractor.fit(all_norm[pca_fit_mask])

    # --- Extract features ---
    all_features = np.array([
        extractor.extract(u8, norm)
        for u8, norm in zip(all_u8, all_norm)
    ])
    print(f"[Bonus3-Face] Feature dim: {all_features.shape[1]}")

    # --- Define splits ---
    ol_train_ids  = np.arange(0, 28)
    ol_test_known = np.arange(28, 35)
    ol_test_unk   = np.arange(35, 40)
    team_ids      = np.unique(team_labels)   # e.g. [1000, 1001]

    train_ids    = np.concatenate([ol_train_ids, team_ids])
    impostor_ids = np.concatenate([ol_test_known, ol_test_unk, team_ids])

    X_train, y_train, train_templates = prepare_trials(
        all_features, all_labels,
        enrolled_ids=train_ids,
        impostor_ids=train_ids,
    )

    X_test, y_test, test_templates = prepare_trials(
        all_features, all_labels,
        enrolled_ids=np.concatenate([ol_test_known, team_ids]),
        impostor_ids=impostor_ids,
    )

    print(f"[Bonus3-Face] Train pairs: {len(X_train)}, Test pairs: {len(X_test)}")

    # --- Train models ---
    models = {
        "SVM (RBF)": SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    }

    best_model_name = None
    best_eer = 1.0
    best_bundle = None

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)

        test_probs = model.predict_proba(X_test)[:, 1]
        eer, opt_thresh = compute_eer(y_test, test_probs)
        metrics = compute_metrics(y_test, test_probs, threshold=opt_thresh)

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  FAR:       {metrics['far']:.4f}")
        print(f"  FRR:       {metrics['frr']:.4f}")
        print(f"  EER:       {eer:.4f} (at threshold {opt_thresh:.4f})")
        print(f"  AUC:       {metrics['auc']:.4f}")

        if eer < best_eer:
            best_eer = eer
            best_model_name = name
            best_bundle = {
                "model": model,
                "model_name": name,
                "threshold": opt_thresh,
                "extractor": extractor,
                "templates": test_templates,
                "team_names": {v: k for k, v in zip(team_names, team_labels)
                               if k == team_labels[team_names.index(k)]},
            }

    # Build proper name map
    name_map = {}
    for lbl, nm in zip(team_labels, team_names):
        name_map[int(lbl)] = nm
    best_bundle["label_to_name"] = name_map

    save_path = os.path.join("saved_models", "face_verifier_team.pkl")
    joblib.dump(best_bundle, save_path)
    print(f"\n[Bonus3-Face] Saved best face verifier ({best_model_name}, EER={best_eer:.4f}) to {save_path}")
    print(f"[Bonus3-Face] Team members enrolled: {list(name_map.values())}")


if __name__ == "__main__":
    main()
