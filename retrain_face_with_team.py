"""
Bonus 3 — Retrain face verifier including team member faces.

Loads Olivetti (baseline) + team faces from data/team_faces/<name>/*.jpg
Automatically rejects invalid false-positive crops (e.g. collar/chest boxes).
Uses StandardScaler + RBF SVM pipeline for robust lighting-invariant verification.
Saves to saved_models/face_verifier_team.pkl.
"""
import os
import cv2
import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from face_features import preprocess_face, FaceFeatureExtractor, face_cascade
from metrics import compute_metrics, compute_eer

TEAM_FACES_DIR = Path("data/team_faces")


def extract_clean_face(img):
    """Detect and crop the real face, filtering out collar/chest artifacts."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80)
    )
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=2, minSize=(60, 60)
        )

    # Valid face criteria: upper half of image (y < h * 0.55) and size >= 100px
    valid = [f for f in faces if f[1] < h * 0.55 and f[2] >= 100]
    if len(valid) == 0:
        # Fallback if image was already pre-cropped
        valid = [f for f in faces if f[2] >= 80]
        if len(valid) == 0:
            return None, None

    x, y, fw, fh = max(valid, key=lambda b: b[2] * b[3])
    crop = gray[y:y+fh, x:x+fw]
    resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    return resized, norm


def load_team_faces():
    """Load team faces with automatic bad-crop filtering."""
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
            u8, norm = extract_clean_face(img)
            if u8 is None:
                print(f"  [Filter] Skipped invalid crop: {person_dir.name}/{f.name}")
                continue

            u8_list.append(u8)
            norm_list.append(norm)
            labels.append(label_id)
            names.append(name)
            valid_count += 1

        print(f"  Loaded {valid_count}/{len(img_files)} valid faces for '{name}' (label={label_id})")
        label_id += 1

    return u8_list, norm_list, np.array(labels), names, name_map


def prepare_trials(features, labels, enrolled_ids, impostor_ids, enroll_ratio=0.6):
    """Build difference-vector pairs for verification."""
    templates = {}
    for uid in enrolled_ids:
        idxs = np.where(labels == uid)[0]
        if len(idxs) == 0:
            continue
        # Use first 60% of samples (at least 3) for template
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
            n_imp = min(len(query_idxs) * 2, len(imp_pool))
            chosen = np.random.choice(imp_pool, size=n_imp, replace=False)
            for idx in chosen:
                X_pairs.append(np.abs(features[idx] - t))
                y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs), templates


def main():
    np.random.seed(42)
    os.makedirs("saved_models", exist_ok=True)

    # 1. Olivetti Baseline
    print("[Bonus3-Face] Loading Olivetti dataset...")
    raw = fetch_olivetti_faces()
    ol_u8 = [cv2.resize((img * 255).astype(np.uint8), (64, 64)) for img in raw.images]
    ol_norm = [cv2.resize(img.astype(np.float32), (64, 64)) for img in raw.images]
    ol_labels = raw.target

    # 2. Team Faces
    print("[Bonus3-Face] Loading team faces with quality filter...")
    team_u8, team_norm, team_labels, team_names, name_map = load_team_faces()

    all_u8 = np.array(ol_u8 + team_u8)
    all_norm = np.array(ol_norm + team_norm)
    all_labels = np.concatenate([ol_labels, team_labels])

    print(f"[Bonus3-Face] Total samples: {len(all_labels)} (Olivetti={len(ol_labels)}, Team={len(team_labels)})")

    # 3. PCA Feature Extractor
    extractor = FaceFeatureExtractor(n_pca_components=40)
    extractor.fit(all_norm[all_labels < 28])
    all_features = np.array([extractor.extract(u, n) for u, n in zip(all_u8, all_norm)])
    print(f"[Bonus3-Face] Feature dim: {all_features.shape[1]}")

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

    print(f"[Bonus3-Face] Train pairs: {len(X_train)}, Test pairs: {len(X_test)}")

    # 5. Models (with StandardScaler)
    models = {
        "SVM (RBF) with Scaler": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, probability=True, random_state=42))
        ]),
        "Random Forest with Scaler": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42))
        ])
    }

    best_bundle = None
    best_eer = 1.0
    best_name = None

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

        if eer <= best_eer:
            best_eer = eer
            best_name = name

            # Combine templates: all enrolled subjects
            full_templates = {}
            for uid in np.unique(all_labels):
                idxs = np.where(all_labels == uid)[0]
                full_templates[uid] = np.mean(all_features[idxs], axis=0)

            best_bundle = {
                "model": model,
                "model_name": name,
                "threshold": opt_thresh,
                "extractor": extractor,
                "templates": full_templates,
                "label_to_name": name_map
            }

    save_path = os.path.join("saved_models", "face_verifier_team.pkl")
    joblib.dump(best_bundle, save_path)
    print(f"\n[Bonus3-Face] Saved best face verifier ({best_name}, EER={best_eer:.4f}) to {save_path}")
    print(f"[Bonus3-Face] Enrolled subjects: {name_map}")


if __name__ == "__main__":
    main()
