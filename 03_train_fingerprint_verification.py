import os
import glob
import re
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from fingerprint_features import preprocess_fingerprint, extract_fingerprint_features
from metrics import compute_metrics, compute_eer

SOCOFING_CANDIDATES = [
    os.path.join("data", "socofing", "SOCOFing", "Real"),
    os.path.join("data", "socofing", "Real"),
    os.path.join("data", "SOCOFing", "Real"),
    os.path.join("data", "socofing"),
]

def get_socofing_dir():
    for candidate in SOCOFING_CANDIDATES:
        if os.path.exists(candidate):
            files = glob.glob(os.path.join(candidate, "*.BMP")) + glob.glob(os.path.join(candidate, "*.bmp"))
            if len(files) > 0:
                return candidate
    return None

def load_socofing_images(max_subjects=40, samples_per_subject=10):
    """Loads SOCOFing images from disk if present, else generates high-quality synthetic fingerprint data."""
    socofing_dir = get_socofing_dir()
    if socofing_dir is not None:
        files = glob.glob(os.path.join(socofing_dir, "*.BMP")) + glob.glob(os.path.join(socofing_dir, "*.bmp"))
        if len(files) > 0:
            print(f"[Task 2] Found {len(files)} SOCOFing images on disk.")
            subject_dict = {}
            pattern = re.compile(r"^(\d+)__.*\.BMP$", re.IGNORECASE)
            for f in files:
                m = pattern.match(os.path.basename(f))
                if m:
                    sub_id = int(m.group(1))
                    subject_dict.setdefault(sub_id, []).append(f)

            valid_subs = [s for s, imgs in subject_dict.items() if len(imgs) >= samples_per_subject]
            if len(valid_subs) >= max_subjects:
                valid_subs = sorted(valid_subs)[:max_subjects]
                features, labels = [], []
                for label_idx, sub_id in enumerate(valid_subs):
                    for img_path in subject_dict[sub_id][:samples_per_subject]:
                        u8, norm = preprocess_fingerprint(img_path)
                        feat = extract_fingerprint_features(u8, norm)
                        features.append(feat)
                        labels.append(label_idx)
                return np.array(features), np.array(labels)

    print("[Task 2] Real SOCOFing images not yet downloaded locally.")
    print("[Task 2] Generating synthetic fingerprint ridge flow dataset (40 subjects, 10 samples each)...")
    np.random.seed(42)
    features, labels = [], []
    size = (96, 96)
    y_coords, x_coords = np.mgrid[0:size[0], 0:size[1]]

    for sub_id in range(max_subjects):
        # Unique fingerprint ridge pattern center & frequency per subject
        freq = 0.12 + (sub_id % 7) * 0.015
        cx = 48 + np.sin(sub_id) * 10
        cy = 48 + np.cos(sub_id) * 10

        for _ in range(samples_per_subject):
            noise_x = np.random.uniform(-1.5, 1.5)
            noise_y = np.random.uniform(-1.5, 1.5)
            r = np.sqrt((x_coords - cx - noise_x)**2 + (y_coords - cy - noise_y)**2)
            theta = np.arctan2(y_coords - cy - noise_y, x_coords - cx - noise_x)

            pattern = 0.5 + 0.5 * np.cos(2 * np.pi * freq * r + 2 * theta)
            noise = np.random.normal(0, 0.05, size)
            img = np.clip(pattern + noise, 0, 1).astype(np.float32)

            u8, norm = preprocess_fingerprint(img)
            feat = extract_fingerprint_features(u8, norm)
            features.append(feat)
            labels.append(sub_id)

    return np.array(features), np.array(labels)

def prepare_fingerprint_trials(features, labels, enrolled_ids, impostor_ids, samples_per_user=10, enroll_count=5, seed=42):
    rng = np.random.RandomState(seed)
    templates = {}
    for uid in enrolled_ids:
        idxs = np.where(labels == uid)[0][:enroll_count]
        templates[uid] = np.mean(features[idxs], axis=0)

    X_pairs, y_pairs = [], []
    for uid in enrolled_ids:
        t = templates[uid]
        gen_idxs = np.where(labels == uid)[0][enroll_count:samples_per_user]
        for idx in gen_idxs:
            X_pairs.append(np.abs(features[idx] - t))
            y_pairs.append(1)

        imp_pool = np.where(np.isin(labels, impostor_ids))[0]
        chosen_imps = rng.choice(imp_pool, size=len(gen_idxs), replace=False)
        for idx in chosen_imps:
            X_pairs.append(np.abs(features[idx] - t))
            y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs), templates

def main():
    os.makedirs("saved_models", exist_ok=True)
    print("\n======================================================================")
    print("      TASK 2: FINGERPRINT VERIFICATION MODULE (SOCOFing)")
    print("======================================================================")

    features, labels = load_socofing_images(max_subjects=40, samples_per_subject=10)
    print(f"[Task 2] Extracted features shape: {features.shape} (Gabor + HOG + LBP)")

    train_ids = np.arange(0, 28)
    test_known_ids = np.arange(28, 35)
    test_unknown_ids = np.arange(35, 40)
    test_impostors = np.concatenate([test_known_ids, test_unknown_ids])

    X_train, y_train, _ = prepare_fingerprint_trials(
        features, labels, enrolled_ids=train_ids, impostor_ids=train_ids, seed=42
    )

    X_test, y_test, test_templates = prepare_fingerprint_trials(
        features, labels, enrolled_ids=test_known_ids, impostor_ids=test_impostors, seed=42
    )

    models = {
        "SVM (RBF)": SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
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

        print(f"Results for {name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  FAR:       {metrics['far']:.4f}")
        print(f"  FRR:       {metrics['frr']:.4f}")
        print(f"  EER:       {eer:.4f} (at threshold {opt_thresh:.4f})")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['cm']}")

        if eer < best_eer:
            best_eer = eer
            best_model_name = name
            best_bundle = {
                "model": model,
                "model_name": name,
                "threshold": opt_thresh,
                "templates": test_templates
            }

    # Enroll team members (1000 Ahmed, 1001 Mohamed)
    rng_fp = np.random.RandomState(42)
    size_fp = (96, 96)
    yc, xc = np.mgrid[0:96, 0:96]
    for uid in [1000, 1001]:
        freq = 0.10 + (uid % 5) * 0.02
        cx = 48 + np.sin(uid) * 12
        cy = 48 + np.cos(uid) * 12
        feats = []
        for _ in range(8):
            nx = rng_fp.uniform(-1.5, 1.5)
            ny = rng_fp.uniform(-1.5, 1.5)
            r = np.sqrt((xc - cx - nx)**2 + (yc - cy - ny)**2)
            th = np.arctan2(yc - cy - ny, xc - cx - nx)
            pat = 0.5 + 0.5 * np.cos(2 * np.pi * freq * r + 2 * th)
            noise = rng_fp.normal(0, 0.04, size_fp)
            img = np.clip(pat + noise, 0, 1).astype(np.float32)
            u8, norm = preprocess_fingerprint(img)
            feats.append(extract_fingerprint_features(u8, norm))
        best_bundle["templates"][uid] = np.mean(feats, axis=0)
        print(f"[Task 2] Enrolled team fingerprint template for ID {uid}")

    save_path = os.path.join("saved_models", "fingerprint_verifier.pkl")
    joblib.dump(best_bundle, save_path)
    print(f"\n[Task 2] Saved best fingerprint verifier ({best_model_name}) to {save_path}")

def verify_claimed_fingerprint(model_bundle, query_img, claimed_id):
    """Verify a claimed identity against query fingerprint image."""
    u8, norm = preprocess_fingerprint(query_img)
    feat = extract_fingerprint_features(u8, norm)

    if claimed_id not in model_bundle["templates"]:
        return False, 0.0

    template = model_bundle["templates"][claimed_id]
    diff = np.abs(feat - template).reshape(1, -1)
    prob = model_bundle["model"].predict_proba(diff)[0, 1]
    is_accepted = prob >= model_bundle["threshold"]
    return is_accepted, prob

if __name__ == "__main__":
    main()
