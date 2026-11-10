import os
import glob
import re
import shutil
import joblib
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fingerprint_features import preprocess_fingerprint, extract_fingerprint_features
from metrics import compute_metrics, compute_eer

SOCOFING_REAL_CANDIDATES = [
    os.path.join("data", "socofing", "SOCOFing", "Real"),
    os.path.join("data", "socofing", "Real"),
    os.path.join("data", "SOCOFing", "Real"),
    os.path.join("data", "socofing"),
]

SOCOFING_ALTERED_CANDIDATES = [
    os.path.join("data", "socofing", "SOCOFing", "Altered", "Altered-Easy"),
    os.path.join("data", "socofing", "Altered", "Altered-Easy"),
    os.path.join("data", "SOCOFing", "Altered", "Altered-Easy"),
    os.path.join("data", "socofing", "SOCOFing", "Altered"),
]

def get_socofing_dirs():
    real_dir = None
    for cand in SOCOFING_REAL_CANDIDATES:
        if os.path.exists(cand):
            files = glob.glob(os.path.join(cand, "*.BMP")) + glob.glob(os.path.join(cand, "*.bmp"))
            if len(files) > 0:
                real_dir = cand
                break

    altered_dir = None
    for cand in SOCOFING_ALTERED_CANDIDATES:
        if os.path.exists(cand):
            files = glob.glob(os.path.join(cand, "*.BMP")) + glob.glob(os.path.join(cand, "*.bmp"))
            if len(files) > 0:
                altered_dir = cand
                break

    return real_dir, altered_dir

def load_socofing_dataset(n_subjects=80, seed=42):
    """
    Loads real SOCOFing fingerprint data using rigorous biometric protocol:
    - Enrolled template = Pristine fingerprint from 'Real' directory.
    - Genuine probes = Altered impressions (rotation, scratch, cuts) of the SAME finger from 'Altered' directory.
    - Impostor probes = Altered impressions from DIFFERENT subjects/fingers.
    """
    real_dir, altered_dir = get_socofing_dirs()
    if real_dir is not None and altered_dir is not None:
        print(f"[Task 2] Found real SOCOFing dataset:\n  Real: {real_dir}\n  Altered: {altered_dir}")
        real_imgs = {}
        probe_imgs = {}

        # Look for Right index fingers across subjects
        for s in range(1, 400):
            if len(real_imgs) >= n_subjects:
                break
            r_files = glob.glob(os.path.join(real_dir, f"{s}__*_Right_index_finger.BMP"))
            if len(r_files) > 0:
                a_files = glob.glob(os.path.join(altered_dir, f"{s}__*_Right_index_finger_*.BMP"))
                if len(a_files) >= 2:
                    real_imgs[s] = r_files[0]
                    probe_imgs[s] = a_files[:3]

        valid_subs = list(real_imgs.keys())
        if len(valid_subs) >= 20:
            print(f"[Task 2] Enrolling {len(valid_subs)} unique human fingers from SOCOFing...")
            templates = {}
            probes = {}
            for s in valid_subs:
                u8_r, norm_r = preprocess_fingerprint(real_imgs[s])
                templates[s] = extract_fingerprint_features(u8_r, norm_r)
                probes[s] = []
                for p_path in probe_imgs[s]:
                    u8_p, norm_p = preprocess_fingerprint(p_path)
                    probes[s].append(extract_fingerprint_features(u8_p, norm_p))

            return templates, probes, valid_subs, real_imgs, probe_imgs

    # Fallback to high-quality synthetic fingerprint generation if dataset is not found
    print("[Task 2] Real SOCOFing dataset not found locally. Generating synthetic ridge flow dataset...")
    np.random.seed(seed)
    valid_subs = list(range(n_subjects))
    templates = {}
    probes = {}
    size = (96, 96)
    y_coords, x_coords = np.mgrid[0:size[0], 0:size[1]]

    for sub_id in valid_subs:
        freq = 0.12 + (sub_id % 7) * 0.015
        cx = 48 + np.sin(sub_id) * 10
        cy = 48 + np.cos(sub_id) * 10

        # Template
        r = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        theta = np.arctan2(y_coords - cy, x_coords - cx)
        pat = 0.5 + 0.5 * np.cos(2 * np.pi * freq * r + 2 * theta)
        u8_t, norm_t = preprocess_fingerprint(np.clip(pat, 0, 1).astype(np.float32))
        templates[sub_id] = extract_fingerprint_features(u8_t, norm_t)

        probes[sub_id] = []
        for _ in range(3):
            nx = np.random.uniform(-1.5, 1.5)
            ny = np.random.uniform(-1.5, 1.5)
            r_p = np.sqrt((x_coords - cx - nx)**2 + (y_coords - cy - ny)**2)
            th_p = np.arctan2(y_coords - cy - ny, x_coords - cx - nx)
            pat_p = 0.5 + 0.5 * np.cos(2 * np.pi * freq * r_p + 2 * th_p)
            noise = np.random.normal(0, 0.05, size)
            u8_p, norm_p = preprocess_fingerprint(np.clip(pat_p + noise, 0, 1).astype(np.float32))
            probes[sub_id].append(extract_fingerprint_features(u8_p, norm_p))

    return templates, probes, valid_subs, None, None

def generate_trial_pairs(subjects, templates, probes, impostor_ratio=1.0, seed=42):
    """
    Builds biometric comparison trials (genuine and impostor pairs):
    - Genuine: abs(probe_finger_i - template_finger_i) -> y = 1
    - Impostor: abs(probe_finger_j - template_finger_i) (j != i) -> y = 0
    """
    rng = np.random.RandomState(seed)
    X, y = [], []

    for s in subjects:
        # Genuine comparisons
        for pf in probes[s]:
            X.append(np.abs(pf - templates[s]))
            y.append(1)

        # Impostor comparisons
        other_subs = [x for x in subjects if x != s]
        n_imp = int(len(probes[s]) * impostor_ratio)
        imp_choices = rng.choice(other_subs, size=n_imp, replace=False)
        for imp_s in imp_choices:
            chosen_probe = probes[imp_s][rng.randint(0, len(probes[imp_s]))]
            X.append(np.abs(chosen_probe - templates[s]))
            y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def main():
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs(os.path.join("demo_assets", "fingerprints"), exist_ok=True)

    print("\n======================================================================")
    print("      TASK 2: FINGERPRINT VERIFICATION MODULE (SOCOFing)")
    print("======================================================================")

    templates, probes, valid_subs, real_imgs, probe_imgs = load_socofing_dataset(n_subjects=80, seed=42)
    feat_dim = len(next(iter(templates.values())))
    print(f"[Task 2] Extracted features shape: {feat_dim} dims (Gabor + HOG + LBP)")

    # 70% Train subjects, 30% Unseen Test subjects
    n_train = int(len(valid_subs) * 0.70)
    train_subs = valid_subs[:n_train]
    test_subs = valid_subs[n_train:]
    print(f"[Task 2] Subject split: {len(train_subs)} train identities, {len(test_subs)} unseen test identities.")

    X_train, y_train = generate_trial_pairs(train_subs, templates, probes, impostor_ratio=1.0, seed=42)
    X_test, y_test = generate_trial_pairs(test_subs, templates, probes, impostor_ratio=1.0, seed=123)
    print(f"[Task 2] Trial pairs -> Train: {X_train.shape[0]} ({np.sum(y_train==1)} genuine, {np.sum(y_train==0)} impostor)")
    print(f"[Task 2] Trial pairs -> Test:  {X_test.shape[0]} ({np.sum(y_test==1)} genuine, {np.sum(y_test==0)} impostor)")

    # Build calibrated machine learning pipelines with StandardScaler
    models = {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
        ])
    }

    best_model_name = None
    best_eer = 1.0
    best_bundle = None

    for name, pipeline in models.items():
        print(f"\n--- Training {name} ---")
        pipeline.fit(X_train, y_train)

        test_probs = pipeline.predict_proba(X_test)[:, 1]
        eer, opt_thresh = compute_eer(y_test, test_probs)
        metrics = compute_metrics(y_test, test_probs, threshold=opt_thresh)

        print(f"Results for {name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  FAR:       {metrics['far']:.4f}")
        print(f"  FRR:       {metrics['frr']:.4f}")
        print(f"  EER:       {eer:.4f} (at threshold {opt_thresh:.4f})")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['cm']}")

        if eer < best_eer or (eer == best_eer and name == "SVM (RBF)"):
            best_eer = eer
            best_model_name = name
            best_bundle = {
                "model": pipeline,
                "model_name": name,
                "threshold": opt_thresh,
                "templates": {s: templates[s] for s in test_subs}
            }

    # Enroll Team Members: Ahmed (1000) and Mohamed (1001)
    # Map them to real reference fingerprint templates and export demo assets
    if real_imgs is not None and len(real_imgs) >= 3:
        # User 1000 (Ahmed) mapped to subject 1
        s_ahmed = valid_subs[0]
        best_bundle["templates"][1000] = templates[s_ahmed]
        shutil.copy2(real_imgs[s_ahmed], os.path.join("demo_assets", "fingerprints", "fp_1000.png"))
        if probe_imgs and s_ahmed in probe_imgs and len(probe_imgs[s_ahmed]) > 0:
            shutil.copy2(probe_imgs[s_ahmed][0], os.path.join("demo_assets", "fingerprints", "fp_1000_probe.png"))

        # User 1001 (Mohamed) mapped to subject 2
        s_mohamed = valid_subs[1]
        best_bundle["templates"][1001] = templates[s_mohamed]
        shutil.copy2(real_imgs[s_mohamed], os.path.join("demo_assets", "fingerprints", "fp_1001.png"))
        if probe_imgs and s_mohamed in probe_imgs and len(probe_imgs[s_mohamed]) > 0:
            shutil.copy2(probe_imgs[s_mohamed][0], os.path.join("demo_assets", "fingerprints", "fp_1001_probe.png"))

        # Impostor fingerprint for testing
        s_imp = valid_subs[2]
        shutil.copy2(real_imgs[s_imp], os.path.join("demo_assets", "fingerprints", "fp_impostor.png"))
        print("[Task 2] Enrolled Ahmed (1000) and Mohamed (1001) using real SOCOFing fingerprint templates.")
        print("[Task 2] Exported demo fingerprint files to demo_assets/fingerprints/ (fp_1000.png, fp_1001.png, fp_impostor.png)")
    else:
        rng_fp = np.random.RandomState(42)
        size_fp = (96, 96)
        yc, xc = np.mgrid[0:96, 0:96]
        for uid in [1000, 1001]:
            freq = 0.10 + (uid % 5) * 0.02
            cx = 48 + np.sin(uid) * 12
            cy = 48 + np.cos(uid) * 12
            r = np.sqrt((xc - cx)**2 + (yc - cy)**2)
            th = np.arctan2(yc - cy, xc - cx)
            pat = 0.5 + 0.5 * np.cos(2 * np.pi * freq * r + 2 * th)
            u8, norm = preprocess_fingerprint(np.clip(pat, 0, 1).astype(np.float32))
            best_bundle["templates"][uid] = extract_fingerprint_features(u8, norm)
            cv2.imwrite(os.path.join("demo_assets", "fingerprints", f"fp_{uid}.png"), u8)
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
