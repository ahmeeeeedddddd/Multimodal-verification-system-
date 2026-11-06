import os
import joblib
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from face_features import preprocess_face, FaceFeatureExtractor
from voice_features import get_voice_dataset
from metrics import compute_metrics, compute_eer

def build_trial_pairs(feat_matrix, labels, enrolled_ids, impostor_ids, enroll_count=5, samples_per_user=10, seed=42):
    rng = np.random.RandomState(seed)
    templates = {}
    for uid in enrolled_ids:
        idxs = np.where(labels == uid)[0][:enroll_count]
        templates[uid] = np.mean(feat_matrix[idxs], axis=0)

    X_pairs, y_pairs = [], []
    for uid in enrolled_ids:
        t = templates[uid]
        gen_idxs = np.where(labels == uid)[0][enroll_count:samples_per_user]
        for idx in gen_idxs:
            X_pairs.append(np.abs(feat_matrix[idx] - t))
            y_pairs.append(1)

        imp_pool = np.where(np.isin(labels, impostor_ids))[0]
        chosen_imps = rng.choice(imp_pool, size=len(gen_idxs), replace=False)
        for idx in chosen_imps:
            X_pairs.append(np.abs(feat_matrix[idx] - t))
            y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs), templates

def main():
    os.makedirs("saved_models", exist_ok=True)
    print("[Task 5] Early-Fusion Multimodal Verification Pipeline\n")

    print("1. Extracting Face Features (HOG + LBP + PCA)...")
    raw_faces = fetch_olivetti_faces()
    face_images, face_labels = raw_faces.images, raw_faces.target

    pre_u8, pre_norm = [], []
    for img in face_images:
        u8, norm = preprocess_face(img)
        pre_u8.append(u8)
        pre_norm.append(norm)

    train_mask = face_labels < 28
    face_extractor = FaceFeatureExtractor(n_pca_components=40)
    face_extractor.fit(np.array(pre_norm)[train_mask])
    face_features = np.array([
        face_extractor.extract(u8, norm) 
        for u8, norm in zip(pre_u8, pre_norm)
    ])

    print("2. Extracting Voice Features (MFCC + ZCR + RMS)...")
    voice_features, voice_labels = get_voice_dataset(n_speakers=40, samples_per_speaker=10)

    print("3. Creating Chimeric Users and Standardizing Modalities...")
    scaler_face = StandardScaler().fit(face_features[train_mask])
    scaler_voice = StandardScaler().fit(voice_features[train_mask])

    face_norm = scaler_face.transform(face_features)
    voice_norm = scaler_voice.transform(voice_features)
    fused_features = np.hstack([face_norm, voice_norm])

    train_ids = np.arange(0, 28)
    test_known_ids = np.arange(28, 35)
    test_unknown_ids = np.arange(35, 40)
    test_impostors = np.concatenate([test_known_ids, test_unknown_ids])

    modalities = {
        "Face-Only": face_norm,
        "Voice-Only": voice_norm,
        "Early Fusion": fused_features
    }

    model_factories = {
        "SVM (RBF)": lambda: SVC(kernel='rbf', C=5.0, probability=True, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }

    best_bundle = None
    best_eer = 1.0

    for model_name, factory in model_factories.items():
        print(f"\n======================================================================")
        print(f"       THREE-WAY COMPARISON: {model_name.upper()} (IDENTICAL TEST TRIALS)")
        print(f"======================================================================")
        print(f"{'Modality':<16} | {'Accuracy':<10} | {'FAR':<8} | {'FRR':<8} | {'EER':<8} | {'AUC':<8}")
        print("-" * 70)

        for mod_name, mat in modalities.items():
            X_tr, y_tr, _ = build_trial_pairs(
                mat, face_labels, 
                enrolled_ids=train_ids, 
                impostor_ids=train_ids, 
                seed=42
            )
            X_te, y_te, te_templates = build_trial_pairs(
                mat, face_labels, 
                enrolled_ids=test_known_ids, 
                impostor_ids=test_impostors, 
                seed=42
            )

            clf = factory().fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
            eer, opt_thresh = compute_eer(y_te, probs)
            m = compute_metrics(y_te, probs, threshold=opt_thresh)

            print(f"{mod_name:<16} | {m['accuracy']:<10.4f} | {m['far']:<8.4f} | {m['frr']:<8.4f} | {eer:<8.4f} | {m['auc']:<8.4f}")

            if mod_name == "Early Fusion" and eer < best_eer:
                best_eer = eer
                best_bundle = {
                    "model": clf,
                    "model_name": model_name,
                    "threshold": opt_thresh,
                    "scaler_face": scaler_face,
                    "scaler_voice": scaler_voice,
                    "face_extractor": face_extractor,
                    "templates": te_templates
                }

    save_path = os.path.join("saved_models", "fusion_verifier.pkl")
    joblib.dump(best_bundle, save_path)
    print(f"\n[Task 5] Saved best early fusion verifier to {save_path}")

if __name__ == "__main__":
    main()
