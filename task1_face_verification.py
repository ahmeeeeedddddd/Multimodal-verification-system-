import os
import joblib
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from face_features import preprocess_face, FaceFeatureExtractor
from metrics import compute_metrics, compute_eer

def prepare_trials(features, labels, enrolled_ids, query_ids, impostor_ids, samples_per_user=10, enroll_count=5):
    templates = {}
    for uid in enrolled_ids:
        idxs = np.where(labels == uid)[0][:enroll_count]
        templates[uid] = np.mean(features[idxs], axis=0)

    X_pairs, y_pairs = [], []

    for uid in enrolled_ids:
        t = templates[uid]
        gen_idxs = np.where(labels == uid)[0][enroll_count:samples_per_user]
        for idx in gen_idxs:
            diff = np.abs(features[idx] - t)
            X_pairs.append(diff)
            y_pairs.append(1)

        imp_pool = np.where(np.isin(labels, impostor_ids))[0]
        chosen_imps = np.random.choice(imp_pool, size=len(gen_idxs), replace=False)
        for idx in chosen_imps:
            diff = np.abs(features[idx] - t)
            X_pairs.append(diff)
            y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs), templates

def main():
    np.random.seed(42)
    os.makedirs("saved_models", exist_ok=True)

    print("[Task 1] Loading Olivetti faces dataset...")
    raw = fetch_olivetti_faces()
    images, labels = raw.images, raw.target

    print("[Task 1] Preprocessing images and extracting features (HOG, LBP, PCA)...")
    preprocessed_norm, preprocessed_uint8 = [], []
    for img in images:
        u8, norm = preprocess_face(img)
        preprocessed_uint8.append(u8)
        preprocessed_norm.append(norm)

    train_mask = labels < 28
    extractor = FaceFeatureExtractor(n_pca_components=40)
    extractor.fit(np.array(preprocessed_norm)[train_mask])

    all_features = np.array([
        extractor.extract(u8, norm) 
        for u8, norm in zip(preprocessed_uint8, preprocessed_norm)
    ])

    train_ids = np.arange(0, 28)
    test_known_ids = np.arange(28, 35)
    test_unknown_ids = np.arange(35, 40)

    X_train, y_train, train_templates = prepare_trials(
        all_features, labels, 
        enrolled_ids=train_ids, 
        query_ids=train_ids, 
        impostor_ids=train_ids
    )

    test_enrolled = test_known_ids
    test_impostors = np.concatenate([test_known_ids, test_unknown_ids])
    X_test, y_test, test_templates = prepare_trials(
        all_features, labels, 
        enrolled_ids=test_enrolled, 
        query_ids=test_enrolled, 
        impostor_ids=test_impostors
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
                "extractor": extractor,
                "templates": test_templates
            }

    save_path = os.path.join("saved_models", "face_verifier.pkl")
    joblib.dump(best_bundle, save_path)
    print(f"\n[Task 1] Saved best face verifier ({best_model_name}) to {save_path}")

def verify_claimed_identity(model_bundle, query_img, claimed_id):
    u8, norm = preprocess_face(query_img)
    feat = model_bundle["extractor"].extract(u8, norm)
    
    if claimed_id not in model_bundle["templates"]:
        return False, 0.0

    template = model_bundle["templates"][claimed_id]
    diff = np.abs(feat - template).reshape(1, -1)
    prob = model_bundle["model"].predict_proba(diff)[0, 1]
    is_accepted = prob >= model_bundle["threshold"]
    return is_accepted, prob

if __name__ == "__main__":
    main()
