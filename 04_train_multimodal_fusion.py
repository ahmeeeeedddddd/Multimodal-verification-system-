import os
import glob
import cv2
import joblib
import numpy as np
import soundfile as sf
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_features import preprocess_face, FaceFeatureExtractor
from voice_features import extract_voice_features, get_voice_dataset
from metrics import compute_metrics, compute_eer


def main():
    np.random.seed(42)
    os.makedirs("saved_models", exist_ok=True)
    print("[04-Fusion] Extracting Face and Voice Features...")

    # 1. Olivetti Face Features
    raw_faces = fetch_olivetti_faces()
    ol_face_u8, ol_face_norm = [], []
    for img in raw_faces.images:
        u8, norm = preprocess_face(img)
        ol_face_u8.append(u8)
        ol_face_norm.append(norm)

    # 2. Team Face Features
    team_face_u8 = {1000: [], 1001: []}
    team_face_norm = {1000: [], 1001: []}
    for f in sorted(glob.glob("data/team_faces/ahmed/*")):
        img = cv2.imread(f)
        if img is not None:
            u8, norm = preprocess_face(img)
            team_face_u8[1000].append(u8)
            team_face_norm[1000].append(norm)

    for f in sorted(glob.glob("data/team_faces/mohamed/*")):
        img = cv2.imread(f)
        if img is not None:
            u8, norm = preprocess_face(img)
            team_face_u8[1001].append(u8)
            team_face_norm[1001].append(norm)

    # Fit Face PCA on Olivetti training set (IDs < 28)
    face_extractor = FaceFeatureExtractor(n_pca_components=40)
    train_mask = raw_faces.target < 28
    face_extractor.fit(np.array(ol_face_norm)[train_mask])

    ol_face_feats = np.array([
        face_extractor.extract(u8, norm) 
        for u8, norm in zip(ol_face_u8, ol_face_norm)
    ])

    team_face_feats = {}
    for uid in [1000, 1001]:
        team_face_feats[uid] = np.array([
            face_extractor.extract(u8, norm)
            for u8, norm in zip(team_face_u8[uid], team_face_norm[uid])
        ])

    # 3. Voice Features
    ol_voice_feats, ol_voice_labels = get_voice_dataset(n_speakers=40, samples_per_speaker=10)

    team_voice_feats = {1000: [], 1001: []}
    for f in sorted(glob.glob("data/team_voice/*_ahmed_*.wav")):
        y, sr = sf.read(f)
        team_voice_feats[1000].append(extract_voice_features(y, sr=sr))
    for f in sorted(glob.glob("data/team_voice/*_mohamed_*.wav")):
        y, sr = sf.read(f)
        team_voice_feats[1001].append(extract_voice_features(y, sr=sr))

    team_voice_feats[1000] = np.array(team_voice_feats[1000])
    team_voice_feats[1001] = np.array(team_voice_feats[1001])

    # Standardizers
    all_train_face = np.vstack([
        ol_face_feats[train_mask],
        team_face_feats[1000][:6],
        team_face_feats[1001][:4]
    ])
    all_train_voice = np.vstack([
        ol_voice_feats[train_mask],
        team_voice_feats[1000][:30],
        team_voice_feats[1001][:30]
    ])

    scaler_face = StandardScaler().fit(all_train_face)
    scaler_voice = StandardScaler().fit(all_train_voice)

    # Multimodal representations
    ol_fused = np.hstack([
        scaler_face.transform(ol_face_feats),
        scaler_voice.transform(ol_voice_feats)
    ])

    team_fused = {}
    for uid in [1000, 1001]:
        n_faces = len(team_face_feats[uid])
        v_subset = team_voice_feats[uid][:n_faces] if len(team_voice_feats[uid]) >= n_faces else team_voice_feats[uid]
        f_norm = scaler_face.transform(team_face_feats[uid][:len(v_subset)])
        v_norm = scaler_voice.transform(v_subset)
        team_fused[uid] = np.hstack([f_norm, v_norm])

    # 4. Multimodal Templates
    templates = {}
    for uid in range(28, 35):
        idxs = np.where(raw_faces.target == uid)[0][:5]
        templates[uid] = np.mean(ol_fused[idxs], axis=0)

    for uid in [1000, 1001]:
        face_tmpl_norm = scaler_face.transform(np.mean(team_face_feats[uid], axis=0, keepdims=True))
        voice_tmpl_norm = scaler_voice.transform(np.mean(team_voice_feats[uid], axis=0, keepdims=True))
        templates[uid] = np.hstack([face_tmpl_norm, voice_tmpl_norm])[0]

    # 5. Build Training Trials
    X_train, y_train = [], []
    for uid in range(0, 28):
        idxs = np.where(raw_faces.target == uid)[0]
        t = np.mean(ol_fused[idxs[:5]], axis=0)
        for idx in idxs[5:]:
            X_train.append(np.abs(ol_fused[idx] - t))
            y_train.append(1)
        imp_idxs = np.where(raw_faces.target != uid)[0]
        chosen = np.random.choice(imp_idxs, size=len(idxs[5:]) * 2, replace=False)
        for idx in chosen:
            X_train.append(np.abs(ol_fused[idx] - t))
            y_train.append(0)

    for uid in [1000, 1001]:
        t = templates[uid]
        for row in team_fused[uid]:
            X_train.append(np.abs(row - t))
            y_train.append(1)
        other_uid = 1001 if uid == 1000 else 1000
        for row in team_fused[other_uid]:
            X_train.append(np.abs(row - t))
            y_train.append(0)
        imp_ol = np.random.choice(len(ol_fused), size=len(team_fused[uid]), replace=False)
        for idx in imp_ol:
            X_train.append(np.abs(ol_fused[idx] - t))
            y_train.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"[04-Fusion] Total training pairs: {len(X_train)} (Positive: {(y_train == 1).sum()}, Negative: {(y_train == 0).sum()})")

    clf_fusion = SVC(kernel="rbf", C=5.0, probability=True, random_state=42)
    clf_fusion.fit(X_train, y_train)

    for uid, name in [(1000, "Ahmed"), (1001, "Mohamed")]:
        diff_gen = np.abs(team_fused[uid] - templates[uid])
        probs_gen = clf_fusion.predict_proba(diff_gen)[:, 1]
        print(f"  [04-Fusion] {name} genuine fusion scores: min={probs_gen.min():.3f}, mean={probs_gen.mean():.3f}, max={probs_gen.max():.3f}")

        other_uid = 1001 if uid == 1000 else 1000
        diff_imp = np.abs(team_fused[other_uid] - templates[uid])
        probs_imp = clf_fusion.predict_proba(diff_imp)[:, 1]
        print(f"  [04-Fusion] Impostor claiming {name} fusion scores: min={probs_imp.min():.3f}, mean={probs_imp.mean():.3f}, max={probs_imp.max():.3f}")

    fusion_bundle = {
        "model": clf_fusion,
        "model_name": "SVM (RBF)",
        "threshold": 0.50,
        "scaler_face": scaler_face,
        "scaler_voice": scaler_voice,
        "face_extractor": face_extractor,
        "templates": templates
    }

    joblib.dump(fusion_bundle, "saved_models/fusion_verifier.pkl")
    print(f"[04-Fusion] Successfully saved saved_models/fusion_verifier.pkl")


if __name__ == "__main__":
    main()
