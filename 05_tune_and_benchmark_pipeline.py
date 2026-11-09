import os
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from face_features import preprocess_face, FaceFeatureExtractor
from voice_features import get_voice_dataset
from fingerprint_features import preprocess_fingerprint, extract_fingerprint_features
from metrics import compute_metrics, compute_eer, plot_roc_curve, plot_det_curve
from hierarchical_pipeline import HierarchicalVerifier, tune_hierarchical_thresholds
from speech_commands import get_trained_recognizer, CommandExecutor


def build_trial_pairs(feat_matrix, labels, enrolled_ids, impostor_ids, enroll_count=5, samples_per_user=10, seed=42):
    """Generate genuine and impostor pair differences against enrolled templates."""
    rng = np.random.RandomState(seed)
    templates = {}
    for uid in enrolled_ids:
        idxs = np.where(labels == uid)[0][:enroll_count]
        templates[uid] = np.mean(feat_matrix[idxs], axis=0)

    X_pairs, y_pairs, claimed_ids = [], [], []
    for uid in enrolled_ids:
        t = templates[uid]
        gen_idxs = np.where(labels == uid)[0][enroll_count:samples_per_user]
        for idx in gen_idxs:
            X_pairs.append(np.abs(feat_matrix[idx] - t))
            y_pairs.append(1)
            claimed_ids.append(uid)

        imp_pool = np.where(np.isin(labels, impostor_ids))[0]
        chosen_imps = rng.choice(imp_pool, size=len(gen_idxs), replace=False)
        for idx in chosen_imps:
            X_pairs.append(np.abs(feat_matrix[idx] - t))
            y_pairs.append(0)
            claimed_ids.append(uid)

    return np.array(X_pairs), np.array(y_pairs), np.array(claimed_ids), templates


def main():
    os.makedirs("saved_models", exist_ok=True)
    print("=" * 80)
    print("  [Task 6] HIERARCHICAL MULTIMODAL VERIFICATION PIPELINE & BENCHMARK")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/5] Extracting biometric representations for Face, Voice, and Fingerprint...")
    raw_faces = fetch_olivetti_faces()
    face_images, face_labels = raw_faces.images, raw_faces.target

    pre_u8, pre_norm = [], []
    for img in face_images:
        u8, norm = preprocess_face(img)
        pre_u8.append(u8)
        pre_norm.append(norm)

    train_ids = np.arange(0, 21)
    val_ids = np.arange(21, 28)
    test_known_ids = np.arange(28, 35)
    test_unknown_ids = np.arange(35, 40)
    test_impostor_pool = np.concatenate([test_known_ids, test_unknown_ids])

    # Fit PCA on training face partition
    face_extractor = FaceFeatureExtractor(n_pca_components=40)
    train_face_mask = np.isin(face_labels, train_ids)
    face_extractor.fit(np.array(pre_norm)[train_face_mask])

    face_features = np.array([
        face_extractor.extract(u8, norm) 
        for u8, norm in zip(pre_u8, pre_norm)
    ])

    voice_features, voice_labels = get_voice_dataset(n_speakers=40, samples_per_speaker=10)

    # Standardize modalities
    scaler_face = StandardScaler().fit(face_features[train_face_mask])
    scaler_voice = StandardScaler().fit(voice_features[train_face_mask])

    face_std = scaler_face.transform(face_features)
    voice_std = scaler_voice.transform(voice_features)
    fused_features = np.hstack([face_std, voice_std])

    # 2. Build Models on Training Set
    print("\n[2/5] Training individual and fusion models on train partition (IDs 0-20)...")
    # Face trials
    X_tr_face, y_tr_face, _, _ = build_trial_pairs(
        face_features, face_labels, enrolled_ids=train_ids, impostor_ids=train_ids, seed=42
    )
    # Voice trials
    X_tr_voice, y_tr_voice, _, _ = build_trial_pairs(
        voice_features, voice_labels, enrolled_ids=train_ids, impostor_ids=train_ids, seed=42
    )
    # Fusion trials
    X_tr_fusion, y_tr_fusion, _, _ = build_trial_pairs(
        fused_features, face_labels, enrolled_ids=train_ids, impostor_ids=train_ids, seed=42
    )

    clf_face = SVC(kernel="rbf", C=5.0, probability=True, random_state=42).fit(X_tr_face, y_tr_face)
    clf_voice = SVC(kernel="rbf", C=5.0, probability=True, random_state=42).fit(X_tr_voice, y_tr_voice)
    clf_fusion = SVC(kernel="rbf", C=5.0, probability=True, random_state=42).fit(X_tr_fusion, y_tr_fusion)

    # 3. Validation Phase: Tune Upper & Lower Thresholds
    print("\n[3/5] Tuning Stage 1 Upper (theta_high) and Lower (theta_low) Thresholds on Validation Data (IDs 21-27)...")
    X_val_face, y_val, _, _ = build_trial_pairs(
        face_features, face_labels, enrolled_ids=val_ids, impostor_ids=val_ids, seed=42
    )
    X_val_voice, _, _, _ = build_trial_pairs(
        voice_features, voice_labels, enrolled_ids=val_ids, impostor_ids=val_ids, seed=42
    )
    X_val_fusion, _, _, _ = build_trial_pairs(
        fused_features, face_labels, enrolled_ids=val_ids, impostor_ids=val_ids, seed=42
    )

    val_face_probs = clf_face.predict_proba(X_val_face)[:, 1]
    val_voice_probs = clf_voice.predict_proba(X_val_voice)[:, 1]
    val_fusion_probs = clf_fusion.predict_proba(X_val_fusion)[:, 1]
    eer_fusion, opt_thr_fusion = compute_eer(y_val, val_fusion_probs)

    theta_low, theta_high, tune_metrics = tune_hierarchical_thresholds(
        y_val, val_face_probs, val_fusion_probs, theta_fusion=opt_thr_fusion
    )
    print(f"  -> Optimal theta_low  = {theta_low:.3f}")
    print(f"  -> Optimal theta_high = {theta_high:.3f}")
    print(f"  -> Validation Acc = {tune_metrics['accuracy']:.4f} | Stage 2 Escalation Rate = {tune_metrics['stage2_fraction']*100:.1f}%")

    # 4. Evaluation on Identical Test Trials (IDs 28-34 Enrolled, 28-39 Impostors)
    print("\n[4/5] Evaluating Architectures on Identical Test Trials (IDs 28-34)...")
    X_te_face, y_test, _, te_face_tmpl = build_trial_pairs(
        face_features, face_labels, enrolled_ids=test_known_ids, impostor_ids=test_impostor_pool, seed=42
    )
    X_te_voice, _, _, te_voice_tmpl = build_trial_pairs(
        voice_features, voice_labels, enrolled_ids=test_known_ids, impostor_ids=test_impostor_pool, seed=42
    )
    X_te_fusion, _, _, te_fusion_tmpl = build_trial_pairs(
        fused_features, face_labels, enrolled_ids=test_known_ids, impostor_ids=test_impostor_pool, seed=42
    )

    t0 = time.perf_counter()
    p_te_face = clf_face.predict_proba(X_te_face)[:, 1]
    p_te_voice = clf_voice.predict_proba(X_te_voice)[:, 1]
    p_te_fusion = clf_fusion.predict_proba(X_te_fusion)[:, 1]

    # Save models in unified bundle
    face_bundle = {
        "model": clf_face,
        "model_name": "SVM (RBF)",
        "threshold": float(opt_thr_fusion),
        "extractor": face_extractor,
        "templates": te_face_tmpl
    }
    fusion_bundle = {
        "model": clf_fusion,
        "model_name": "SVM (RBF)",
        "threshold": float(opt_thr_fusion),
        "scaler_face": scaler_face,
        "scaler_voice": scaler_voice,
        "face_extractor": face_extractor,
        "templates": te_fusion_tmpl
    }
    joblib.dump(face_bundle, "saved_models/face_verifier.pkl")
    joblib.dump(fusion_bundle, "saved_models/fusion_verifier.pkl")

    # Architectures to compare
    n_trials = len(y_test)

    # 1. Hierarchical Pipeline (Face -> Fusion)
    h_preds = np.zeros(n_trials, dtype=int)
    h_stage2_mask = (p_te_face >= theta_low) & (p_te_face < theta_high)
    h_preds[p_te_face >= theta_high] = 1
    h_preds[p_te_face < theta_low] = 0
    h_preds[h_stage2_mask] = (p_te_fusion[h_stage2_mask] >= opt_thr_fusion).astype(int)
    # Effective continuous confidence score for ROC calculation
    h_effective_scores = np.copy(p_te_face)
    h_effective_scores[h_stage2_mask] = p_te_fusion[h_stage2_mask]

    # 2. Always-Fusion Pipeline
    af_preds = (p_te_fusion >= opt_thr_fusion).astype(int)

    # 3. Voice-First Hierarchical Pipeline
    # Tune voice dual thresholds on val
    vf_low, vf_high, _ = tune_hierarchical_thresholds(
        y_val, val_voice_probs, val_fusion_probs, theta_fusion=opt_thr_fusion
    )
    vf_preds = np.zeros(n_trials, dtype=int)
    vf_stage2_mask = (p_te_voice >= vf_low) & (p_te_voice < vf_high)
    vf_preds[p_te_voice >= vf_high] = 1
    vf_preds[p_te_voice < vf_low] = 0
    vf_preds[vf_stage2_mask] = (p_te_fusion[vf_stage2_mask] >= opt_thr_fusion).astype(int)
    vf_effective_scores = np.copy(p_te_voice)
    vf_effective_scores[vf_stage2_mask] = p_te_fusion[vf_stage2_mask]

    # 4. Face-Only Baseline
    eer_face, thr_face = compute_eer(y_test, p_te_face)
    fo_preds = (p_te_face >= thr_face).astype(int)

    # 5. Voice-Only Baseline
    eer_voice, thr_voice = compute_eer(y_test, p_te_voice)
    vo_preds = (p_te_voice >= thr_voice).astype(int)

    architectures = [
        ("Hierarchical (Face->Fusion)", h_preds, h_effective_scores, np.mean(h_stage2_mask)),
        ("Always-Fusion (Full Multimodal)", af_preds, p_te_fusion, 1.000),
        ("Voice-First (Voice->Fusion)", vf_preds, vf_effective_scores, np.mean(vf_stage2_mask)),
        ("Face-Only Baseline", fo_preds, p_te_face, 0.000),
        ("Voice-Only Baseline", vo_preds, p_te_voice, 0.000)
    ]

    print("\n" + "=" * 90)
    print(f"{'Architecture':<32} | {'Accuracy':<9} | {'FAR':<7} | {'FRR':<7} | {'F1':<7} | {'Stage 2 %':<10} | {'AUC':<7}")
    print("=" * 90)

    curves_dict = {}
    for name, preds, scores, s2_rate in architectures:
        m = compute_metrics(y_test, preds)
        auc_val = compute_metrics(y_test, scores)["auc"]
        curves_dict[name] = (y_test, scores)
        print(f"{name:<32} | {m['accuracy']:<9.4f} | {m['far']:<7.4f} | {m['frr']:<7.4f} | {m['f1']:<7.4f} | {s2_rate*100:<9.1f}% | {auc_val:<7.4f}")
    print("=" * 90)

    # 5. Generate and save publication-quality ROC and DET curves
    print("\n[5/5] Generating ROC and DET comparison figures...")
    det_path = "saved_models/det_comparison.png"
    plot_det_curve(curves_dict, title="DET Curve Comparison (Task 6)", save_path=det_path)
    print(f"  -> Saved DET curve to {det_path}")

    roc_path = "saved_models/roc_comparison.png"
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    for name, (y_t, scores) in curves_dict.items():
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_t, scores)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess")
    ax_roc.set_xlabel("False Accept Rate (FAR)")
    ax_roc.set_ylabel("True Accept Rate (TAR)")
    ax_roc.set_title("ROC Curve: Architecture Comparison (Identical Test Trials)")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(roc_path, dpi=300)
    print(f"  -> Saved ROC curve to {roc_path}")

    # Save complete hierarchical pipeline configuration bundle
    pipeline_bundle = {
        "theta_low": theta_low,
        "theta_high": theta_high,
        "theta_fusion": opt_thr_fusion,
        "face_bundle_path": "saved_models/face_verifier.pkl",
        "fusion_bundle_path": "saved_models/fusion_verifier.pkl",
        "fingerprint_bundle_path": "saved_models/fingerprint_verifier.pkl",
        "enrolled_ids": list(te_fusion_tmpl.keys())
    }
    joblib.dump(pipeline_bundle, "saved_models/hierarchical_pipeline_config.pkl")
    print(f"  -> Saved pipeline configuration bundle to saved_models/hierarchical_pipeline_config.pkl")

    # Ensure Speech Command recognizer is ready
    get_trained_recognizer()
    print("  -> Speech Command Recognizer ready for live demo.\n")
    print("[Task 6] Verification Pipeline & Comparison complete successfully!")


if __name__ == "__main__":
    main()
