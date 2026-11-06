# Task 2 – Fingerprint Verification Module

**Dataset:** [Sokoto Coventry Fingerprint Dataset (SOCOFing)](https://www.kaggle.com/datasets/ruizgara/socofing)
**Access:** Kaggle API (no manual browser download required)

---

## Module layout

```
fingerprint/
├── models/                  # saved model bundles (fingerprint_svm.joblib, fingerprint_rf.joblib)
├── notebooks/               # exploration notebooks (optional)
└── src/
    ├── config.py            # all hyper-parameters and paths
    ├── download_dataset.py  # Kaggle API download (run once)
    ├── preprocessing.py     # load → resize → CLAHE → normalize
    ├── features.py          # Gabor + HOG + LBP feature extraction
    ├── dataset.py           # subject split + verification trial generator
    ├── metrics.py           # EER, FAR/FRR, AUC, ROC, full report
    ├── train_fingerprint.py # train & tune SVM + Random Forest
    ├── verify_fingerprint.py# FingerprintVerifier class (fusion-ready)
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_dataset.py
    └── test_verify.py
```

---

## Quick start

### 1. Install dependencies

```bash
pip install kaggle opencv-python scikit-image scikit-learn numpy joblib tqdm
```

### 2. Set Kaggle credentials

**Option A – Environment variables (recommended for CI):**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

**Option B – JSON file:**
1. Go to kaggle.com → Account → API → **Create New API Token**
2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`

### 3. Download the SOCOFing dataset via API

```bash
cd fingerprint/src
python download_dataset.py
```
This calls `kaggle.api.dataset_download_files("ruizgara/socofing", ...)` and unzips
directly into `data/socofing/` — **no browser or manual download required**.

### 4. Train the models

```bash
cd fingerprint/src
python train_fingerprint.py
```

Output (saved to `fingerprint/models/`):
- `fingerprint_svm.joblib`
- `fingerprint_rf.joblib`

Each bundle contains `{model, threshold, classes, feature_dim}`.

### 5. Verify a fingerprint

```bash
# Command line
python verify_fingerprint.py path/to/finger.BMP 42

# Python API
from verify_fingerprint import FingerprintVerifier
v = FingerprintVerifier("svm")
result = v.verify_file("path/to/finger.BMP", claimed_id=42)
# → {"claimed": 42, "probability": 0.87, "accepted": True, "threshold": 0.65}
```

### 6. Run all tests (no real images needed)

```bash
cd fingerprint/src
python test_preprocessing.py
python test_features.py
python test_dataset.py
python test_verify.py
```

---

## Feature vector layout

| Block | Method | Dim |
|-------|--------|-----|
| Gabor | 3 wavelengths × 4 orientations, mean+std per filter | 24 |
| HOG | 9 orientations, 8×8 px/cell, 2×2 cells/block, L2-Hys | 4356 |
| LBP | Uniform, radius=3, 24 points, 26-bin histogram | 26 |
| **Total** | | **4406** |

---

## Preprocessing pipeline

```
Raw BMP  →  uint8 grayscale
         →  resize to 96×96 (bicubic)
         →  CLAHE (clipLimit=2.0, tileGrid=8×8)   ← enhances ridge contrast
         →  normalize to [0, 1] float32
```

---

## Fusion interface

`FingerprintVerifier` exposes:

```python
# 1-D float32 feature vector (plug into fusion pipeline)
vec = verifier.feature_vector("finger.BMP")          # shape (4406,)

# Genuine probability (plug into score-level fusion)
p = verifier.genuine_probability("finger.BMP", claimed_id=42)  # float ∈ [0,1]
```

