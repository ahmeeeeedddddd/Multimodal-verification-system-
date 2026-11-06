# Hierarchical Multimodal Biometric Verification System

A multimodal biometric verification system built with traditional Machine Learning techniques (SVM, Random Forest) using face images, voice recordings, and fingerprints.

## Project Structure
- `face_features.py`: Face preprocessing (Haar cascade face crop, grayscale, resize $64\times 64$, normalization) and feature extraction (HOG, LBP, PCA/eigenfaces).
- `voice_features.py`: Audio feature extraction (MFCC, Zero-Crossing Rate, RMS Energy) and multi-speaker chimeric dataset handling.
- `fingerprint_features.py`: Fingerprint preprocessing (grayscale, resize $96\times 96$, CLAHE ridge enhancement, normalization) and feature extraction (Gabor bank, HOG, LBP).
- `metrics.py`: Biometric evaluation metrics (FAR, FRR, EER, Accuracy, AUC, Confusion Matrix).
- `task1_face_verification.py`: Task 1 face-only verification pipeline (SVM & Random Forest) with saved model export.
- `task2_fingerprint_verification.py`: Task 2 fingerprint verification pipeline (SOCOFing, Gabor + HOG + LBP, SVM & RF).
- `realtime_face_verification.py`: Bonus 2 live webcam face verification interface.
- `task5_multimodal_verification.py`: Task 5 early-fusion multimodal verification with three-way performance comparison.
- `fingerprint/`: Standalone modular fingerprint package with Kaggle API automatic fetcher, verification API, and unit tests.

## Getting Started

### Requirements
```bash
pip install numpy scikit-learn scikit-image opencv-python librosa joblib matplotlib kaggle
```

### Running the Modules
1. **Task 1: Face Verification**
   ```bash
   python task1_face_verification.py
   ```
2. **Task 2: Fingerprint Verification**
   ```bash
   python task2_fingerprint_verification.py
   ```
3. **Bonus 2: Real-Time Webcam Face Verification**
   ```bash
   python realtime_face_verification.py
   ```
4. **Task 5: Multimodal Early Fusion & 3-Way Comparison**
   ```bash
   python task5_multimodal_verification.py
   ```
