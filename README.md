# Hierarchical Multimodal Biometric Verification System

A multimodal biometric verification system built with traditional Machine Learning techniques (SVM, Random Forest) using face images and voice recordings.

## Project Structure
- `face_features.py`: Face preprocessing (Haar cascade face crop, grayscale, resize $64\times 64$, normalization) and feature extraction (HOG, LBP, PCA/eigenfaces).
- `voice_features.py`: Audio feature extraction (MFCC, Zero-Crossing Rate, RMS Energy) and multi-speaker chimeric dataset handling.
- `metrics.py`: Biometric evaluation metrics (FAR, FRR, EER, Accuracy, AUC, Confusion Matrix).
- `task1_face_verification.py`: Task 1 face-only verification pipeline (SVM & Random Forest) with saved model export.
- `realtime_face_verification.py`: Bonus 2 live webcam face verification interface.
- `task5_multimodal_verification.py`: Task 5 early-fusion multimodal verification with three-way performance comparison.

## Getting Started

### Requirements
```bash
pip install numpy scikit-learn scikit-image opencv-python librosa joblib matplotlib
```

### Running the Modules
1. **Task 1: Face Verification**
   ```bash
   python task1_face_verification.py
   ```
2. **Bonus 2: Real-Time Webcam Face Verification**
   ```bash
   python realtime_face_verification.py
   ```
3. **Task 5: Multimodal Early Fusion & 3-Way Comparison**
   ```bash
   python task5_multimodal_verification.py
   ```
