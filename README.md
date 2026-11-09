# Hierarchical Multimodal Biometric Verification & Access Control System

An end-to-end multimodal biometric verification system employing traditional Machine Learning (SVM, Random Forest) with Face, Voice, and Fingerprint modalities, coupled with a hierarchical decision engine and isolated speech command recognition for Role-Based Access Control (RBAC).

---

## Numbered Execution Pipeline

To reproduce, train, benchmark, or run the verification system, execute the numbered scripts in sequential order:

| Step | Script | Description |
|:---:|:---|:---|
| **01** | `01_preprocess_and_crop_faces.py` | Detects and crops raw face photos into centered, square crops using deep MTCNN face detection. |
| **02** | `02_train_face_verification.py` | Extracts HOG, LBP, and PCA/Eigenface features ($D=1618$) from Olivetti and enrolled team faces, trains RBF SVM verifier, and exports templates. |
| **03** | `03_train_fingerprint_verification.py` | Enhances ridge flow with CLAHE and extracts Gabor, HOG, and LBP features ($D=4406$) to train fingerprint verifier. |
| **04** | `04_train_multimodal_fusion.py` | Generates early feature fusion pairs (Face + Voice MFCC) for enrolled subjects, fits SVM fusion verifier, and saves templates. |
| **05** | `05_tune_and_benchmark_pipeline.py` | Tunes Stage 1 thresholds ($\theta_{low}$, $\theta_{high}$) on validation data, benchmarks 5 biometric architectures, and plots ROC & DET curves. |
| **06** | `06_train_speech_commands.py` | Trains dual-engine speech command recognizer (Google Speech Recognition + MFCC/SVM fallback) across 8 system commands. |
| **07** | `07_realtime_webcam_stream.py` | Runs standalone OpenCV desktop camera stream with real-time face detection, 5-second stable hold, and live Stage 1 score display. |
| **08** | `08_launch_streamlit_app.py` | Launches the interactive multi-page Streamlit web dashboard (`streamlit run app.py`). |

---

## Supporting Library Modules

- [`face_features.py`](file:///c:/Users/Administrator/Desktop/Task%202%20supervised%20learning/face_features.py): Preprocessing (grayscale, CLAHE, resizing to $64\times 64$, normalization) and feature extraction (HOG, Uniform LBP, PCA Eigenfaces).
- [`voice_features.py`](file:///c:/Users/Administrator/Desktop/Task%202%20supervised%20learning/voice_features.py): Preprocessing (pre-emphasis, trimming) and feature extraction (13 MFCCs + deltas, Zero-Crossing Rate, RMS energy).
- [`fingerprint_features.py`](file:///c:/Users/Administrator/Desktop/Task%202%20supervised%20learning/fingerprint_features.py): Ridge enhancement with CLAHE and feature extraction (Gabor filter bank at 8 orientations, HOG, LBP).
- [`hierarchical_pipeline.py`](file:///c:/Users/Administrator/Desktop/Task%202%20supervised%20learning/hierarchical_pipeline.py): Core hierarchical verification decision engine:
  - **Stage 1 (Face)**: Fast path. Score $\ge \theta_{high}$ $\rightarrow$ Accept. Score $< \theta_{low}$ $\rightarrow$ Reject.
  - **Stage 2 (Face + Voice Fusion)**: Triggered when $\theta_{low} \le \text{Score} < \theta_{high}$. Score $\ge \theta_{fusion}$ $\rightarrow$ Accept.
  - **Stage 3 (Fingerprint)**: Optional fallback or high-security escalation.
- [`speech_commands.py`](file:///c:/Users/Administrator/Desktop/Task%202%20supervised%20learning/speech_commands.py): Speech command recognition with Role-Based Access Control (Admin, Operator, User, Guest).
- [`metrics.py`](file:///c:/Users/Administrator/Desktop/Task%202%20supervised%20learning/metrics.py): Shared evaluation metrics (FAR, FRR, EER, ROC AUC, DET curves).
- [`app.py`](file:///c:/Users/Administrator/Desktop/Task%202%20supervised%20learning/app.py): The full interactive Streamlit web dashboard.

---

## Installation & Setup

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline from start to finish:
   ```bash
   python 01_preprocess_and_crop_faces.py
   python 02_train_face_verification.py
   python 03_train_fingerprint_verification.py
   python 04_train_multimodal_fusion.py
   python 05_tune_and_benchmark_pipeline.py
   python 06_train_speech_commands.py
   ```

3. Launch the web dashboard:
   ```bash
   python 08_launch_streamlit_app.py
   ```
   *(or `streamlit run app.py`)*

---

## Diagnostic Findings: Face Cropping & Webcam Verification

### Why Old Uploaded Photos Failed
1. **Haar Cascade Failure on High-Resolution / Angled Photos**:
   - The standard OpenCV Haar cascade failed on portrait phone photos ($959\times 1280$) when taken with eyeglasses, tilted heads, or natural indoor lighting. It frequently detected shirt collars, towels, or chest areas as faces (`y=719`), saving non-face crops into the template.
   - On photos where it did find the face, the bounding box was overly broad ($700\times 700$), including background and neck.
2. **Landmark Alignment Mismatch with Live Webcam**:
   - The live webcam captured a tight, upright face crop ($140\times 140$ to $240\times 240$).
   - When resized to $64\times 64$, eyes, nose, and mouth fell into completely different spatial cells compared to the broad/corrupted crops in the template. This caused the difference vector $|x - \text{template}|$ to look like an impostor, resulting in low scores (~$20\%$).

### How It Was Resolved
- **MTCNN Face Cropping (`01_preprocess_and_crop_faces.py`)**: All 13 photos for Ahmed and 7 for Mohamed were re-cropped using MTCNN deep face detection with square margins and landmark centering.
- **Laptop Camera Photos**: Adding direct webcam captures (`10.jpg`, `11.jpg`, `12.jpg`) eliminated the camera sensor domain gap.
- **RBF SVM + Dual Cascade Detection**: The face verifier achieves 98.0% accuracy, 3.15% EER, and scores 72%–100% on live webcam frames.
