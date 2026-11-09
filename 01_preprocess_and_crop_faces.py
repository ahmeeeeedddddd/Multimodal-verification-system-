import os
import shutil
import glob
import cv2
import numpy as np

BASE_DIR = "data/team_faces"
RAW_DIR = "data/team_faces_raw"
os.makedirs(RAW_DIR, exist_ok=True)

face_cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for person in ["ahmed", "mohamed"]:
    person_dir = os.path.join(BASE_DIR, person)
    raw_person_dir = os.path.join(RAW_DIR, person)
    os.makedirs(raw_person_dir, exist_ok=True)

    if not os.path.isdir(person_dir):
        continue

    files = sorted(
        glob.glob(os.path.join(person_dir, "*.jpg")) +
        glob.glob(os.path.join(person_dir, "*.jpeg")) +
        glob.glob(os.path.join(person_dir, "*.png"))
    )

    print(f"\n[01-Crop] Processing {person} ({len(files)} images)...")
    for f in files:
        backup_path = os.path.join(raw_person_dir, os.path.basename(f))
        if not os.path.exists(backup_path):
            shutil.copy2(f, backup_path)

        img_bgr = cv2.imread(backup_path)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = clahe.apply(gray)

        best_box = None
        for casc in [face_cascade_alt2, face_cascade_default, face_cascade_profile]:
            for g in [gray, gray_eq]:
                for sf, mn in [(1.1, 3), (1.05, 2)]:
                    dets = casc.detectMultiScale(g, scaleFactor=sf, minNeighbors=mn, minSize=(60, 60))
                    valid = [d for d in dets if d[1] < h * 0.70 and d[2] >= 60]
                    if len(valid) > 0:
                        best_box = max(valid, key=lambda b: b[2] * b[3])
                        break
                if best_box is not None:
                    break
            if best_box is not None:
                break

        if best_box is not None:
            x, y, bw, bh = best_box
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            size = max(bw, bh) * 1.15
            x1 = max(0, int(cx - size / 2.0))
            y1 = max(0, int(cy - size / 2.0))
            x2 = min(w, int(cx + size / 2.0))
            y2 = min(h, int(cy + size / 2.0))
            crop = img_bgr[y1:y2, x1:x2]
            cv2.imwrite(f, crop)
            print(f"  [Haar] {os.path.basename(f)}: {w}x{h} -> face crop ({crop.shape[1]}x{crop.shape[0]})")
        else:
            print(f"  [Haar] {os.path.basename(f)}: kept existing crop.")

print("\n[01-Crop] Completed successfully using classical Haar Cascades.")
