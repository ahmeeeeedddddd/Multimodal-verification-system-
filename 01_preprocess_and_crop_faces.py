"""01 — Preprocess and Crop Team Faces.

Uses MTCNN deep cascaded networks to detect and crop face squares
from raw photos in data/team_faces/<person>/*.jpg.
Backs up raw photos to data/team_faces_raw/ to preserve originals.
"""

import os
import shutil
import glob
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=False, select_largest=True)

BASE_DIR = "data/team_faces"
RAW_DIR = "data/team_faces_raw"

os.makedirs(RAW_DIR, exist_ok=True)

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
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(Image.fromarray(img_rgb))

        if boxes is not None and len(boxes) > 0 and probs[0] > 0.8:
            box = boxes[0]
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            # Use 1.2x face size to capture full face from forehead to chin
            size = max(bw, bh) * 1.2
            x1 = max(0, int(cx - size / 2))
            y1 = max(0, int(cy - size / 2))
            x2 = min(w, int(cx + size / 2))
            y2 = min(h, int(cy + size / 2))

            crop = img_bgr[y1:y2, x1:x2]
            cv2.imwrite(f, crop)
            print(f"  [MTCNN] {os.path.basename(f)}: {w}x{h} -> clean face crop ({crop.shape[1]}x{crop.shape[0]})")
        else:
            print(f"  [Warning] No face detected by MTCNN in {f}, keeping original.")

print("\n[01-Crop] Completed successfully.")
