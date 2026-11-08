import cv2
import joblib
import os
import numpy as np
from face_features import preprocess_face, face_cascade


def run_realtime():
    # Prefer team model with ahmed and mohamed enrolled
    if os.path.exists("saved_models/face_verifier_team.pkl"):
        model_path = "saved_models/face_verifier_team.pkl"
    else:
        model_path = os.path.join("saved_models", "face_verifier.pkl")

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train a model first.")
        return

    print(f"[Real-Time Face Verification] Loading model from: {model_path}")
    bundle = joblib.load(model_path)
    templates = bundle["templates"]
    available_ids = list(templates.keys())
    name_map = bundle.get("label_to_name", {1000: "Ahmed", 1001: "Mohamed"})

    print(f"[Real-Time Face Verification] Enrolled identities: {[f'{i} ({name_map.get(i, i)})' for i in available_ids]}")

    # Default to Ahmed (1000) if available, otherwise first enrolled ID
    claimed_id = 1000 if 1000 in available_ids else available_ids[0]
    print(f"Default claimed identity set to: {claimed_id} ({name_map.get(claimed_id, claimed_id)})")
    print("Controls: Press 'q' to exit | Press 'c' to change claimed identity.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    extractor = bundle["extractor"]
    model = bundle["model"]
    threshold = bundle.get("threshold", 0.50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face with frontal cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

        accepted = False
        prob = 0.0

        if len(faces) > 0:
            # Pick largest face
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            face_crop = frame[y:y+fh, x:x+fw]

            u8, norm = preprocess_face(face_crop)
            feat = extractor.extract(u8, norm)

            if claimed_id in templates:
                diff = np.abs(feat - templates[claimed_id]).reshape(1, -1)
                prob = float(model.predict_proba(diff)[0, 1])
                accepted = prob >= threshold

            box_color = (0, 255, 0) if accepted else (0, 0, 255)
            label = f"{name_map.get(claimed_id, claimed_id)}: {prob * 100:.1f}%"
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), box_color, 2)
            cv2.putText(frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        else:
            box_color = (0, 165, 255)

        status_color = (0, 255, 0) if accepted else (0, 0, 255)
        claimed_name = name_map.get(claimed_id, str(claimed_id))
        status_text = f"Claim: {claimed_name} | {'ACCEPTED' if accepted else 'REJECTED'}"
        prob_text = f"Prob: {prob * 100:.1f}% (Threshold: {threshold * 100:.1f}%)"

        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, prob_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Real-Time Face Verification (Bonus 2)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            try:
                print(f"Available IDs: {[f'{i}: {name_map.get(i, i)}' for i in available_ids]}")
                val = input("Enter new claimed ID: ").strip()
                new_id = int(val)
                if new_id in available_ids:
                    claimed_id = new_id
                    print(f"Claimed identity changed to: {claimed_id} ({name_map.get(claimed_id, claimed_id)})")
                else:
                    print(f"ID {new_id} not in available IDs.")
            except Exception as e:
                print(f"Error changing ID: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()
