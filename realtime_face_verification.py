import cv2
import joblib
import os
from task1_face_verification import verify_claimed_identity

def run_realtime():
    model_path = os.path.join("saved_models", "face_verifier.pkl")
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please run task1_face_verification.py first.")
        return

    bundle = joblib.load(model_path)
    available_ids = list(bundle["templates"].keys())
    print(f"[Real-Time Face Verification] Enrolled identities: {available_ids}")

    claimed_id = available_ids[0]
    print(f"Default claimed identity set to: {claimed_id}")
    print("Controls: Press 'q' to exit | Press 'c' to enter a new claimed ID.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        accepted, prob = verify_claimed_identity(bundle, frame, claimed_id)

        color = (0, 255, 0) if accepted else (0, 0, 255)
        status_text = f"Claimed ID: {claimed_id} | Status: {'ACCEPTED' if accepted else 'REJECTED'}"
        prob_text = f"Genuine Prob: {prob * 100:.1f}% (Threshold: {bundle['threshold'] * 100:.1f}%)"

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, prob_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Real-Time Face Verification (Bonus 2)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            try:
                new_id = int(input(f"Enter new claimed identity {available_ids}: "))
                if new_id in available_ids:
                    claimed_id = new_id
                else:
                    print(f"Invalid ID. Available: {available_ids}")
            except ValueError:
                print("Invalid integer.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()
