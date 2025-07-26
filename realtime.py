# capture_and_recognize.py
import cv2
import requests

def capture_and_recognize():
    cap = cv2.VideoCapture(0)
    print("📸 Capture en cours...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Erreur lors de la capture.")
        return

    temp_path = "temp_face.jpg"
    cv2.imwrite(temp_path, frame)

    with open(temp_path, "rb") as img_file:
        response = requests.post(
            "http://127.0.0.1:8000/realtime_recognize",
            files={"file": img_file}
        )

    print("🔍 Résultat:", response.json())

