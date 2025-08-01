import cv2
import requests

API_URL = "http://127.0.0.1:8000/realtime_recognize"

def recognize_face(face_img):
    _, img_encoded = cv2.imencode('.jpg', face_img)
    try:
        response = requests.post(
            API_URL,
            files={"file": ("face.jpg", img_encoded.tobytes(), "image/jpeg")}
        )
        return response.json()
    except:
        return {"identity": "Erreur", "similarity": 0.0}

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Erreur caméra.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("🎥 Démarrage - Appuie sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #print("Nombre de visages détectés :", len(faces))
        

        for (x, y, w, h) in faces:
            face = frame_resized[y:y+h, x:x+w]
            #print('Appel de Visage')
            result = recognize_face(face)
            name = result.get("identity")
            distance = result.get("distance", 999)

            if name is None or distance > 1.1:
                label = "Inconnu"
            else:
                label = f"{name} ({distance:.2f})"

            print(result)

            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Reconnaissance", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
