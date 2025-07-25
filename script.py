import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def register_face(face_cascade):
    name = input("Enter your name: ").strip()
    user_id = input("Enter your user ID: ").strip()

    save_path = f"data/{name}_{user_id}"
    ensure_dir(save_path)

    cap = cv2.VideoCapture(0)
    saved_count = 0
    last_save_time = time.time()

    print("Tip: Move your face left, right, up, down and change expressions while registering!")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))

            cv2.imshow("Face ROI", face_img)

            current_time = time.time()
            if current_time - last_save_time >= 0.5 and saved_count < 20:
                saved_count += 1
                cv2.imwrite(f"{save_path}/{saved_count}.jpg", face_img)
                print(f"Saved image {saved_count}/20")
                last_save_time = current_time

        cv2.imshow("Register Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or saved_count >= 20:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {saved_count} images for {name}")

def train_model(data_path='data'):
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for dir_name in os.listdir(data_path):
        dir_path = os.path.join(data_path, dir_name)
        if not os.path.isdir(dir_path):
            continue

        label_map[current_label] = dir_name
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(current_label)
        current_label += 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    return recognizer, label_map

def mark_attendance(name, user_id):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    attendance_file = "Attendance.csv"

    already_marked = False
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3 and row[0] == name and row[1] == user_id and row[2] == date_str:
                    already_marked = True
                    break

    if not already_marked:
        with open(attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, user_id, date_str, time_str])
        print(f"Attendance marked for {name}")
    else:
        print(f"Already marked today for {name}")

def take_attendance(face_cascade, eye_cascade):
    recognizer, label_map = train_model()
    cap = cv2.VideoCapture(0)

    marked_today = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

            if len(eyes) >= 1:
                face_img = cv2.resize(roi_gray, (200, 200))
                label, confidence = recognizer.predict(face_img)

                if confidence < 70:
                    name_id = label_map[label]
                    name, user_id = name_id.split("_")

                    if (name, user_id) not in marked_today:
                        mark_attendance(name, user_id)
                        marked_today.add((name, user_id))

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({user_id})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if marked_today:
            cv2.putText(frame, "Attendance marked!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    print("1. Register new user")
    print("2. Take attendance")
    choice = input("Enter choice (1/2): ").strip()

    if choice == '1':
        register_face(face_cascade)
    elif choice == '2':
        take_attendance(face_cascade, eye_cascade)
    else:
        print("Invalid choice")




