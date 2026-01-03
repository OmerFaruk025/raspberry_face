import cv2
import os
import numpy as np

DATA_DIR = "data/faces"

faces = []
face_labels = []

label_map = {}
current_label = 0

# Kişileri sabit sırayla al
for person in sorted(os.listdir(DATA_DIR)):
    person_path = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # bozuk dosya varsa geç

        faces.append(img)
        face_labels.append(current_label)

    current_label += 1

if len(faces) == 0:
    raise RuntimeError("Hiç yüz verisi bulunamadı")

# LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

recognizer.train(faces, np.array(face_labels))
recognizer.save("lbph_model.yml")

# 🔥 LABEL DOSYASINI YAZ
with open("labels.txt", "w", encoding="utf-8") as f:
    for label, name in label_map.items():
        f.write(f"{label}:{name}\n")

print("MODEL EĞİTİLDİ")
print(f"Toplam kişi: {len(label_map)}")
print(f"Toplam foto: {len(faces)}")
