import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DATA_PATH = ROOT_DIR / "data" / "faces"
MODEL_PATH = ROOT_DIR / "lbph_model.yml"
LABEL_PATH = ROOT_DIR / "labels.txt"

MAX_IMAGES_PER_PERSON = 30
IMG_SIZE = (200, 200)

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

def is_face_quality_ok(img):
    h, w = img.shape
    if h < 120 or w < 120:
        return False
    return True

faces = []
labels = []
label_map = {}
current_id = 0

print("🧠 Train başladı (optimize mod)")

for person_dir in DATA_PATH.iterdir():
    if not person_dir.is_dir():
        continue

    name = person_dir.name
    label_map[current_id] = name
    print(f"📂 {name} işleniyor")

    count = 0

    for img_path in sorted(person_dir.glob("*.jpg")):
        if count >= MAX_IMAGES_PER_PERSON:
            break

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if not is_face_quality_ok(img):
            continue

        img = cv2.resize(img, IMG_SIZE)

        faces.append(img)
        labels.append(current_id)
        count += 1

    current_id += 1

if len(faces) == 0:
    print("❌ Eğitilecek veri yok")
    exit()

recognizer.train(faces, np.array(labels))
recognizer.write(str(MODEL_PATH))

with open(LABEL_PATH, "w", encoding="utf-8") as f:
    for idx, name in label_map.items():
        f.write(f"{idx}:{name}\n")

print(f"✅ Train tamamlandı | Kişi: {len(label_map)} | Toplam yüz: {len(faces)}")
