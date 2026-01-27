import cv2
import numpy as np
from pathlib import Path
from face_detect import FaceDetector

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DATA_PATH  = ROOT_DIR / "data" / "faces"
MODEL_PATH = ROOT_DIR / "lbph_model.yml"
LABEL_PATH = ROOT_DIR / "labels.txt"

MAX_IMAGES_PER_PERSON = 30
IMG_SIZE = (200, 200)

# 🔥 DAHA STABİL
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

detector = FaceDetector()

faces = []
labels = []
label_map = {}
current_id = 0

print("🧠 Train başladı (FACE-DETECT uyumlu)")

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

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        face_img, _ = detector.detect_and_crop(frame)
        if face_img is None:
            continue

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.equalizeHist(gray)

        faces.append(gray)
        labels.append(current_id)
        count += 1

    current_id += 1

if not faces:
    print("❌ Eğitilecek veri yok")
    exit()

recognizer.train(faces, np.array(labels))
recognizer.write(str(MODEL_PATH))

with open(LABEL_PATH, "w", encoding="utf-8") as f:
    for idx, name in label_map.items():
        f.write(f"{idx}:{name}\n")

print(
    f"✅ Train tamamlandı | "
    f"Kişi: {len(label_map)} | "
    f"Toplam yüz: {len(faces)}"
)
