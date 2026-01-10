import cv2
import numpy as np
from pathlib import Path

# -----------------------------
# YOLLAR
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "faces"
MODEL_PATH = BASE_DIR / "lbph_model.yml"
LABEL_PATH = BASE_DIR / "labels.txt"

# -----------------------------
# LBPH OPTİMİZE
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=16,
    grid_x=8,
    grid_y=8
)

# -----------------------------
# CLAHE
# -----------------------------
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# -----------------------------
# DATA OKUMA
# -----------------------------
def load_faces(path):
    faces = []
    labels = []
    label_map = {}
    current_id = 0

    if not path.exists():
        print("❌ data/faces yok")
        return [], [], {}

    for person_dir in path.iterdir():
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        label_map[name] = current_id
        print(f"📂 {name} işleniyor...")
        
        for img_path in person_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape
            if w < 120 or h < 120:
                continue  # küçük / hatalı crop

            img = cv2.resize(img, (200, 200))
            img = clahe.apply(img)

            faces.append(img)
            labels.append(current_id)

        current_id += 1

    return faces, labels, label_map

# -----------------------------
# TRAIN
# -----------------------------
print("🧠 Eğitim başlıyor...")
faces, labels, label_map = load_faces(DATA_PATH)

if not faces:
    print("❌ Eğitim verisi yok")
    exit()

recognizer.train(faces, np.array(labels))
recognizer.save(str(MODEL_PATH))

with open(LABEL_PATH, "w", encoding="utf-8") as f:
    for name, idx in label_map.items():
        f.write(f"{idx}:{name}\n")

print("✅ Eğitim tamamlandı")
print(f"👤 Kişi sayısı: {len(label_map)}")
print(f"📦 Model: {MODEL_PATH}")
