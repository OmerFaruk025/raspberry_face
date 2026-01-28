import cv2
import time
import shutil
import math
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# AYARLAR
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "faces"
DATA_PATH.mkdir(parents=True, exist_ok=True)

MAX_COUNT = 30
FACE_SIZE = 200

# Daha esnek filtreler
MIN_FACE_SIZE = 70        # çok küçük yüzleri alma
ASPECT_MIN = 0.65         # en/boy oranı
ASPECT_MAX = 1.4
CENTER_MARGIN = 0.2       # frame kenarına yakınlığı gevşettik
BBOX_JUMP_LIMIT = 100     # bbox çok zıplıyorsa alma
MOVE_THRESHOLD = 5        # çok az hareketi alma

# -----------------------------
def get_registered_users():
    return [d.name for d in DATA_PATH.iterdir() if d.is_dir()]

# -----------------------------
def bbox_distance(b1, b2):
    return math.hypot(b1[0] - b2[0], b1[1] - b2[1])

# -----------------------------
def collect_data(user_name, mode="ekle"):
    user_dir = DATA_PATH / user_name

    if mode == "guncelle" and user_dir.exists():
        print(f"🔄 '{user_name}' eski verileri siliniyor...")
        shutil.rmtree(user_dir)

    user_dir.mkdir(parents=True, exist_ok=True)

    cam = Camera()
    detector = FaceDetector()

    count = 0
    last_bbox = None

    print(f"📸 Kayıt başlıyor: {user_name}")
    time.sleep(0.5)

    try:
        while count < MAX_COUNT:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            h_frame, w_frame = frame.shape[:2]

            face_img, bbox = detector.detect_and_crop(frame)
            if bbox is None:
                continue

            x, y, w, h = bbox

            # -------------------------
            # TEMEL KALİTE KONTROLLERİ
            # -------------------------
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            aspect = w / h
            if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
                continue

            cx = x + w / 2
            cy = y + h / 2
            if (
                cx < w_frame * CENTER_MARGIN or
                cx > w_frame * (1 - CENTER_MARGIN) or
                cy < h_frame * CENTER_MARGIN or
                cy > h_frame * (1 - CENTER_MARGIN)
            ):
                continue

            if last_bbox:
                dist = bbox_distance(bbox, last_bbox)
                if dist > BBOX_JUMP_LIMIT or dist < MOVE_THRESHOLD:
                    continue

            last_bbox = bbox

            # -------------------------
            # 🎯 LBPH UYUMLU FINAL CROP
            # -------------------------
            pad_left  = int(w * 0.1)
            pad_right = int(w * 0.1)
            pad_up    = int(h * 0.15)
            pad_down  = int(h * 0.05)

            x1 = max(0, x - pad_left)
            y1 = max(0, y - pad_up)
            x2 = min(w_frame, x + w + pad_right)
            y2 = min(h_frame, y + h + pad_down)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            count += 1
            img_path = user_dir / f"{user_name}_{count}.jpg"
            cv2.imwrite(str(img_path), gray)

            print(f"{count}/{MAX_COUNT} kaydedildi")
            time.sleep(0.1)  # çok kısa bekleme

    finally:
        cam.release()
        print(f"✅ Kayıt tamamlandı → data/faces/{user_name}")

# -----------------------------
def main_menu():
    while True:
        users = get_registered_users()

        print("\n" + "=" * 35)
        print("🛡️ Pi-FaceID | VERİ TOPLAMA")
        print("=" * 35)

        if users:
            for i, u in enumerate(users, 1):
                print(f"{i}- {u}")
            print("1-Yeni Ekle | 2-Güncelle | 3-Çıkış")
        else:
            print("⚠️ Kayıt yok.")
            print("1-Yeni Ekle | 3-Çıkış")

        secim = input("Seçim: ").strip()

        if secim == "1":
            name = input("İsim: ").strip()
            if name:
                collect_data(name)

        elif secim == "2" and users:
            idx = input("No: ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(users):
                collect_data(users[int(idx) - 1], mode="guncelle")

        elif secim == "3":
            break

# -----------------------------
if __name__ == "__main__":
    main_menu()
