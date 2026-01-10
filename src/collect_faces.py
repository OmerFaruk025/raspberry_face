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

MAX_COUNT = 50
FACE_SIZE = 200

MIN_FACE_SIZE = 90          # çok küçük yüzleri alma
ASPECT_MIN = 0.75           # en/boy oranı
ASPECT_MAX = 1.3
CENTER_MARGIN = 0.35        # frame kenarına çok yakın yüzleri alma
BBOX_JUMP_LIMIT = 40        # bbox çok zıplıyorsa alma
MOVE_THRESHOLD = 12         # aynı poz spamını engelle

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
    time.sleep(1)

    try:
        while count < MAX_COUNT:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            h_frame, w_frame = frame.shape[:2]
            result = detector.detect_and_crop(frame)
            if result is None:
                continue  # yüz yok, bu frame’i geç
            face_img, bbox = result

            if bbox is None:
                time.sleep(0.03)
                continue

            x, y, w, h = bbox

            # --- BOYUT ---
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            # --- ORAN ---
            aspect = w / h
            if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
                continue

            # --- MERKEZ ---
            cx = x + w / 2
            cy = y + h / 2
            if (
                cx < w_frame * CENTER_MARGIN or
                cx > w_frame * (1 - CENTER_MARGIN) or
                cy < h_frame * CENTER_MARGIN or
                cy > h_frame * (1 - CENTER_MARGIN)
            ):
                continue

            # --- BBOX ZIPLAMA ---
            if last_bbox:
                if bbox_distance(bbox, last_bbox) > BBOX_JUMP_LIMIT:
                    last_bbox = bbox
                    continue

            # --- AYNI POZ ---
            if last_bbox:
                if bbox_distance(bbox, last_bbox) < MOVE_THRESHOLD:
                    continue

            last_bbox = bbox

            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (FACE_SIZE, FACE_SIZE))

            count += 1
            img_path = user_dir / f"{user_name}_{count}.jpg"
            cv2.imwrite(str(img_path), gray_face)

            print(f"{count}/{MAX_COUNT} kaydedildi")

            time.sleep(0.3)

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
                collect_data(name, mode="ekle")

        elif secim == "2" and users:
            idx = input("No: ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(users):
                collect_data(users[int(idx) - 1], mode="guncelle")

        elif secim == "3":
            break

# -----------------------------
if __name__ == "__main__":
    main_menu()
