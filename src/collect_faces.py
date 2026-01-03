import cv2
import os
import time
from face_detect import FaceDetector

BASE_DIR = "data/faces"
MAX_SAMPLES = 25
SAVE_DELAY = 0.5  # saniye

def normalize_name(name: str) -> str:
    tr_map = {
        "ç":"c", "Ç":"c",
        "ğ":"g", "Ğ":"g",
        "ı":"i", "İ":"i",
        "ö":"o", "Ö":"o",
        "ş":"s", "Ş":"s",
        "ü":"u", "Ü":"u"
    }
    for k, v in tr_map.items():
        name = name.replace(k, v)
    return name.lower().strip().replace(" ", "_")

def list_people():
    if not os.path.exists(BASE_DIR):
        return []
    return [
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]

os.makedirs(BASE_DIR, exist_ok=True)

detector = FaceDetector()
cap = cv2.VideoCapture(0)

print("Yüz algılanması bekleniyor...")

# 1️⃣ YÜZ GÖRÜLENE KADAR BEKLE
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera okunamadı")
        exit()

    face_img, _ = detector.detect_and_crop(frame)

    if face_img is not None:
        break

    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# 2️⃣ MENÜ
people = list_people()

options = ["Yeni kişi ekle"]
if people:
    options.append("Kişiyi güncelle")
options.append("Çıkış")

print("\nSeçim yap:")
for i, opt in enumerate(options, 1):
    print(f"{i} - {opt}")

try:
    choice = int(input(">>> "))
    action = options[choice - 1]
except:
    print("❌ Geçersiz seçim")
    exit()

if action == "Çıkış":
    exit()

# 3️⃣ KİŞİ SEÇİMİ
if action == "Yeni kişi ekle":
    raw_name = input("Kişi adı: ")
    name = normalize_name(raw_name)
else:
    print("\nGüncellenecek kişi:")
    for i, p in enumerate(people, 1):
        print(f"{i} - {p}")

    try:
        idx = int(input("Numara seç: ")) - 1
        name = people[idx]
    except:
        print("❌ Geçersiz seçim")
        exit()

person_dir = os.path.join(BASE_DIR, name)
os.makedirs(person_dir, exist_ok=True)

# 🔥 GÜNCELLEMEDE ESKİ FOTOĞRAFLARI TAMAMEN SİL
if action == "Kişiyi güncelle":
    for f in os.listdir(person_dir):
        file_path = os.path.join(person_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

print("\nYüzler otomatik kaydediliyor")
print("Kameraya bak, kafanı hafif oynat")

count = 0
saved = 0
last_save_time = 0

# 4️⃣ OTOMATİK – GRAYSCALE – DELAY'Lİ KAYIT
while saved < MAX_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    face_img, _ = detector.detect_and_crop(frame)

    if face_img is not None:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Yuz Kaydi", gray_face)

        now = time.time()
        if now - last_save_time >= SAVE_DELAY:
            img_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(img_path, gray_face)

            count += 1
            saved += 1
            last_save_time = now

            print(f"{saved}/{MAX_SAMPLES} kaydedildi")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("Kayıt tamamlandı. Menüye dönülüyor.")

cap.release()
cv2.destroyAllWindows()
