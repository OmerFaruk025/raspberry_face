import cv2
import os
import time
from camera import Camera  # Kanka senin sınıfa geçtik
from face_detect import FaceDetector

BASE_DIR = "data/faces"
MAX_SAMPLES = 25
SAVE_DELAY = 0.5 

# Laptop IP'ni buraya da giriyoruz (main.py ile aynı olmalı)
LAPTOP_IP = "192.168.1.47" 
stream_url = f"http://{LAPTOP_IP}:5000/video"

def normalize_name(name: str) -> str:
    tr_map = {"ç":"c", "Ç":"c", "ğ":"g", "Ğ":"g", "ı":"i", "İ":"i", "ö":"o", "Ö":"o", "ş":"s", "Ş":"s", "ü":"u", "Ü":"u"}
    for k, v in tr_map.items():
        name = name.replace(k, v)
    return name.lower().strip().replace(" ", "_")

def list_people():
    if not os.path.exists(BASE_DIR):
        return []
    return [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

os.makedirs(BASE_DIR, exist_ok=True)

detector = FaceDetector()
# Kanka burada artık Camera sınıfını kullanıyoruz ki IP stream çalışsın
cam = Camera(source=stream_url)

print("Yüz algılanması bekleniyor...")

# 1️⃣ YÜZ GÖRÜLENE KADAR BEKLE
while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    # Senin face_detect içindeki yeni fonksiyonu çağırdık
    face_img, _ = detector.detect_and_crop(frame)

    if face_img is not None:
        break

    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cam.release()
        cv2.destroyAllWindows()
        exit()

# 2️⃣ MENÜ (Senin orijinal mantığın)
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

if action == "Kişiyi güncelle":
    for f in os.listdir(person_dir):
        file_path = os.path.join(person_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

print("\nYüzler otomatik kaydediliyor...")

count = 0
saved = 0
last_save_time = 0

# 4️⃣ OTOMATİK KAYIT
while saved < MAX_SAMPLES:
    ret, frame = cam.read()
    if not ret or frame is None:
        break

    face_img, _ = detector.detect_and_crop(frame)

    if face_img is not None:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Eğitimle uyumlu olması için resize ekledik (200x200)
        gray_face = cv2.resize(gray_face, (200, 200)) 
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

print("Kayıt tamamlandı.")
cam.release()
cv2.destroyAllWindows()