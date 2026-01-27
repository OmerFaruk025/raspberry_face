from flask import Flask, render_template, send_from_directory, redirect, url_for, jsonify
import subprocess, os, sys, csv, logging

# -------------------------
# Temel ayarlar
# -------------------------
BASE_DIR = os.path.dirname(__file__)                  # src klasörü
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")  # src/templates
CSV_LOG_PATH = os.path.join(BASE_DIR, "..", "hakan_fidan.csv")  # kökteki CSV

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Gereksiz GET/POST loglarını kapat
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

recognize_process = None

# -------------------------
# Yüz tanıma işlemi
# -------------------------
def start_recognize():
    global recognize_process
    if recognize_process is None or recognize_process.poll() is not None:
        recognize_process = subprocess.Popen(
            [sys.executable, os.path.join(BASE_DIR, "recognize.py")]
        )

def stop_recognize():
    global recognize_process
    if recognize_process and recognize_process.poll() is None:
        recognize_process.terminate()
        try:
            recognize_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            recognize_process.kill()
        recognize_process = None

# -------------------------
# CSS / JS dosyalarını servis et
# -------------------------
@app.route('/assets/<path:filename>')
def static_file(filename):
    return send_from_directory(TEMPLATES_DIR, filename)

# -------------------------
# Logları JSON olarak döndür
# -------------------------
@app.route("/logs")
def get_logs():
    logs = []
    if os.path.exists(CSV_LOG_PATH):
        try:
            with open(CSV_LOG_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = [row for row in reader if row and len(row) >= 2]
                for row in rows[-20:]:
                    logs.append({"time": row[0], "name": row[1]})
        except Exception as e:
            logs.append({"time": "", "name": f"Hata: {e}"})
    return jsonify(logs)

# -------------------------
# Ana sayfa
# -------------------------
@app.route("/")
def index():
    status = "Çalışıyor" if recognize_process and recognize_process.poll() is None else "Durduruldu"
    return render_template("index.html", status=status)

# -------------------------
# Start / Stop
# -------------------------
@app.route("/start")
def start():
    start_recognize()
    return redirect(url_for("index"))

@app.route("/stop")
def stop():
    stop_recognize()
    return redirect(url_for("index"))

# -------------------------
# Uygulama başlat
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
