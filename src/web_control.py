from flask import Flask, render_template, redirect, url_for, jsonify, Response, send_from_directory
import threading, queue, time, os
import logging

# -----------------------------
# Flask setup
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # request loglarını kapat

# -----------------------------
# PATHLER
BASE_DIR = os.path.dirname(__file__)
CSV_LOG_PATH = os.path.join(os.path.dirname(BASE_DIR), "hakan_fidan.csv")  # raspberry_face/hakan_fidan.csv

# -----------------------------
# GLOBALS
FRAME_QUEUE = queue.Queue(maxsize=1)
RUNNING = [False]  # mutable list
recognize_thread = None

# -----------------------------
# Recognize.py thread
def run_recognize():
    import recognize
    recognize.run(FRAME_QUEUE, RUNNING)  # sadece frame_queue ve running gönderiyoruz

# -----------------------------
# Static dosyalar (css/js)
@app.route('/assets/<path:filename>')
def static_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'templates'), filename)

# -----------------------------
# Logları JSON olarak döndür
@app.route("/logs")
def get_logs():
    logs = []
    if os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, "r", encoding="utf-8") as f:
            reader = f.read().splitlines()
            for row in reader[-5:]:
                parts = row.split(",")
                if len(parts) >= 2:
                    time_str = parts[0].split(" ")[1]
                    logs.append({"time": time_str, "name": parts[1]})
    return jsonify(logs)

# -----------------------------
# Ana sayfa
@app.route("/")
def index():
    status = "Çalışıyor" if RUNNING[0] else "Durduruldu"
    return render_template("index.html", status=status)

# -----------------------------
# Start / Stop
@app.route("/start")
def start():
    global recognize_thread
    if not RUNNING[0]:
        RUNNING[0] = True
        if recognize_thread is None or not recognize_thread.is_alive():
            recognize_thread = threading.Thread(target=run_recognize, daemon=True)
            recognize_thread.start()
    return redirect(url_for("index"))

@app.route("/stop")
def stop():
    RUNNING[0] = False
    return redirect(url_for("index"))

# -----------------------------
# Canlı video stream
def gen_frames():
    while True:
        if not FRAME_QUEUE.empty():
            frame = FRAME_QUEUE.get()
            import cv2
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------
# MAIN
if __name__ == "__main__":
    # debug=False + use_reloader=False → duplicate logları önler
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
