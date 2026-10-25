import sqlite3, time, threading
import cv2
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

VIDEO = r"C:\Users\ssrih\OneDrive\Desktop\People in group project\people.mp4.mp4"

db = sqlite3.connect("counts.db", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS counts (ts REAL, count INT)")
latest = {"count": 0}
frame_jpg = {"data": None}

def run():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(VIDEO)
    last_save = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # video mudinja loop
            continue

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (960, int(h * 960 / w)))

        r = model(frame, classes=[0], conf=0.3, imgsz=960, verbose=False)[0]
        count = len(r.boxes)
        latest["count"] = count

        annotated = r.plot()  # person mela box
        cv2.rectangle(annotated, (0, 0), (330, 55), (0, 0, 0), -1)
        cv2.putText(annotated, f"People Count: {count}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        _, jpg = cv2.imencode(".jpg", annotated)
        frame_jpg["data"] = jpg.tobytes()

        if time.time() - last_save >= 1:  # DB la 1 sec ku oru dhadavai save
            db.execute("INSERT INTO counts VALUES (?,?)", (time.time(), count))
            db.commit()
            last_save = time.time()

threading.Thread(target=run, daemon=True).start()
app = FastAPI()

def gen():
    while True:
        d = frame_jpg["data"]
        if d is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + d + b"\r\n"
        time.sleep(0.03)

@app.get("/video")
def video():
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/count")
def count():
    return latest

@app.get("/history")
def history():
    rows = db.execute("SELECT ts, count FROM counts ORDER BY ts DESC LIMIT 50").fetchall()
    return [{"ts": t, "count": c} for t, c in rows]

@app.get("/", response_class=HTMLResponse)
def page():
    return """
    <body style="background:#111;color:white;font-family:sans-serif;text-align:center">
    <h1>People Count</h1>
    <img src="/video" style="max-width:95%;border-radius:8px">
    </body>"""