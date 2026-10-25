import cv2
import os
import csv
import time
import math
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))


def find_video():
    # folder la irukkura video-va thaanaaga kandupidikkum (people.mp4 / people.mp4.mp4 / vera edhaavadhu .mp4)
    for name in ("people.mp4", "people.mp4.mp4"):
        p = os.path.join(BASE, name)
        if os.path.exists(p):
            return p
    for fn in sorted(os.listdir(BASE)):
        if fn.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return os.path.join(BASE, fn)
    return os.path.join(BASE, "people.mp4")


VIDEO = os.environ.get("COUNT_VIDEO") or find_video()
OUT_DIR = os.environ.get("COUNT_OUT") or BASE
COUNT_FILE = os.path.join(OUT_DIR, "count_10min.csv")    # every 10 min: frame la yaar irukkanga (IDs kooda)
CROSS_FILE = os.path.join(OUT_DIR, "crossing_2min.csv")  # every 2 min: line-a yaar kadandhanga IN / OUT (IDs kooda)
TRACKER_FILE = os.path.join(OUT_DIR, "tracker_custom.yaml")

# ---------------- settings ----------------
MODEL_NAME = os.environ.get("COUNT_MODEL") or "yolov8s.pt"   # slow na yolov8n.pt, accuracy venum na yolov8m.pt
IMGSZ = 640
CONF = 0.35
REALTIME = os.environ.get("COUNT_REALTIME", "1") == "1"      # True = video normal speed
SHOW = os.environ.get("COUNT_SHOW", "1") == "1"              # False = window kaattaama (test)

PROCESS_FPS = 10            # video la vinadiku ithana frame mattum model-ku anuppum (vega + tracker stable)

# ---- IN / OUT (frame ulla vandha = IN, frame vittu pona = OUT) ----
MIN_HITS = 5                # ithana frame therinja aal dhaan "nijamaana aal" (flicker / false detection avoid)
MIN_LIFE = 1.0              # ithana sec frame la irundha dhaan IN count aagum
OUT_GRACE = 3.0             # frame NADULA maraindha aal: ithana sec kazhichu dhaan OUT (maraichu nindrutaanga na wait)
OUT_GRACE_EDGE = 1.0        # frame OORAM la maraindha aal: seekiram OUT
EDGE_MARGIN = 0.10          # frame ooram = 10% border
START_IGNORE = 1.0          # video modhal ithana sec la irukkavanga "munnadiye ulla irundhaanga" (IN count aagaadhu)

# ---- ID heal: tracker ID maatha-na, pazhaya ID-a thirumba use pannum (IN/OUT double count aagaadhu) ----
HEAL_DIST = 1.0             # (box height x ithu) dhooram kulla irundha dhaan same aal

CROSS_INTERVAL = int(os.environ.get("COUNT_CROSS_INT", 120))    # 2 min (seconds). Test ku 10 nu vainga
COUNT_INTERVAL = int(os.environ.get("COUNT_COUNT_INT", 600))    # 10 min (seconds). Test ku 20 nu vainga
# ------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# tracker settings (ByteTrack)
with open(TRACKER_FILE, "w") as f:
    f.write(f"""tracker_type: bytetrack
track_high_thresh: 0.30
track_low_thresh: 0.10
new_track_thresh: 0.35
track_buffer: {int(PROCESS_FPS * 3)}
match_thresh: 0.8
fuse_score: True
""")


def save_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def mmss(sec):
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def ids_text(lst):
    return ";".join(str(i) for i in lst)


CROSS_HDR = ["time", "in", "out", "total_crossed", "in_ids", "out_ids"]
COUNT_HDR = ["time", "live_count", "peak_count", "ids_in_frame"]

# person ID: 5 digit - 10001, 10002, 10003 ... (ovvoru run-lum same maadhiri varum)
next_pid = 10001
tid_to_pid = {}             # tracker id -> person id
people = {}                 # person id -> {last_t, cx, fy, h, hits, side, in_done, out_done}


def new_pid():
    global next_pid
    p = next_pid
    next_pid += 1
    return p


model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
STRIDE = max(1, int(round(fps / PROCESS_FPS)))

print("Video file:", VIDEO)
print("File irukka:", os.path.exists(VIDEO))
print("Video open aachu:", cap.isOpened(), "| fps:", round(fps, 1), "| every", STRIDE, "frame process")

win_in_ids, win_out_ids = [], []
tot_in = tot_out = 0
peak = 0
live = 0
live_ids = []
next_cross = CROSS_INTERVAL
next_count = COUNT_INTERVAL
frame_no = 0
t = 0.0
start = time.time()
finished = False

while not finished:
    # real-time: video neram ku pinnaadi irundha frames skip pannu (STRIDE frames-a thaan process)
    target = int((time.time() - start) * fps) if REALTIME else frame_no + STRIDE
    while frame_no < target - STRIDE:
        if not cap.grab():
            finished = True
            break
        frame_no += 1
    if finished:
        break
    # STRIDE frames munnadi poi, adhula irukkura frame-a edu
    ok = False
    for _ in range(STRIDE):
        ok = cap.grab()
        if not ok:
            break
        frame_no += 1
    if not ok:
        break
    ok, frame = cap.retrieve()
    if not ok:
        break
    t = frame_no / fps      # video time (seconds)

    h, w = frame.shape[:2]
    scale = min(960 / w, 800 / h)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    H, W = frame.shape[:2]
    r = model.track(frame, persist=True, classes=[0], conf=CONF, imgsz=IMGSZ,
                    tracker=TRACKER_FILE, verbose=False)[0]

    live_ids = []
    seen_now = set()
    dets = []
    if r.boxes.id is not None:
        tids = r.boxes.id.int().tolist()
        boxes = r.boxes.xyxy.tolist()
        dets = list(zip(tids, boxes))

    # 1) munnadiye theriyum tracker ID-kal
    for tid, _ in dets:
        if tid in tid_to_pid and tid_to_pid[tid] in people:
            seen_now.add(tid_to_pid[tid])

    mx, my = EDGE_MARGIN * W, EDGE_MARGIN * H

    # 2) pudhu tracker ID: sila nodi munnadi maraindha aal-a irukkalaam -> heal (same person ID)
    for tid, (x1, y1, x2, y2) in dets:
        if tid in tid_to_pid and tid_to_pid[tid] in people:
            continue
        cx, fy, bh = (x1 + x2) / 2, y2, max(1.0, y2 - y1)
        best, bd = None, HEAL_DIST * bh
        for lp, li in people.items():
            if lp in seen_now:
                continue
            gap = t - li["last_t"]
            if gap > OUT_GRACE or (li["edge"] and gap > OUT_GRACE_EDGE):
                continue            # idhu veliya poiduchu
            d = math.hypot(cx - li["cx"], fy - li["fy"])
            if d < bd:
                best, bd = lp, d
        if best is not None:
            pid = best
            for k in [k for k, v in tid_to_pid.items() if v == pid]:
                del tid_to_pid[k]
        else:
            pid = new_pid()
            people[pid] = {"first_t": t, "last_t": t, "cx": cx, "fy": fy, "h": bh, "hits": 0,
                           "edge": False, "ok": False, "initial": t <= START_IGNORE}
        tid_to_pid[tid] = pid
        seen_now.add(pid)

    # 3) update + IN
    for tid, (x1, y1, x2, y2) in dets:
        pid = tid_to_pid[tid]
        p = people[pid]
        cx, fy, bh = (x1 + x2) / 2, y2, max(1.0, y2 - y1)
        p.update(last_t=t, cx=cx, fy=fy, h=bh)
        p["hits"] += 1
        p["edge"] = (x1 < mx or x2 > W - mx or y1 < my or y2 > H - my)
        live_ids.append(pid)

        if not p["ok"] and p["hits"] >= MIN_HITS and (t - p["first_t"]) >= MIN_LIFE:
            p["ok"] = True
            if not p["initial"]:
                tot_in += 1
                win_in_ids.append(pid)
                print(f"ID {pid} IN  at {mmss(t)}")

        if SHOW:
            color = (0, 255, 0) if p["ok"] else (0, 165, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID {pid}", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # 4) OUT: kanaama ponavanga
    for pid in list(people.keys()):
        p = people[pid]
        if pid in seen_now:
            continue
        grace = OUT_GRACE_EDGE if p["edge"] else OUT_GRACE
        if t - p["last_t"] > grace:
            if p["ok"]:
                tot_out += 1
                win_out_ids.append(pid)
                print(f"ID {pid} OUT at {mmss(p['last_t'])}")
            del people[pid]
            for k in [k for k, v in tid_to_pid.items() if v == pid]:
                del tid_to_pid[k]

    live = len(live_ids)
    peak = max(peak, live)

    # 2 min: in / out save (IDs kooda)
    while t >= next_cross:
        save_row(CROSS_FILE, CROSS_HDR,
                 [mmss(next_cross), len(win_in_ids), len(win_out_ids),
                  len(win_in_ids) + len(win_out_ids),
                  ids_text(win_in_ids), ids_text(win_out_ids)])
        print(f"[{mmss(next_cross)}] IN {len(win_in_ids)}, OUT {len(win_out_ids)}")
        win_in_ids, win_out_ids = [], []
        next_cross += CROSS_INTERVAL

    # 10 min: frame la irukkura aatkal + IDs
    while t >= next_count:
        save_row(COUNT_FILE, COUNT_HDR,
                 [mmss(next_count), live, peak, ids_text(live_ids)])
        print(f"[{mmss(next_count)}] live {live}, peak {peak}")
        peak = 0
        next_count += COUNT_INTERVAL

    if SHOW:
        cv2.rectangle(frame, (0, 0), (420, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Live Count: {live}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"IN: {tot_in}   OUT: {tot_out}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {mmss(t)}", (10, 93),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("People Count", frame)

        # processing vegama mudinjaa, video normal speed ku wait pannu
        wait_ms = 1
        if REALTIME:
            ahead = t - (time.time() - start)
            if ahead > 0:
                wait_ms = max(1, int(ahead * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break

# partial window save
save_row(CROSS_FILE, CROSS_HDR,
         [mmss(t) + " (end)", len(win_in_ids), len(win_out_ids),
          len(win_in_ids) + len(win_out_ids),
          ids_text(win_in_ids), ids_text(win_out_ids)])
save_row(COUNT_FILE, COUNT_HDR,
         [mmss(t) + " (end)", live, peak, ids_text(live_ids)])

cap.release()
if SHOW:
    cv2.destroyAllWindows()
print(f"Mudinjuchu. Total IN {tot_in}, OUT {tot_out}, unique people {next_pid - 10001}. CSV files folder la save aagirukku.")