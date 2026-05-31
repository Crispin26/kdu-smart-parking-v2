import cv2
import json
import os
import sys
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VEHICLE_CLASSES   = [2, 5, 7, 67]   # car, bus, truck + top-down misclass
YOLO_CONF         = 0.25
YOLO_OVERLAP_THR  = 0.20            # fraction of slot area covered by bbox
DARK_THR          = 75              # brightness below this = dark object in slot
DARK_FRAC         = 0.25            # fraction of slot pixels that must be dark
STABILITY_FRAMES  = 8               # consecutive frames before flipping state
SKIP_FRAMES       = 3               # process every Nth frame
LOG_INTERVAL_SEC  = 5               # log occupancy every N seconds of video time
RESULTS_PATH      = "data/results/occupancy.json"
# ──────────────────────────────────────────────────────────────────────────────

model = YOLO("models/yolov8m.pt")

slots_path = "data/annotated/parking_slots.json"
if not os.path.exists(slots_path):
    print(f"ERROR: Slot definitions not found at {slots_path}")
    sys.exit(1)

with open(slots_path, "r", encoding="utf-8") as f:
    slots = json.load(f)

os.makedirs(os.path.dirname(os.path.abspath(RESULTS_PATH)), exist_ok=True)


# ── Occupancy Logger ──────────────────────────────────────────────────────────
class OccupancyLogger:
    def __init__(self, path):
        self.path    = path
        self.records = []
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.records = json.load(f)
                print(f"Loaded {len(self.records)} existing records from {path}")
            except Exception:
                self.records = []

    def log(self, stable_status, video_time_sec):
        record = {
            "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_time_sec": round(video_time_sec, 1),
            "slots": {
                f"S{slot['id']}": "OCCUPIED" if stable_status[i] else "FREE"
                for i, slot in enumerate(slots)
            },
            "total_free":     sum(1 for s in stable_status if not s),
            "total_occupied": sum(1 for s in stable_status if s),
            "total_slots":    len(stable_status)
        }
        self.records.append(record)
        self._save()
        return record

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def summary(self):
        print(f"\n📊 Logged {len(self.records)} records -> {self.path}")


# ── Detection helpers ─────────────────────────────────────────────────────────
def slot_mask(slot_points, frame_shape):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(slot_points, dtype=np.int32)], 255)
    return mask


def yolo_occupied(box, slot_points, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2  = box
    vehicle_poly     = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.int32)
    slot_poly        = np.array(slot_points, dtype=np.int32)
    sx, sy, sw, sh   = cv2.boundingRect(slot_poly)
    margin = 50
    rx1 = max(0, sx-margin);          ry1 = max(0, sy-margin)
    rx2 = min(frame_w, sx+sw+margin); ry2 = min(frame_h, sy+sh+margin)
    mv  = np.zeros((ry2-ry1, rx2-rx1), dtype=np.uint8)
    ms  = np.zeros((ry2-ry1, rx2-rx1), dtype=np.uint8)
    cv2.fillPoly(mv, [vehicle_poly - [rx1,ry1]], 255)
    cv2.fillPoly(ms, [slot_poly    - [rx1,ry1]], 255)
    slot_area    = cv2.countNonZero(ms)
    overlap_area = cv2.countNonZero(cv2.bitwise_and(mv, ms))
    if slot_area == 0:
        return False
    return (overlap_area / slot_area) > YOLO_OVERLAP_THR


def pixel_occupied(frame_gray, slot_points, frame_shape):
    mask        = slot_mask(slot_points, frame_shape)
    slot_area   = cv2.countNonZero(mask)
    if slot_area == 0:
        return False
    slot_pixels = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
    dark_pixels = np.sum((slot_pixels > 0) & (slot_pixels < DARK_THR))
    return (dark_pixels / slot_area) > DARK_FRAC


def process_frame(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = model(frame, conf=YOLO_CONF, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2, float(box.conf[0])))

    raw_status = []
    for slot in slots:
        yolo_hit = any(yolo_occupied(d[:4], slot["points"], frame.shape)
                       for d in detections)
        if yolo_hit:
            raw_status.append(True)
        else:
            raw_status.append(pixel_occupied(gray, slot["points"], frame.shape))
    return detections, raw_status


def draw_results(frame, detections, stable_status, video_time_sec):
    overlay = frame.copy()
    for i, slot in enumerate(slots):
        pts   = np.array(slot["points"], dtype=np.int32)
        color = (0, 0, 255) if stable_status[i] else (0, 255, 0)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cx = sum(p[0] for p in slot["points"]) // 4
        cy = sum(p[1] for p in slot["points"]) // 4
        cv2.putText(frame, f"S{slot['id']}", (cx-10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    free  = sum(1 for s in stable_status if not s)
    total = len(stable_status)
    cv2.rectangle(frame, (10, 10), (420, 90), (0,0,0), -1)
    cv2.putText(frame, f"FREE: {free}/{total}  |  TAKEN: {total-free}/{total}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    mins = int(video_time_sec // 60)
    secs = int(video_time_sec % 60)
    cv2.putText(frame, f"Video time: {mins:02d}:{secs:02d}",
                (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
video_path = "data/raw/Video_edited_ksms.mp4"
if not os.path.exists(video_path):
    print(f"ERROR: Video not found at {video_path}")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: Could not open video.")
    sys.exit(1)

fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}  |  Total frames: {total_frames}")
print(f"Logging every {LOG_INTERVAL_SEC}s -> {RESULTS_PATH}")
print("Press Q to quit.\n")

num_slots       = len(slots)
stable_status   = [False] * num_slots
consecutive     = [0]     * num_slots
frame_count     = 0
last_detections = []
last_log_sec    = -LOG_INTERVAL_SEC

logger = OccupancyLogger(RESULTS_PATH)

cv2.namedWindow("Smart Parking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Parking", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count   += 1
    video_time_sec = frame_count / fps

    if frame_count % SKIP_FRAMES == 0:
        last_detections, raw_status = process_frame(frame)
        for i in range(num_slots):
            if raw_status[i] == stable_status[i]:
                consecutive[i] = 0
            else:
                consecutive[i] += 1
                if consecutive[i] >= STABILITY_FRAMES:
                    stable_status[i] = raw_status[i]
                    consecutive[i]   = 0

    if video_time_sec - last_log_sec >= LOG_INTERVAL_SEC:
        rec = logger.log(stable_status, video_time_sec)
        print(f"  [{rec['video_time_sec']:6.1f}s] "
              f"FREE={rec['total_free']}  OCCUPIED={rec['total_occupied']}")
        last_log_sec = video_time_sec

    frame = draw_results(frame, last_detections, stable_status, video_time_sec)
    cv2.imshow("Smart Parking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
logger.summary()

print(f"\nProcessed {frame_count}/{total_frames} frames")
print("\n--- Final Slot Status ---")
free = 0
for i, slot in enumerate(slots):
    status = "OCCUPIED" if stable_status[i] else "FREE"
    if not stable_status[i]:
        free += 1
    print(f"  Slot S{slot['id']}: {status}")
print(f"\nTotal: {free} free / {num_slots} total")