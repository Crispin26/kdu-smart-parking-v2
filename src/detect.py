import cv2
import json
import os
import sys
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 5, 7]  # COCO: car, bus, truck

model = YOLO("models/yolov8m.pt")

slots_path = "data/annotated/parking_slots.json"
if not os.path.exists(slots_path):
    print(f"ERROR: Slot definitions not found at {slots_path}")
    print("Run define_slots.py first to create parking slot annotations.")
    sys.exit(1)

with open(slots_path, "r", encoding="utf-8") as f:
    slots = json.load(f)


def is_occupied(box, slot_points, frame_shape, threshold=0.20):
    """Check overlap between vehicle bbox and slot polygon."""
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = box
    vehicle_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    slot_poly = np.array(slot_points, dtype=np.int32)

    sx, sy, sw, sh = cv2.boundingRect(slot_poly)
    margin = 50
    rx1 = max(0, sx - margin)
    ry1 = max(0, sy - margin)
    rx2 = min(frame_w, sx + sw + margin)
    ry2 = min(frame_h, sy + sh + margin)

    mask_vehicle = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
    mask_slot = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)

    shifted_vehicle = vehicle_poly - [rx1, ry1]
    shifted_slot = slot_poly - [rx1, ry1]

    cv2.fillPoly(mask_vehicle, [shifted_vehicle], 255)
    cv2.fillPoly(mask_slot, [shifted_slot], 255)

    intersection = cv2.bitwise_and(mask_vehicle, mask_slot)
    slot_area = cv2.countNonZero(mask_slot)
    overlap_area = cv2.countNonZero(intersection)

    if slot_area == 0:
        return False
    return (overlap_area / slot_area) > threshold


def process_frame(frame):
    """Detect vehicles and return raw occupancy per slot."""
    results = model(frame, conf=0.4, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))

    raw_status = []
    for slot in slots:
        occupied = False
        for det in detections:
            if is_occupied(det[:4], slot["points"], frame.shape):
                occupied = True
                break
        raw_status.append(occupied)

    return detections, raw_status


def draw_results(frame, detections, stable_status):
    """Draw slots and detections on frame."""
    overlay = frame.copy()
    for i, slot in enumerate(slots):
        pts = np.array(slot["points"], dtype=np.int32)
        occupied = stable_status[i]
        color = (0, 0, 255) if occupied else (0, 255, 0)

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        cx = sum(p[0] for p in slot["points"]) // 4
        cy = sum(p[1] for p in slot["points"]) // 4
        label = f"S{slot['id']}"
        cv2.putText(frame, label, (cx - 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    free = sum(1 for s in stable_status if not s)
    total = len(stable_status)
    cv2.rectangle(frame, (10, 10), (350, 65), (0, 0, 0), -1)
    cv2.putText(frame, f"FREE: {free}/{total}  |  TAKEN: {total-free}/{total}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame


# --- Main ---
video_path = "data/raw/parking_video.mov"
if not os.path.exists(video_path):
    print(f"ERROR: Video not found at {video_path}")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: OpenCV could not open the video.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
skip_frames = 3

print(f"Video FPS: {fps}")
print(f"Total frames: {total_frames}")
print("Press Q to quit.")

num_slots = len(slots)
stable_status = [False] * num_slots
consecutive_count = [0] * num_slots
STABILITY_THRESHOLD = 5

frame_count = 0
last_detections = []

cv2.namedWindow("Smart Parking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Parking", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % skip_frames == 0:
        last_detections, raw_status = process_frame(frame)

        for i in range(num_slots):
            if raw_status[i] == stable_status[i]:
                consecutive_count[i] = 0
            else:
                consecutive_count[i] += 1
                if consecutive_count[i] >= STABILITY_THRESHOLD:
                    stable_status[i] = raw_status[i]
                    consecutive_count[i] = 0

    frame = draw_results(frame, last_detections, stable_status)
    cv2.imshow("Smart Parking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nProcessed {frame_count}/{total_frames} frames")
print("\n--- Final Slot Status ---")
free = 0
for i, slot in enumerate(slots):
    status = "OCCUPIED" if stable_status[i] else "FREE"
    if not stable_status[i]:
        free += 1
    print(f"  Slot S{slot['id']}: {status}")
print(f"\nTotal: {free} free / {num_slots} total")