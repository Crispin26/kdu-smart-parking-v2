import argparse
import json
import os
import sys

import cv2

WINDOW_NAME = "Define Slots"


class SlotAnnotator:
    def __init__(self, image, output_path):
        self.output_path = output_path
        self.image = image
        self.temp_image = image.copy()
        self.slots = []
        self.current_box = []

    def _draw_slot(self, img, pts, slot_id):
        for i in range(4):
            cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
        cx = sum(p[0] for p in pts) // 4
        cy = sum(p[1] for p in pts) // 4
        cv2.putText(img, f"S{slot_id}", (cx - 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at ({x}, {y}) - Point {len(self.current_box) + 1}/4")
            self.current_box.append([x, y])
            cv2.circle(self.temp_image, (x, y), 4, (0, 255, 255), -1)
            cv2.imshow(WINDOW_NAME, self.temp_image)

            if len(self.current_box) == 4:
                slot_id = len(self.slots)
                self.slots.append({"id": slot_id, "points": self.current_box.copy()})
                self._draw_slot(self.temp_image, self.current_box, slot_id)
                print(f">>> Slot S{slot_id} completed!")
                self.current_box = []
                cv2.imshow(WINDOW_NAME, self.temp_image)

        elif event == cv2.EVENT_RBUTTONDOWN and self.slots:
            removed = self.slots.pop()
            print(f"Removed slot S{removed['id']}")
            # Also discard any in-progress points for the next slot
            self.current_box = []
            self.temp_image = self.image.copy()
            for slot in self.slots:
                self._draw_slot(self.temp_image, slot["points"], slot["id"])
            cv2.imshow(WINDOW_NAME, self.temp_image)

    def save(self):
        if self.current_box:
            print(f"WARNING: {len(self.current_box)} unfinished point(s) for the "
                  f"current slot will not be saved. Complete 4 clicks to finish it.")
        if not self.slots:
            print("WARNING: No slots defined. Nothing saved.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.slots, f, indent=2)
        print(f"\nSaved {len(self.slots)} slots to {self.output_path}")
        print(f"File size: {os.path.getsize(self.output_path)} bytes")

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        cv2.imshow(WINDOW_NAME, self.temp_image)
        cv2.setMouseCallback(WINDOW_NAME, self.click_event)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save()
                break
            elif key == ord('q'):
                print("Quit without saving.")
                break

        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Interactive parking slot definition tool")
    p.add_argument("--image",  default="data/raw/reference_frame.jpg",       help="Reference image path")
    p.add_argument("--output", default="data/annotated/parking_slots.json",  help="Output JSON path")
    return p.parse_args()


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Could not load image: {args.image}")
        sys.exit(1)

    print("=== Parking Slot Definition Tool ===")
    print("LEFT CLICK : mark corners (4 clicks = 1 slot)")
    print("RIGHT CLICK: undo last completed slot")
    print("Press S    : save  |  Press Q: quit without saving")
    print("====================================")

    SlotAnnotator(image, args.output).run()


if __name__ == "__main__":
    main()
