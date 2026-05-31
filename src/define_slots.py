import argparse
import json
import os
import sys
import cv2
import numpy as np

WINDOW_NAME = "Define Slots"
WIN_W, WIN_H = 1280, 720


class SlotAnnotator:
    def __init__(self, image, output_path):
        self.output_path = output_path
        self.image       = image
        self.img_h, self.img_w = image.shape[:2]
        # Scale factors: window → real image
        self.sx = self.img_w / WIN_W
        self.sy = self.img_h / WIN_H
        print(f"Image resolution: {self.img_w}x{self.img_h}")
        print(f"Scale factors: sx={self.sx:.3f}, sy={self.sy:.3f}")
        self.temp_image  = image.copy()
        self.slots       = []
        self.current_box = []

    def _win_to_img(self, x, y):
        """Convert window pixel coords to real image coords."""
        return int(x * self.sx), int(y * self.sy)

    def _img_to_win(self, x, y):
        """Convert real image coords to window pixel coords."""
        return int(x / self.sx), int(y / self.sy)

    def _draw_slot(self, img, pts_img, slot_id):
        """Draw slot using image coords (img is full resolution)."""
        pts_win = np.array([self._img_to_win(p[0], p[1]) for p in pts_img], dtype=np.int32)
        for i in range(4):
            cv2.line(img, tuple(pts_win[i]), tuple(pts_win[(i+1)%4]), (0,255,0), 2)
        cx = sum(p[0] for p in pts_win) // 4
        cy = sum(p[1] for p in pts_win) // 4
        cv2.putText(img, f"S{slot_id}", (cx-10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert window coords → real image coords
            rx, ry = self._win_to_img(x, y)
            print(f"  Window ({x},{y}) → Image ({rx},{ry})  — Point {len(self.current_box)+1}/4")
            self.current_box.append([rx, ry])

            # Draw dot at window coords
            cv2.circle(self.temp_image, (x, y), 5, (0,0,255), -1)
            cv2.imshow(WINDOW_NAME, self.temp_image)

            if len(self.current_box) == 4:
                slot_id = len(self.slots)
                self.slots.append({"id": slot_id, "points": self.current_box.copy()})
                self._draw_slot(self.temp_image, self.current_box, slot_id)
                print(f"  ✅ Slot S{slot_id} saved (image coords): {self.current_box}")
                self.current_box = []
                cv2.imshow(WINDOW_NAME, self.temp_image)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_box:
                self.current_box.pop()
                print("  ↩ Last point removed")
            elif self.slots:
                removed = self.slots.pop()
                print(f"  ↩ Slot S{removed['id']} removed")
                self.current_box = []
                self.temp_image  = self.image.copy()
                for slot in self.slots:
                    self._draw_slot(self.temp_image, slot["points"], slot["id"])
            cv2.imshow(WINDOW_NAME, self.temp_image)

    def save(self):
        if not self.slots:
            print("WARNING: No slots defined. Nothing saved.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.slots, f, indent=2)
        print(f"\n✅ Saved {len(self.slots)} slots → {self.output_path}")
        print(f"   File size: {os.path.getsize(self.output_path)} bytes")

    def run(self):
        # Resize image for display only
        display = cv2.resize(self.image, (WIN_W, WIN_H))
        self.temp_image = display.copy()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)
        cv2.imshow(WINDOW_NAME, self.temp_image)
        cv2.setMouseCallback(WINDOW_NAME, self.click_event)

        print("\n=== Controls ===")
        print("LEFT CLICK  : add corner (4 = 1 slot, auto-closes)")
        print("RIGHT CLICK : undo last point / remove last slot")
        print("S           : save & exit")
        print("Q           : quit without saving")
        print("================\n")

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
    p = argparse.ArgumentParser()
    p.add_argument("--image",  default="data/raw/reference_frame_new.jpg")
    p.add_argument("--output", default="data/annotated/parking_slots.json")
    return p.parse_args()


def main():
    args  = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Could not load image: {args.image}")
        sys.exit(1)
    SlotAnnotator(image, args.output).run()


if __name__ == "__main__":
    main()