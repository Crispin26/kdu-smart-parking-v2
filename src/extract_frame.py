import argparse
import sys

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Extract a single frame from a video file")
    p.add_argument("--video",  default="data/raw/parking_video.mov", help="Path to video file")
    p.add_argument("--time",   type=float, default=5.0,              help="Time in seconds to extract (default: 5)")
    p.add_argument("--output", default="data/raw/sample_frame.jpg",  help="Output image path")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {args.video}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("ERROR: Video reports FPS of 0. File may be corrupted or in an unsupported format.")
        cap.release()
        sys.exit(1)

    duration = total_frames / fps
    print(f"Video FPS    : {fps}")
    print(f"Total frames : {total_frames}")
    print(f"Duration     : {duration:.1f} seconds")

    target_frame = int(fps * args.time)
    if target_frame >= total_frames:
        print(f"WARNING: Requested time {args.time}s exceeds video duration {duration:.1f}s. Using last frame.")
        target_frame = total_frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame.")
        sys.exit(1)

    cv2.imwrite(args.output, frame)
    print(f"Frame saved to {args.output}")


if __name__ == "__main__":
    main()
