import argparse
import cv2
import numpy as np
import time
import subprocess

parser = argparse.ArgumentParser(description="Compare two videos")
parser.add_argument("video1", help="First video path")
parser.add_argument("video2", help="Second video path")
parser.add_argument("fps", help="Playback FPS", type=float)
parser.add_argument(
    "-o",
    "--output",
    default="comparison.mp4",
    help="Output video file",
)

args = parser.parse_args()

cap1 = cv2.VideoCapture(args.video1)
cap2 = cv2.VideoCapture(args.video2)

if not cap1.isOpened():
    raise RuntimeError(f"Cannot open {args.video1}")

if not cap2.isOpened():
    raise RuntimeError(f"Cannot open {args.video2}")

# Read first frame to determine output size
r1, f1 = cap1.read()
r2, f2 = cap2.read()

if not r1 or not r2:
    raise RuntimeError("Could not read frames")

if f1.shape != f2.shape:
    f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))

diff = cv2.absdiff(f1, f2)

top = np.hstack((f1, f2))
bottom = np.hstack((diff, diff))
combined = np.vstack((top, bottom))

scale = 0.2
combined = cv2.resize(combined, None, fx=scale, fy=scale)

height, width = combined.shape[:2]

cmd = [
    "ffmpeg",
    "-y",
    "-f",
    "rawvideo",
    "-vcodec",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "-s",
    f"{width}x{height}",
    "-r",
    str(args.fps),
    "-i",
    "-",
    "-an",
    "-vcodec",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    args.output,
]

proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Comparison", width, height)

# rewind videos
cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    r1, f1 = cap1.read()
    r2, f2 = cap2.read()

    if not r1 or not r2:
        break

    if f1.shape != f2.shape:
        f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))

    diff = cv2.absdiff(f1, f2)

    top = np.hstack((f1, f2))
    bottom = np.hstack((diff, diff))

    combined = np.vstack((top, bottom))
    combined = cv2.resize(combined, None, fx=scale, fy=scale)

    # writer.write(combined)
    proc.stdin.write(combined.tobytes())
    cv2.imshow("Comparison", combined)

    if cv2.waitKey(1) == 27:  # ESC
        break

    time.sleep(1 / args.fps)

cap1.release()
cap2.release()

cv2.destroyAllWindows()
