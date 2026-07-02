import argparse
import cv2
import numpy as np
import time
import subprocess


def create_layout(f1, f2, scale=0.2):
    """Oblicza różnicę, powiększa ją proporcjonalnie i łączy w jeden kadr."""
    # 1. Obliczenie różnicy na oryginalnej rozdzielczości
    diff = cv2.absdiff(f1, f2)

    # 2. Skalowanie obrazów do górnego rzędu
    f1_s = cv2.resize(f1, None, fx=scale, fy=scale)
    f2_s = cv2.resize(f2, None, fx=scale, fy=scale)

    H, W = f1_s.shape[:2]

    # 3. Górny rząd (wideo 1 obok wideo 2 -> łączna szerokość to 2 * W)
    top = np.hstack((f1_s, f2_s))

    # 4. Przeskalowanie diffa na CAŁĄ szerokość (2 * W).
    # Aby zachować proporcje, wysokość również mnożymy x2 (2 * H).
    diff_large = cv2.resize(diff, (2 * W, 2 * H))

    # 5. Dolny rząd - czarne tło z miejscem na tekst (wysokość to teraz 2 * H + tekst)
    text_height = 40
    bottom = np.zeros((2 * H + text_height, 2 * W, 3), dtype=np.uint8)

    # 6. Dodanie wyśrodkowanego podpisu
    text = "movie1 - movie2"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    text_x = (2 * W - text_size[0]) // 2
    text_y = text_height - 10
    cv2.putText(
        bottom,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # 7. Wklejenie powiększonego obrazu różnicy (wypełnia idealnie przestrzeń)
    bottom[text_height : text_height + 2 * H, 0 : 2 * W] = diff_large

    # 8. Połączenie góry i dołu
    combined = np.vstack((top, bottom))
    return combined


parser = argparse.ArgumentParser(description="Compare two videos")
parser.add_argument("video1", help="First video path")
parser.add_argument("video2", help="Second video path")
parser.add_argument("fps", help="Playback FPS", type=float)
parser.add_argument(
    "-o", "--output", default="comparison.mp4", help="Output video file"
)
args = parser.parse_args()

cap1 = cv2.VideoCapture(args.video1)
cap2 = cv2.VideoCapture(args.video2)

if not cap1.isOpened():
    raise RuntimeError(f"Cannot open {args.video1}")

if not cap2.isOpened():
    raise RuntimeError(f"Cannot open {args.video2}")

r1, f1 = cap1.read()
r2, f2 = cap2.read()

if not r1 or not r2:
    raise RuntimeError("Could not read frames")

if f1.shape != f2.shape:
    f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))

scale = 0.16
combined = create_layout(f1, f2, scale)
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

cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    r1, f1 = cap1.read()
    r2, f2 = cap2.read()

    if not r1 or not r2:
        break

    if f1.shape != f2.shape:
        f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))

    combined = create_layout(f1, f2, scale)

    proc.stdin.write(combined.tobytes())
    cv2.imshow("Comparison", combined)

    if cv2.waitKey(1) == 27:  # ESC
        break

    time.sleep(1 / args.fps)

cap1.release()
cap2.release()

proc.stdin.close()
proc.wait()

cv2.destroyAllWindows()
