import cv2
import numpy as np
import argparse
import sys
import time
from collections import deque


# ── Color Presets (HSV) ──────────────────────────────────────────────────────
# Each preset: (H_low, S_low, V_low, H_high, S_high, V_high, display_bgr)
COLOR_PRESETS = {
    "red": {
        # Red wraps around 0/180 in HSV, so we use two ranges
        "ranges": [
            (np.array([0,   120, 70]),  np.array([10,  255, 255])),
            (np.array([170, 120, 70]),  np.array([180, 255, 255])),
        ],
        "trackbar_low":  [0,   120, 70],
        "trackbar_high": [10,  255, 255],
        "bgr": (0, 0, 220),
        "name": "Red",
    },
    "green": {
        "ranges": [
            (np.array([35, 80, 60]),  np.array([85, 255, 255])),
        ],
        "trackbar_low":  [35, 80, 60],
        "trackbar_high": [85, 255, 255],
        "bgr": (0, 200, 0),
        "name": "Green",
    },
    "blue": {
        "ranges": [
            (np.array([100, 100, 60]),  np.array([130, 255, 255])),
        ],
        "trackbar_low":  [100, 100, 60],
        "trackbar_high": [130, 255, 255],
        "bgr": (220, 60, 0),
        "name": "Blue",
    },
    "yellow": {
        "ranges": [
            (np.array([20, 100, 100]),  np.array([35, 255, 255])),
        ],
        "trackbar_low":  [20, 100, 100],
        "trackbar_high": [35, 255, 255],
        "bgr": (0, 215, 255),
        "name": "Yellow",
    },
    "orange": {
        "ranges": [
            (np.array([10, 150, 100]),  np.array([25, 255, 255])),
        ],
        "trackbar_low":  [10, 150, 100],
        "trackbar_high": [25, 255, 255],
        "bgr": (0, 130, 255),
        "name": "Orange",
    },
}

PRESET_KEYS = list(COLOR_PRESETS.keys())   # index 0-4 → keys 1-5


# ── Morphological kernels ────────────────────────────────────────────────────
ERODE_KERNEL  = np.ones((5, 5), np.uint8)
DILATE_KERNEL = np.ones((9, 9), np.uint8)


# ── Trackbar window name ─────────────────────────────────────────────────────
TB_WIN = "HSV Tuner"


def nothing(_):
    pass


def create_trackbars(preset: dict):
    cv2.namedWindow(TB_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TB_WIN, 400, 200)
    lo = preset["trackbar_low"]
    hi = preset["trackbar_high"]
    cv2.createTrackbar("H Low",  TB_WIN, lo[0], 180, nothing)
    cv2.createTrackbar("S Low",  TB_WIN, lo[1], 255, nothing)
    cv2.createTrackbar("V Low",  TB_WIN, lo[2], 255, nothing)
    cv2.createTrackbar("H High", TB_WIN, hi[0], 180, nothing)
    cv2.createTrackbar("S High", TB_WIN, hi[1], 255, nothing)
    cv2.createTrackbar("V High", TB_WIN, hi[2], 255, nothing)


def read_trackbars():
    lo = np.array([
        cv2.getTrackbarPos("H Low",  TB_WIN),
        cv2.getTrackbarPos("S Low",  TB_WIN),
        cv2.getTrackbarPos("V Low",  TB_WIN),
    ])
    hi = np.array([
        cv2.getTrackbarPos("H High", TB_WIN),
        cv2.getTrackbarPos("S High", TB_WIN),
        cv2.getTrackbarPos("V High", TB_WIN),
    ])
    return lo, hi


def update_trackbars(preset: dict):
    lo = preset["trackbar_low"]
    hi = preset["trackbar_high"]
    cv2.setTrackbarPos("H Low",  TB_WIN, lo[0])
    cv2.setTrackbarPos("S Low",  TB_WIN, lo[1])
    cv2.setTrackbarPos("V Low",  TB_WIN, lo[2])
    cv2.setTrackbarPos("H High", TB_WIN, hi[0])
    cv2.setTrackbarPos("S High", TB_WIN, hi[1])
    cv2.setTrackbarPos("V High", TB_WIN, hi[2])


def build_mask(hsv: np.ndarray, preset_name: str, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Build combined mask from preset ranges OR manual trackbar values."""
    if preset_name == "red":
        # two ranges for red (wraps around 0/180)
        m1 = cv2.inRange(hsv, np.array([0,   lo[1], lo[2]]), np.array([10,  hi[1], hi[2]]))
        m2 = cv2.inRange(hsv, np.array([170, lo[1], lo[2]]), np.array([180, hi[1], hi[2]]))
        mask = cv2.bitwise_or(m1, m2)
    else:
        mask = cv2.inRange(hsv, lo, hi)

    # Clean up noise
    mask = cv2.erode(mask,  ERODE_KERNEL,  iterations=1)
    mask = cv2.dilate(mask, DILATE_KERNEL, iterations=2)
    return mask


def find_largest_contour(mask: np.ndarray):
    """Return (contour, cx, cy, radius) of the largest blob, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 500:           # too small → ignore
        return None
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return c, cx, cy, int(radius)


def draw_hud(frame: np.ndarray, fps: float, preset_name: str,
             trail_on: bool, mask_on: bool, paused: bool, found: bool,
             cx: int = 0, cy: int = 0):
    """Draw semi-transparent HUD overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top bar
    cv2.rectangle(overlay, (0, 0), (w, 36), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    info = (f"FPS:{fps:5.1f}  Color:{preset_name.upper()}"
            f"  Trail:{'ON' if trail_on else 'OFF'}"
            f"  Mask:{'ON' if mask_on else 'OFF'}"
            + ("  [PAUSED]" if paused else ""))
    cv2.putText(frame, info, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Detection status
    status = f"TRACKING ({cx},{cy})" if found else "SEARCHING..."
    color  = (0, 255, 80) if found else (0, 80, 255)
    cv2.putText(frame, status, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Key hint (bottom-right)
    hint = "q=quit  t=trail  m=mask  r=reset  1-5=color  spc=pause"
    tw, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0], None
    cv2.putText(frame, hint, (w - tw[0] - 6, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, trail: deque, color_bgr: tuple):
    """Draw fading dot trail."""
    pts = list(trail)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        alpha = i / len(pts)
        thickness = max(1, int(alpha * 5))
        fade = tuple(int(c * alpha) for c in color_bgr)
        cv2.line(frame, pts[i - 1], pts[i], fade, thickness)


# ── Demo frame generator ─────────────────────────────────────────────────────
def demo_frame(t: float, w=640, h=480):
    """Generate a synthetic frame with a moving colored ball."""
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    # Background texture
    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (40, 40, 40), 1)
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (40, 40, 40), 1)

    # Moving ball (Lissajous path)
    cx = int(w // 2 + (w // 3) * np.sin(t * 0.9))
    cy = int(h // 2 + (h // 3) * np.sin(t * 1.3))
    cv2.circle(frame, (cx, cy), 35, (0, 0, 200), -1)   # red ball
    cv2.circle(frame, (cx, cy), 35, (0, 0, 255), 2)
    # Highlight
    cv2.circle(frame, (cx - 10, cy - 10), 8, (80, 80, 255), -1)
    return frame


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Color-based Object Tracker")
    parser.add_argument("--video",  default=None, help="Path to video file (default: webcam)")
    parser.add_argument("--color",  default="red", choices=PRESET_KEYS, help="Starting color preset")
    parser.add_argument("--trail",  type=int, default=64, help="Trail length (default 64)")
    parser.add_argument("--demo",   action="store_true", help="Use synthetic demo frames")
    args = parser.parse_args()

    # ── Source setup ──────────────────────────────────────────────────────────
    demo_mode = args.demo
    cap = None
    if not demo_mode:
        src = args.video if args.video else 0
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {src}")
            print("Tip: run with --demo to use synthetic frames without a camera.")
            sys.exit(1)

    # ── State ─────────────────────────────────────────────────────────────────
    preset_name  = args.color
    preset       = COLOR_PRESETS[preset_name]
    trail        = deque(maxlen=args.trail)
    trail_on     = True
    mask_on      = False
    paused       = False
    prev_time    = time.time()
    fps          = 0.0
    demo_t       = 0.0

    create_trackbars(preset)

    cv2.namedWindow("Color Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Color Tracker", 800, 600)

    print("\n┌─ Color Object Tracker ─────────────────────────────┐")
    print("│  Controls:                                          │")
    print("│  q / ESC  → Quit                                    │")
    print("│  t        → Toggle trail                            │")
    print("│  m        → Toggle mask view                        │")
    print("│  r        → Reset trail                             │")
    print("│  1-5      → Switch preset (red/green/blue/yel/org)  │")
    print("│  SPACE    → Pause / Resume                          │")
    print("└─────────────────────────────────────────────────────┘\n")

    while True:
        # ── Read frame ────────────────────────────────────────────────────────
        if not paused:
            if demo_mode:
                frame = demo_frame(demo_t)
                demo_t += 0.05
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of stream.")
                    break
                frame = cv2.resize(frame, (640, 480))

        # FPS
        now      = time.time()
        fps      = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── HSV conversion + mask ─────────────────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo, hi = read_trackbars()
        mask = build_mask(hsv, preset_name, lo, hi)

        # ── Find object ───────────────────────────────────────────────────────
        result = find_largest_contour(mask)
        found  = result is not None
        cx = cy = 0

        if found:
            contour, cx, cy, radius = result

            if trail_on:
                trail.append((cx, cy))

            # Draw bounding circle
            cv2.circle(frame, (cx, cy), radius + 4, preset["bgr"], 2)
            # Crosshair
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), preset["bgr"], 1)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), preset["bgr"], 1)
            # Centroid dot
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            # Label
            label = f"{preset['name']} r={radius}px"
            cv2.putText(frame, label, (cx + radius + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, preset["bgr"], 1, cv2.LINE_AA)
            # Draw contour outline
            cv2.drawContours(frame, [contour], -1, preset["bgr"], 1)
        else:
            trail.append(None)

        # ── Trail ─────────────────────────────────────────────────────────────
        if trail_on:
            draw_trail(frame, trail, preset["bgr"])

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_hud(frame, fps, preset_name, trail_on, mask_on, paused, found, cx, cy)

        # ── Display ───────────────────────────────────────────────────────────
        if mask_on:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            display  = np.hstack([frame, mask_3ch])
        else:
            display = frame

        cv2.imshow("Color Tracker", display)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):           # q or ESC
            break
        elif key == ord('t'):
            trail_on = not trail_on
        elif key == ord('m'):
            mask_on = not mask_on
        elif key == ord('r'):
            trail.clear()
        elif key == ord(' '):
            paused = not paused
        elif ord('1') <= key <= ord('5'):
            idx          = key - ord('1')
            if idx < len(PRESET_KEYS):
                preset_name  = PRESET_KEYS[idx]
                preset       = COLOR_PRESETS[preset_name]
                update_trackbars(preset)
                trail.clear()
                print(f"[INFO] Switched to preset: {preset_name}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Tracker closed.")


if __name__ == "__main__":
    main()