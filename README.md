# 🎯 Color-Based Object Tracker

A real-time object tracking tool built with OpenCV that detects and follows colored objects using HSV color space filtering. Supports live webcam feeds, video files, and a built-in synthetic demo mode.

---

## Features

- **Real-time color tracking** using HSV masking and contour detection
- **5 built-in color presets**: Red, Green, Blue, Yellow, Orange
- **Live HSV tuner** via trackbars for fine-tuning detection
- **Fading motion trail** to visualize the object's path
- **Mask view** to inspect the binary detection mask side-by-side
- **Demo mode** — no camera required, uses a synthetic moving ball
- **On-screen HUD** showing FPS, tracking status, and coordinates
- **Pause/Resume** support

---

## Requirements

- Python 3.7+
- OpenCV
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Usage

### Webcam (default)
```bash
python object_tracking_based_on_colour.py
```

### Video file
```bash
python object_tracking_based_on_colour.py --video path/to/video.mp4
```

### Demo mode (no camera needed)
```bash
python object_tracking_based_on_colour.py --demo
```

### Start with a specific color preset
```bash
python object_tracking_based_on_colour.py --color green
```

### Custom trail length
```bash
python object_tracking_based_on_colour.py --trail 128
```

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | `None` (webcam) | Path to a video file |
| `--color` | `red` | Starting color preset (`red`, `green`, `blue`, `yellow`, `orange`) |
| `--trail` | `64` | Number of trail points to keep |
| `--demo` | `False` | Use synthetic demo frames instead of a camera |

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit the tracker |
| `t` | Toggle motion trail on/off |
| `m` | Toggle mask view (side-by-side with frame) |
| `r` | Reset/clear the trail |
| `1` | Switch to **Red** preset |
| `2` | Switch to **Green** preset |
| `3` | Switch to **Blue** preset |
| `4` | Switch to **Yellow** preset |
| `5` | Switch to **Orange** preset |
| `SPACE` | Pause / Resume |

---

## How It Works

1. Each frame is converted from BGR to **HSV color space**, which separates color (hue) from brightness, making detection more robust to lighting changes.
2. A **binary mask** is created by thresholding pixels within the target HSV range.
3. Morphological operations (**erosion + dilation**) remove noise and fill gaps in the mask.
4. The **largest contour** in the mask is found and used to compute the object's centroid and bounding circle.
5. A **fading trail** is drawn by storing recent centroid positions in a rolling deque.

> **Note on Red:** Red wraps around 0°/180° in HSV, so two separate ranges are used and combined with a bitwise OR.

---

## HSV Tuner

When the tracker runs, an **HSV Tuner** window opens with 6 sliders:

- `H Low` / `H High` — Hue range (0–180)
- `S Low` / `S High` — Saturation range (0–255)
- `V Low` / `V High` — Value/brightness range (0–255)

Adjust these sliders to fine-tune detection for your specific lighting conditions or object color. Switching presets (keys `1`–`5`) will reset the sliders to that preset's defaults.

---

## Color Presets (HSV Ranges)

| Color  | H Low | H High | S Low | V Low |
|--------|-------|--------|-------|-------|
| Red    | 0–10 & 170–180 | — | 120 | 70 |
| Green  | 35    | 85     | 80    | 60  |
| Blue   | 100   | 130    | 100   | 60  |
| Yellow | 20    | 35     | 100   | 100 |
| Orange | 10    | 25     | 150   | 100 |

---

## Project Structure

```
object_tracking_based_on_colour.py   # Main script (single file)
README.md                            # This file
```

---

## Tips

- For best results, ensure the tracked object is **well-lit** and contrasts with the background.
- If detection is noisy, **lower the V Low** value or **raise S Low** to filter out dim/gray pixels.
- Use **demo mode** to explore controls and behavior without a physical object.
- The minimum detectable contour area is **500 px²** — objects too small or far away will be ignored.
