# 🎭 Meme Face Recognition

A real-time Python application that uses your webcam to match your live facial expression to the closest human meme image using geometric facial features.

## How It Works

The program captures your face via webcam, extracts geometric ratios from key facial landmarks using **MediaPipe FaceMesh**, and compares them against a pre-built database of meme images. The closest match is displayed side-by-side with your live feed in real time.

**Feature vectors extracted per face:**
- **Left Eye Aspect Ratio (EAR)** — measures eye openness
- **Right Eye Aspect Ratio (EAR)** — measures eye openness
- **Mouth Aspect Ratio (MAR)** — measures how open/wide the mouth is

Matching is done by computing the **Euclidean distance** between the live feature vector and each meme's stored feature vector. The meme with the lowest distance wins.

## Project Structure

```
meme-face-recognition-v1/
├── mfr.py              # Entry point — webcam loop and display logic
├── meme_matcher.py     # MemeMatcher class — landmark detection, feature extraction, matching
├── memes/              # Folder of meme images used as the database
└── run.txt             # Setup and run commands
```

## Requirements

- Python 3.9+
- macOS (Apple Silicon recommended; webcam index may differ on other platforms)
- The following Python libraries:
  - `opencv-python==4.8.0.76`
  - `numpy==1.26.4`
  - `scipy`
  - `mediapipe-silicon` (use `mediapipe` on non-Apple Silicon)

## Setup

**1. Create and activate the virtual environment:**
```bash
python3 -m venv recognition.venv
source recognition.venv/bin/activate
```

**2. Install dependencies:**
```bash
./recognition.venv/bin/python -m pip install "numpy<2.0" opencv-python scipy mediapipe-silicon
./recognition.venv/bin/pip uninstall numpy opencv-python -y
./recognition.venv/bin/pip install "numpy==1.26.4" "opencv-python==4.8.0.76"
```

**3. Set environment variable (prevents threading issues on macOS):**
```bash
export OPENBLAS_NUM_THREADS=1
```

**4. Add meme images to the `memes/` folder.**
Any `.jpg`, `.png`, or similar image file with a visible human face will work.

## Running the App

```bash
./recognition.venv/bin/python mfr.py
```

A window titled **"Meme Matcher LIVE"** will open showing your webcam feed on the left and the best-matching meme on the right.

Press **`ESC`** to quit.

> **Note:** The webcam index is set to `1` in `mfr.py` (Mac built-in camera). Change `cv2.VideoCapture(1)` to `cv2.VideoCapture(0)` if your camera is not detected.

## Tech Stack

| Tool | Purpose |
|---|---|
| [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh) | 468-point facial landmark detection |
| [OpenCV](https://opencv.org/) | Webcam capture and image display |
| [SciPy](https://scipy.org/) | Euclidean distance computation |
| [NumPy](https://numpy.org/) | Landmark array processing |

## Future Improvements

- [ ] Allow matching to non-human meme images (i.e., dogs)
- [ ] Support more expressive feature vectors (brow raise, head tilt)
- [ ] Add a confidence threshold to avoid poor matches

## Author

**Krishna M** — [github.com/kmist70](https://github.com/kmist70)
