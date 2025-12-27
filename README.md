# pyblobtracker

## Installation
```bash
git clone https://github.com/VXGN/color-blob-tracker.git
cd color-blob-tracker
pip install -r requirements.txt
```

## Quick Start
```bash
python scripts/main.py input_video.mp4 -c green -o output.mp4 -m 500
```

### Arguments
- `input`: Path to input video file
- `-c, --color`: Color to track (default: green)
- `-o, --output`: Output video path (optional)
- `-m, --min-area`: Minimum blob area in pixels (default: 500)

**`requirements.txt`**
```
opencv-python>=4.5.0
numpy>=1.19.0