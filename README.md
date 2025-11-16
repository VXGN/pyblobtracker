# pyblobtracker

## Installation

### From Source
```bash
git clone https://github.com/yourusername/color-blob-tracker.git
cd color-blob-tracker
pip install -r requirements.txt
```

### Using pip (if published)
```bash
pip install color-blob-tracker
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

## Usage Examples

### Track green objects
```bash
python scripts/main.py video.mp4 -c green
```

### Track red objects and save output
```bash
python scripts/main.py video.mp4 -c red -o tracked_video.mp4
```

### Track with custom minimum area
```bash
python scripts/main.py video.mp4 -c blue -m 1000
```

**`requirements.txt`**
```
opencv-python>=4.5.0
numpy>=1.19.0