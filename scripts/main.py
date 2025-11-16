import argparse
from src.blobtracker.video_processor import VideoProcessor
from src.blobtracker.tracker import BlobTracker
from src.blobtracker.mask_processor import MaskProcessor
from src.blobtracker.contour_detector import ContourDetector
from src.blobtracker.visualizer import Visualizer
from src.blobtracker.color_ranges import COLOR_RANGES

def main():
    parser = argparse.ArgumentParser(description='Decoupled Color Blob Tracker')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-c', '--color', default='green', help='Color to track')
    parser.add_argument('-o', '--output', help='Output video file (optional)')
    parser.add_argument('-m', '--min-area', type=int, default=500, help='Minimum blob area')
    
    args = parser.parse_args()
    
    if args.color not in COLOR_RANGES:
        print(f"Error: Unknown color '{args.color}'")
        print(f"Available: {', '.join(COLOR_RANGES.keys())}")
        return
    
    mask_processor = MaskProcessor(COLOR_RANGES)
    contour_detector = ContourDetector()
    visualizer = Visualizer()
    video_processor = VideoProcessor(args.input)
    
    tracker = BlobTracker(mask_processor, contour_detector, visualizer)
    
    try:
        tracker.track_video(video_processor, args.color, args.min_area, args.output)
    finally:
        video_processor.release()

if __name__ == "__main__":
    main()