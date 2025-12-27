import argparse
from blobtracker.mask_processor import MaskProcessor
from blobtracker.contour_detector import ContourDetector
from blobtracker.visualizer import Visualizer
from blobtracker.video_processor import VideoProcessor
from blobtracker.tracker import BlobTracker
from blobtracker.color_ranges import COLOR_RANGES

def main():
    parser = argparse.ArgumentParser(description='Decoupled Color Blob Tracker')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-c', '--color', default='green', help='Color to track')
    parser.add_argument('-o', '--output', help='Output video file (optional)')
    parser.add_argument('-m', '--min-area', type=int, default=500, help='Minimum blob area')
    parser.add_argument('-M', '--max-area', type=int, default=None, help='Maximum blob area')
    parser.add_argument('-s', '--sort-by', choices=['area', 'centroid_x', 'centroid_y', 'perimeter'], default='area', help='Sort blobs by attribute')
    parser.add_argument('--contour', action='store_true', help='Draw contour outline instead of bounding box')
    parser.add_argument('--connect', action='store_true', help='Draw lines connecting each blob to its closest neighbor')
    
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
        tracker.track_video(
            video_processor, args.color, args.min_area, args.max_area, 
            args.sort_by, args.contour, args.connect, args.output
        )
    finally:
        video_processor.release()

if __name__ == "__main__":
    main()