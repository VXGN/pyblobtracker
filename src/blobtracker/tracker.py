import cv2

class BlobTracker:
    def __init__(self, mask_processor, contour_detector, visualizer):
        self.mask_processor = mask_processor
        self.contour_detector = contour_detector
        self.visualizer = visualizer
    
    def process_frame(self, frame, color_name, min_area):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.mask_processor.create_mask(hsv, color_name)
        mask = self.mask_processor.clean_mask(mask)
        blobs = self.contour_detector.find_blobs(mask, min_area)
        self.visualizer.draw_blobs(frame, blobs)
        return frame, blobs
    
    def track_video(self, video_processor, color_name, min_area=500, output_path=None):
        writer = video_processor.create_writer(output_path) if output_path else None
        
        frame_count = 0
        total_detections = 0
        
        print(f"Tracking {color_name.upper()} blobs...")
        
        while True:
            ret, frame = video_processor.read_frame()
            if not ret:
                break
            
            frame_count += 1
            processed_frame, blobs = self.process_frame(frame, color_name, min_area)
            total_detections += len(blobs)
            
            if writer:
                writer.write(processed_frame)
            
            cv2.imshow('Tracking', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}/{video_processor.total_frames}, Blobs: {len(blobs)}")
        
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nCompleted: {frame_count} frames, {total_detections} total detections")