import cv2

class Visualizer:
    def __init__(self, color=(192, 192, 192), thickness=2):
        self.color = color
        self.thickness = thickness
    
    def draw_blobs(self, frame, blobs):
        for i, blob in enumerate(blobs):
            self._draw_bbox(frame, blob['bbox'])
            if blob['centroid']:
                self._draw_label(frame, blob, i)
    
    def _draw_bbox(self, frame, bbox):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, self.thickness)
    
    def _draw_label(self, frame, blob, index):
        cx, cy = blob['centroid']
        text = f"#{index + 1}: {int(blob['area'])}"
        cv2.putText(frame, text, (cx + 10, cy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, self.thickness)