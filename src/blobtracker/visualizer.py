import cv2
import numpy as np

class Visualizer:
    def __init__(self, color=(255, 255, 255), thickness=2):
        self.color = color
        self.thickness = thickness
    
    def draw_blobs(self, frame, blobs, draw_contour=False, draw_connections=False):
        # Draw contours if enabled
        if draw_contour:
            for blob in blobs:
                cv2.drawContours(frame, [blob['contour']], -1, self.color, self.thickness)
        
        # Draw bounding boxes and labels
        for i, blob in enumerate(blobs):
            if not draw_contour:
                self._draw_bbox(frame, blob['bbox'])
            if blob['centroid']:
                self._draw_label(frame, blob, i)
        
        # Draw connection lines if enabled
        if draw_connections and len(blobs) > 1:
            self._draw_connections(frame, blobs)
    
    def _draw_bbox(self, frame, bbox):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, self.thickness)
    
    def _draw_label(self, frame, blob, index):
        cx, cy = blob['centroid']
        text = f"#{index + 1}: {int(blob['area'])}"
        cv2.putText(frame, text, (cx + 10, cy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, self.thickness)
    
    def _draw_connections(self, frame, blobs):
        """Draw lines connecting each blob to its closest neighbor."""
        centroids = [b['centroid'] for b in blobs if b['centroid']]
        if len(centroids) < 2:
            return
        
        connected = set()
        
        for i, c1 in enumerate(centroids):
            """assign closest centroid"""
            min_dist = float('inf')
            closest_idx = -1
            
            for j, c2 in enumerate(centroids):
                """avoid self-connection"""
                if i == j:
                    continue
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = j
            
            if closest_idx != -1:
                pair = tuple(sorted([i, closest_idx]))
                if pair not in connected:
                    connected.add(pair)
                    c2 = centroids[closest_idx]
                    cv2.line(frame, c1, c2, (255, 255, 255), self.thickness)
                    mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                    cv2.putText(frame, f"{int(min_dist)}px", mid, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)