import cv2
import numpy as np

class ContourDetector:
    def find_blobs(self, mask, min_area=500, max_area=None, sort_by='area'):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or (max_area and area > max_area):
                continue
                
            M = cv2.moments(c)
            if M["m00"] == 0:
                centroid = None
            else:
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            perimeter = cv2.arcLength(c, True)
            bbox = cv2.boundingRect(c)
            x, y, w, h = bbox
            
            blobs.append({
                'contour': c,
                'centroid': centroid,
                'area': area,
                'bbox': bbox,
                'perimeter': perimeter,
                'circularity': 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,
                'aspect_ratio': w / h if h > 0 else 0
            })
        
        if sort_by == 'area':
            return sorted(blobs, key=lambda b: b['area'], reverse=True)
        elif sort_by == 'centroid_x':
            return sorted(blobs, key=lambda b: b['centroid'][0] if b['centroid'] else float('inf'))
        elif sort_by == 'centroid_y':
            return sorted(blobs, key=lambda b: b['centroid'][1] if b['centroid'] else float('inf'))
        elif sort_by == 'perimeter':
            return sorted(blobs, key=lambda b: b['perimeter'], reverse=True)
        return blobs
