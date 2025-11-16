import cv2

class ContourDetector:
    def find_blobs(self, mask, min_area=500):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                blob_data = {
                    'contour': contour,
                    'centroid': self._calculate_centroid(contour),
                    'area': area,
                    'bbox': cv2.boundingRect(contour)
                }
                blobs.append(blob_data)
        
        return sorted(blobs, key=lambda x: x['area'], reverse=True)
    
    def _calculate_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)