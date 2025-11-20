import cv2

class ContourDetector:
    def find_blobs(self, mask, min_area=500):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                centroid = None
            else:
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            blobs.append({
                'contour': c,
                'centroid': centroid,
                'area': area,
                'bbox': cv2.boundingRect(c)
            })

        return sorted(blobs, key=lambda b: b['area'], reverse=True)
