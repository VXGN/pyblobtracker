import cv2
import numpy as np

class MaskProcessor:
    def __init__(self, color_ranges):
        self.color_ranges = color_ranges
    
    def create_mask(self, hsv_frame, color_name):
        ranges = self.color_ranges[color_name]
        masks = [cv2.inRange(hsv_frame, np.array(r[0]), np.array(r[1])) for r in ranges]
        return cv2.bitwise_or(*masks) if len(masks) > 1 else masks[0]
    
    def clean_mask(self, mask, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)