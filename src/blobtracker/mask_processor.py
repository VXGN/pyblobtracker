import cv2
import numpy as np

class MaskProcessor:
    def __init__(self, color_ranges):
        self.color_ranges = color_ranges
    
    def create_mask(self, hsv_frame, color_name):
        masks = [
            cv2.inRange(hsv_frame, np.array(lo), np.array(hi))
            for lo, hi in self.color_ranges[color_name]
        ]
        return masks[0] if len(masks) == 1 else cv2.bitwise_or(masks[0], masks[1]) if len(masks) == 2 else cv2.bitwise_or.reduce(masks)
    
    def clean_mask(self, mask, kernel_size=5, morph_type='close', iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        match morph_type:
            case 'close':
                result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            case 'open':
                result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
            case _:
                result = mask
                
        return result
    
    def combine_masks(self, masks, operation='or'):
        if not masks:
            return None
        if len(masks) == 1:
            return masks[0]
            
        result = masks[0]
        for mask in masks[1:]:
            match operation:
                case 'or':
                    result = cv2.bitwise_or(result, mask)
                case 'and':
                    result = cv2.bitwise_and(result, mask)
        return result
    
    def apply_roi(self, mask, roi):
        x, y, w, h = roi
        result = np.zeros_like(mask)
        result[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
        return result
    
    def invert_mask(self, mask):
        return cv2.bitwise_not(mask)
