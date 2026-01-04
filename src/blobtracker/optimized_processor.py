"""
Optimized processing module using Numba JIT compilation
"""

import numpy as np
from numba import jit, prange
import cv2


@jit(nopython=True, cache=True, fastmath=True)
def fast_distance(x1, y1, x2, y2):
    """Fast euclidean distance calculation"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


@jit(nopython=True, cache=True, parallel=True)
def fast_threshold_mask(hsv_image, h_low, s_low, v_low, h_high, s_high, v_high):
    """Fast HSV thresholding using Numba"""
    height, width = hsv_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for i in prange(height):
        for j in range(width):
            h, s, v = hsv_image[i, j]
            if h_low <= h <= h_high and s_low <= s <= s_high and v_low <= v <= v_high:
                mask[i, j] = 255
    
    return mask


@jit(nopython=True, cache=True, parallel=True)
def fast_threshold_mask_dual(hsv_image, h_low1, s_low1, v_low1, h_high1, s_high1, v_high1,
                              h_low2, s_low2, v_low2, h_high2, s_high2, v_high2):
    """Fast HSV thresholding for colors with two ranges (like red)"""
    height, width = hsv_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for i in prange(height):
        for j in range(width):
            h, s, v = hsv_image[i, j]
            in_range1 = h_low1 <= h <= h_high1 and s_low1 <= s <= s_high1 and v_low1 <= v <= v_high1
            in_range2 = h_low2 <= h <= h_high2 and s_low2 <= s <= s_high2 and v_low2 <= v <= v_high2
            if in_range1 or in_range2:
                mask[i, j] = 255
    
    return mask


@jit(nopython=True, cache=True, fastmath=True)
def compute_centroid(moments_m00, moments_m10, moments_m01):
    """Fast centroid computation"""
    if moments_m00 == 0:
        return -1, -1
    return int(moments_m10 / moments_m00), int(moments_m01 / moments_m00)


@jit(nopython=True, cache=True, fastmath=True)
def compute_circularity(area, perimeter):
    """Fast circularity computation"""
    if perimeter <= 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


class OptimizedMaskProcessor:
    """Optimized mask processor using Numba JIT"""
    
    def __init__(self, color_ranges):
        self.color_ranges = color_ranges
        self._kernel = np.ones((5, 5), np.uint8)
    
    def create_mask_fast(self, hsv_frame, color_name):
        """Create mask using Numba-optimized thresholding"""
        ranges = self.color_ranges[color_name]
        
        if len(ranges) == 1:
            lo, hi = ranges[0]
            return fast_threshold_mask(
                hsv_frame,
                lo[0], lo[1], lo[2],
                hi[0], hi[1], hi[2]
            )
        elif len(ranges) == 2:
            lo1, hi1 = ranges[0]
            lo2, hi2 = ranges[1]
            return fast_threshold_mask_dual(
                hsv_frame,
                lo1[0], lo1[1], lo1[2],
                hi1[0], hi1[1], hi1[2],
                lo2[0], lo2[1], lo2[2],
                hi2[0], hi2[1], hi2[2]
            )
        else:
            # Fallback for more than 2 ranges
            masks = [
                cv2.inRange(hsv_frame, np.array(lo), np.array(hi))
                for lo, hi in ranges
            ]
            result = masks[0]
            for m in masks[1:]:
                result = cv2.bitwise_or(result, m)
            return result
    
    def clean_mask(self, mask):
        """Clean mask using morphological operations"""
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)


class OptimizedContourDetector:
    """Optimized contour detector"""
    
    def find_blobs_fast(self, mask, min_area=500, max_area=None, sort_by='area'):
        """Find blobs with optimized processing"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or (max_area and area > max_area):
                continue
            
            M = cv2.moments(c)
            cx, cy = compute_centroid(M["m00"], M["m10"], M["m01"])
            centroid = (cx, cy) if cx >= 0 else None
            
            perimeter = cv2.arcLength(c, True)
            x, y, w, h = cv2.boundingRect(c)
            
            blobs.append({
                'contour': c,
                'centroid': centroid,
                'area': area,
                'bbox': (x, y, w, h),
                'perimeter': perimeter,
                'circularity': compute_circularity(area, perimeter),
                'aspect_ratio': w / h if h > 0 else 0
            })
        
        # Sort blobs
        if sort_by == 'area':
            blobs.sort(key=lambda b: b['area'], reverse=True)
        elif sort_by == 'centroid_x':
            blobs.sort(key=lambda b: b['centroid'][0] if b['centroid'] else float('inf'))
        elif sort_by == 'centroid_y':
            blobs.sort(key=lambda b: b['centroid'][1] if b['centroid'] else float('inf'))
        elif sort_by == 'perimeter':
            blobs.sort(key=lambda b: b['perimeter'], reverse=True)
        
        return blobs
