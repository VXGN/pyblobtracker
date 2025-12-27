import cv2
import numpy as np

COLOR_MAP = {0: 'red', 30: 'orange', 60: 'yellow', 120: 'green', 180: 'cyan', 240: 'blue', 300: 'purple'}

COLOR_RANGES = {
    'red': [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
    'orange': [([10, 100, 100], [25, 255, 255])],
    'yellow': [([25, 100, 100], [35, 255, 255])],
    'green': [([35, 100, 100], [85, 255, 255])],
    'cyan': [([85, 100, 100], [95, 255, 255])],
    'blue': [([95, 100, 100], [125, 255, 255])],
    'purple': [([125, 100, 100], [155, 255, 255])],
    'white': [([0, 0, 200], [180, 30, 255])],
    'black': [([0, 0, 0], [180, 255, 30])],
}

def detect_color(bgr):
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h = hsv[0] * 2 
    return min(COLOR_MAP.items(), key=lambda x: abs(x[0] - h))[1]

