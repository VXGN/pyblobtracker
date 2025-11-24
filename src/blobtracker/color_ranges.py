import cv2
import numpy as np

COLOR_MAP = {0: 'red', 30: 'orange', 60: 'yellow', 120: 'green', 180: 'cyan', 240: 'blue', 300: 'purple'}

def detect_color(bgr):
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h = hsv[0] * 2 
    return min(COLOR_MAP.items(), key=lambda x: abs(x[0] - h))[1]
