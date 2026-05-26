import cv2
import numpy as np
from pipeline.geometry.geometry_utils import line_to_abc
from config import ROI_POINTS


def apply_polygon_roi(image, roi_points):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [np.array(roi_points, dtype=np.int32)], 255)
    return cv2.bitwise_and(image, mask)


def detect_court_lines(frame):
    #detect blue court lines

    #convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Blue mask 
    lower_blue = np.array([85, 20, 20])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Smooth mask slightly
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    #edge detection
    edges = cv2.Canny(mask, 30, 120)

    #apply ROI
    roi_edges = apply_polygon_roi(edges, ROI_POINTS)

    #Hough line detection
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=80,
        maxLineGap=120
    )

    if lines is None:
        return None

    detected_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        #Ignore very short noisy lines
        if length > 100:
            segment = ((x1, y1), (x2, y2))
            abc = line_to_abc(x1, y1, x2, y2)
            detected_lines.append((segment, abc))

    return detected_lines