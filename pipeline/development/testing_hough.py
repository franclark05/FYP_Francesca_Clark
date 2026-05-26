import cv2
from pipeline.vision.court_line_detector import detect_court_lines
from config import ROI_POINTS
import numpy as np
import pickle

cap = cv2.VideoCapture("fixed/P2_V3.mp4")
ret, frame = cap.read()

# Draw ROI polygon for verification
roi_visual = frame.copy()
cv2.polylines(
    roi_visual,
    [np.array(ROI_POINTS, dtype=np.int32)],
    True,
    (0, 0, 255),
    3
)

lines = detect_court_lines(frame)

if lines:
    for segment, _ in lines:
        (x1, y1), (x2, y2) = segment
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

with open("saved_lines.pkl", "wb") as f:
    pickle.dump(lines, f)
print("Lines saved")

cv2.imshow("Lines", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()