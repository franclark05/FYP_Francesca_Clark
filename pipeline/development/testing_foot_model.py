import cv2
import numpy as np
import pickle

from pipeline.detection.foot_keypoint_detector import FootKeypointDetector
from pipeline.temporal.foot_contact_detector import check_line_fault
from pipeline.geometry.geometry_utils import point_line_distance


VIDEO_PATH = "fixed/P2_V3.mp4"
HIT_FRAME_INDEX = 12000
LINES_FILE = "saved_lines_P2_V3.pkl"


#select service line
def select_service_line(lines):

    if not lines:
        return None

    best_line = None
    max_y_avg = 0

    for segment, abc in lines:
        (x1, y1), (x2, y2) = segment
        y_avg = (y1 + y2) / 2

        if y_avg > max_y_avg:
            max_y_avg = y_avg
            best_line = (segment, abc)

    return best_line


# Draw infinite line from ax+by+c=0
def draw_line_from_abc(frame, abc, color=(255, 0, 0)):
    a, b, c = abc
    h, w = frame.shape[:2]

    if abs(b) > 1e-6:
        x1 = 0
        y1 = int((-c - a * x1) / b)

        x2 = w
        y2 = int((-c - a * x2) / b)
    else:
        x1 = int(-c / a)
        x2 = x1
        y1 = 0
        y2 = h

    cv2.line(frame, (x1, y1), (x2, y2), color, 3)


def main():

    #load stable lines
    with open(LINES_FILE, "rb") as f:
        lines = pickle.load(f)

    print(f"{len(lines)} stable lines loaded.")

    selected = select_service_line(lines)

    if selected is None:
        print("No service line selected.")
        return

    service_line_segment, service_line_abc = selected

    print("Service line selected:", service_line_segment)

    #load hit frame
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video")
        return

    frame_count = 0
    hit_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count == HIT_FRAME_INDEX:
            hit_frame = frame
            break

        frame_count += 1

    cap.release()

    if hit_frame is None:
        print("Hit frame not found.")
        return

    print("Hit frame loaded.")

    #detect foot keypoints
    foot_detector = FootKeypointDetector()
    foot_kpts = foot_detector.detect(hit_frame)

    if not foot_kpts:
        print("No foot keypoints detected.")
        return

    print("Foot keypoints:", foot_kpts)

    #compute distances
    print("\n--- DISTANCES ---")

    for i, foot in enumerate(foot_kpts):

        print(f"Foot {i}")

        heel = foot["heel"]
        toe = foot["toe"]

        if heel is not None:
            heel_dist = point_line_distance(heel, service_line_abc)
            print("Heel distance:", heel_dist)
        else:
            print("Heel: not visible")

        if toe is not None:
            toe_dist = point_line_distance(toe, service_line_abc)
            print("Toe distance :", toe_dist)
        else:
            print("Toe: not visible")

        print("-------------")

    #fault decision
    fault = check_line_fault(foot_kpts, service_line_abc)

    print("\nFOOT LINE FAULT:", fault)

    #visualisation

    # Draw ALL saved lines (yellow)
    for segment, _ in lines:
        (x1, y1), (x2, y2) = segment
        cv2.line(hit_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw feet
    for foot in foot_kpts:
        cv2.circle(hit_frame, foot["heel"], 3, (0, 0, 255), -1)
        cv2.circle(hit_frame, foot["toe"], 3, (0, 255, 0), -1)

    cv2.imshow("Foot Line Detection Debug", hit_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()