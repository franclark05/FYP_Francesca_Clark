import cv2

def draw_court_lines(frame, court_lines, color=(0, 0, 255), thickness=2):
    for segment, abc in court_lines:
        (x1, y1), (x2, y2) = segment
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def draw_shuttle(frame, centroid, color=(0, 0, 255)):
    if centroid is not None:
        cx, cy = centroid
        cv2.circle(frame, (int(cx), int(cy)), 10, color, -1)
        cv2.putText(frame, "Shuttle", (int(cx)+10, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_feet(frame, keypoints, color=(0, 255, 0)):
    for foot in keypoints:
        for part in ["heel", "toe"]:
            pt = foot[part]
            if pt is None:
                continue
            x, y = pt
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
            cv2.putText(frame, part, (int(x)+5, int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def show_debug_frame(frame, window_name="debug"):
    resized = cv2.resize(frame, (1280, 720))
    cv2.imshow(window_name, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
