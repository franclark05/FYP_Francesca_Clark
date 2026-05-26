import cv2
import numpy as np

from detection.shuttle_detector import ShuttleDetector
from detection.foot_keypoint_detector import FootKeypointDetector

from vision.court_line_detector import detect_court_lines
from temporal.hit_detector import detect_hit_frame
from temporal.foot_contact_detector import check_line_fault
from detection.foot_lift_detector import detect_foot_lift

from config import SERVICE_HEIGHT_PIXELS
from utils.video_utils import get_all_frames
from utils.visualisation import draw_court_lines, draw_feet, draw_shuttle


DEBUG_WINDOW = 8


def run_pipeline(video_path):
    print(f"Processing: {video_path}")


    #Load models
    print("Loading models...")
    shuttle_detector = ShuttleDetector()
    foot_detector = FootKeypointDetector()

    frames = get_all_frames(video_path)

    if len(frames) == 0:
        print("ERROR: No frames loaded")
        return "ERROR"

    #Court Lines
    print("Detecting court lines...")
    court_lines = detect_court_lines(frames[0])

    #INITIAL Hit frame
    print("Detecting hit frame...")
    raw_hit_idx = detect_hit_frame(
        shuttle_detector,
        video_path,
        0,
        len(frames) - 1
    )

    if raw_hit_idx is None:
        print("ERROR: No hit frame detected")
        return "ERROR"

    print(f"Raw hit frame: {raw_hit_idx}")

    #NEW hit frame
    HIT_WINDOW = 3
    start = max(0, raw_hit_idx - HIT_WINDOW)
    end = min(len(frames), raw_hit_idx + HIT_WINDOW)

    positions = []
    indices = []

    for i in range(start, end):
        centroid = shuttle_detector.detect(frames[i])
        if centroid is not None:
            positions.append(centroid)
            indices.append(i)

    refined_hit_idx = raw_hit_idx

    if len(positions) >= 3:
        positions = np.array(positions)

        #Velocity
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        #Acceleration (change in velocity)
        accel = np.diff(velocities)

        if len(accel) > 0:
            best_idx = np.argmax(accel)
            refined_hit_idx = indices[best_idx + 1]

    print(f"Refined hit frame: {refined_hit_idx}")

    #Use refined index
    adjusted_idx = refined_hit_idx

    #Shuttle at hit
    print("Detecting shuttle...")
    centroid = shuttle_detector.detect(frames[adjusted_idx])

    height_fault = False
    if centroid is not None:
        _, cy = centroid
        if cy < SERVICE_HEIGHT_PIXELS:
            height_fault = True
            print("HEIGHT FAULT DETECTED")
    else:
        print("WARNING: Shuttle not detected at hit frame")

    #feet at hit
    print("Detecting feet...")
    keypoints_hit = foot_detector.detect(frames[adjusted_idx])

    line_fault = False
    if keypoints_hit is not None and court_lines is not None:
        line_fault = check_line_fault(keypoints_hit, court_lines)
        if line_fault:
            print("LINE FAULT detected")
    else:
        print("WARNING: Could not check line fault")

    #foot lift
    print("Detecting foot lift...")
    foot_lift = detect_foot_lift(
        foot_detector,
        video_path,
        max(0, adjusted_idx - 40),
        adjusted_idx
    )

    if foot_lift:
        print("FOOT LIFT detected")

    #visual debug
    DEBUG_WINDOW = 8
    start = max(0, adjusted_idx - DEBUG_WINDOW)
    end = min(len(frames), adjusted_idx + DEBUG_WINDOW)

    for i in range(start, end):

        temp = frames[i].copy()

        #Feet detection
        keypoints = foot_detector.detect(temp)

        #Court layer
        court_layer = np.zeros_like(temp)
        if court_lines is not None:
            draw_court_lines(court_layer, court_lines)

        #Player mask
        player_mask = np.zeros(temp.shape[:2], dtype=np.uint8)

        if keypoints is not None:
            for foot in keypoints:
                for part in ["heel", "toe"]:
                    pt = foot[part]
                    if pt is None:
                        continue

                    x, y = map(int, pt)

                    cv2.circle(player_mask, (x, y), 40, 255, -1)

                    cv2.rectangle(player_mask,
                                  (x - 40, y - 180),
                                  (x + 40, y + 20),
                                  255, -1)

        inv_mask = cv2.bitwise_not(player_mask)
        court_occluded = cv2.bitwise_and(court_layer, court_layer, mask=inv_mask)

        temp = cv2.add(temp, court_occluded)

        #Shuttle detection
        shuttle = shuttle_detector.detect(temp)

        #Draw
        if shuttle is not None:
            draw_shuttle(temp, shuttle)

        if keypoints is not None:
            draw_feet(temp, keypoints)

        #Hit frame marker
        if i == adjusted_idx:
            cv2.putText(temp, "HIT FRAME", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.putText(temp, f"Frame: {i}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        #Fault display
        if i == adjusted_idx:

            y = 170

            if height_fault:
                cv2.putText(temp, "HEIGHT FAULT", (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                y += 40

            if line_fault:
                cv2.putText(temp, "LINE FAULT", (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                y += 40

            if foot_lift:
                cv2.putText(temp, "FOOT LIFT", (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("DEBUG", temp)

        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()

    #FINAL decision
    if height_fault:
        decision = "HEIGHT_FAULT"
    elif line_fault:
        decision = "LINE_FAULT"
    elif foot_lift:
        decision = "FOOT_LIFT"
    else:
        decision = "NO_FAULT"

    print(f"\nFINAL DECISION: {decision}")

    return decision


if __name__ == "__main__":
    VIDEO_PATH = "test_videos/P8_V4_S28.mp4"
    run_pipeline(VIDEO_PATH)