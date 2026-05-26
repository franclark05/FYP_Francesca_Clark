import cv2
import numpy as np

from pipeline.geometry.homography_utils import compute_homography, warp_point
from pipeline.detection.foot_keypoint_detector import FootKeypointDetector


SMOOTH_WINDOW = 3
LIFT_THRESHOLD = 6
HEEL_EXTRA_THRESHOLD = 2
MIN_CONSECUTIVE = 3

# Target screen size 
SCREEN_W = 1280
SCREEN_H = 720


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def consecutive(arr):
    c = 0
    for v in arr:
        c = c + 1 if v else 0
        if c >= MIN_CONSECUTIVE:
            return True
    return False


def resize_to_fit(img, max_w, max_h):
    #Resize image to fit within screen while keeping aspect ratio.
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale >= 1:
        return img  # no need to upscale
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def test_foot_lift(video_path):

    H, size = compute_homography()
    foot_detector = FootKeypointDetector()

    cap = cv2.VideoCapture(video_path)

    # Make windows resizable
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BirdsEye", cv2.WINDOW_NORMAL)

    left_toe, left_heel = [], []
    right_toe, right_heel = [], []

    fault = False

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        warped = cv2.warpPerspective(frame, H, size)

        feet = foot_detector.detect(frame)

        if feet:

            #Filter invalid detections
            feet = [f for f in feet if f["toe"] is not None and f["heel"] is not None]

            if len(feet) == 2:

                feet = sorted(
                    feet,
                    key=lambda f: warp_point(f["toe"], H)[0]
                )

                left, right = feet

                def process(foot, toe_arr, heel_arr, color):
                    toe = foot["toe"]
                    heel = foot["heel"]

                    toe_w = warp_point(toe, H)
                    heel_w = warp_point(heel, H)

                    toe_arr.append(toe_w[1])
                    heel_arr.append(heel_w[1])

                    cv2.circle(frame, toe, 5, color, -1)
                    cv2.circle(frame, heel, 5, (0, 255, 255), -1)

                    cv2.circle(warped, toe_w, 5, color, -1)
                    cv2.circle(warped, heel_w, 5, (0, 255, 255), -1)

                process(left, left_toe, left_heel, (0, 255, 0))
                process(right, right_toe, right_heel, (255, 0, 0))

        if len(left_toe) > 20 and not fault:

            lt = moving_average(np.array(left_toe), SMOOTH_WINDOW)
            lh = moving_average(np.array(left_heel), SMOOTH_WINDOW)
            rt = moving_average(np.array(right_toe), SMOOTH_WINDOW)
            rh = moving_average(np.array(right_heel), SMOOTH_WINDOW)

            lt_base = np.mean(lt[:5])
            lh_base = np.mean(lh[:5])
            rt_base = np.mean(rt[:5])
            rh_base = np.mean(rh[:5])

            left_lift = (lt_base - lt) > LIFT_THRESHOLD
            left_lift &= (lh_base - lh) > (LIFT_THRESHOLD + HEEL_EXTRA_THRESHOLD)

            right_lift = (rt_base - rt) > LIFT_THRESHOLD
            right_lift &= (rh_base - rh) > (LIFT_THRESHOLD + HEEL_EXTRA_THRESHOLD)

            if consecutive(left_lift) or consecutive(right_lift):
                fault = True

        text = "FOOT LIFT: YES" if fault else "FOOT LIFT: NO"

        cv2.putText(frame, text, (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 255) if fault else (0, 255, 0), 3)

        # Resize both views to fit screen
        display_frame = resize_to_fit(frame, SCREEN_W, SCREEN_H)
        display_warped = resize_to_fit(warped, SCREEN_W, SCREEN_H)

        cv2.imshow("Original", display_frame)
        cv2.imshow("BirdsEye", display_warped)

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_foot_lift("test_videos/P1_V4_S28.mp4")