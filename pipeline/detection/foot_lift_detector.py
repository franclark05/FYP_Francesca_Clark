import cv2
import numpy as np

from pipeline.geometry.homography_utils import compute_homography, warp_point


SMOOTH_WINDOW = 3
LIFT_THRESHOLD = 3
HEEL_EXTRA_THRESHOLD = 2
MIN_CONSECUTIVE = 3

#moving average smoothing
def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


#check if array has min_consecutive True values in a row
def consecutive(arr):
    c = 0
    for v in arr:
        c = c + 1 if v else 0
        if c >= MIN_CONSECUTIVE:
            return True
    return False

#main foot lift detection
def detect_foot_lift(detector, video_path, start_frame, end_frame):

    #compute homography matrix for warping points to top-down view
    H, _ = compute_homography()

    cap = cv2.VideoCapture(video_path)

    left_toe, left_heel = [], []
    right_toe, right_heel = [], []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < start_frame:
            frame_idx += 1
            continue

        if frame_idx > end_frame:
            break

        feet = detector.detect(frame)

        if feet:
            #keep only detections with both toe and heel
            feet = [f for f in feet if f["toe"] and f["heel"]]

            if len(feet) == 2:

                # sort using warped x 
                feet = sorted(feet, key=lambda f: warp_point(f["toe"], H)[0])
                left, right = feet

                def process(foot, toe_arr, heel_arr):
                    toe_w = warp_point(foot["toe"], H)
                    heel_w = warp_point(foot["heel"], H)

                    toe_arr.append(toe_w[1])
                    heel_arr.append(heel_w[1])

                process(left, left_toe, left_heel)
                process(right, right_toe, right_heel)

        frame_idx += 1

    cap.release()

    # not enough data
    if len(left_toe) < 10:
        return False
    
    # smoothing
    lt = moving_average(np.array(left_toe), SMOOTH_WINDOW)
    lh = moving_average(np.array(left_heel), SMOOTH_WINDOW)
    rt = moving_average(np.array(right_toe), SMOOTH_WINDOW)
    rh = moving_average(np.array(right_heel), SMOOTH_WINDOW)

    # baseline
    lt_base = np.mean(lt[:8])
    lh_base = np.mean(lh[:8])
    rt_base = np.mean(rt[:8])
    rh_base = np.mean(rh[:8])

    # detection
    left_lift = (lt_base - lt) > LIFT_THRESHOLD
    left_lift &= (lh_base - lh) > (LIFT_THRESHOLD + HEEL_EXTRA_THRESHOLD)

    right_lift = (rt_base - rt) > LIFT_THRESHOLD
    right_lift &= (rh_base - rh) > (LIFT_THRESHOLD + HEEL_EXTRA_THRESHOLD)

    return consecutive(left_lift) or consecutive(right_lift)