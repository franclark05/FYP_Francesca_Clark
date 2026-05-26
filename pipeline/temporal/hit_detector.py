import numpy as np
import cv2

SMOOTHING_WINDOW = 5


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def detect_hit_frame(detector, video_path, start_frame, end_frame):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video")
        return None

    cx_list = []
    cy_list = []
    frame_indices = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx > end_frame:
            break

        if frame_idx >= start_frame:

            centroid = detector.detect(frame)

            if centroid is not None:
                cx, cy = centroid

                cx_list.append(cx)
                cy_list.append(cy)
                frame_indices.append(frame_idx)

        frame_idx += 1

    cap.release()

    if len(cx_list) < 10:
        return None

    cx_array = np.array(cx_list)
    cy_array = np.array(cy_list)

    cx_smooth = moving_average(cx_array, SMOOTHING_WINDOW)
    cy_smooth = moving_average(cy_array, SMOOTHING_WINDOW)

    valid_frames = frame_indices[SMOOTHING_WINDOW - 1:]

    vx = np.diff(cx_smooth)
    vy = np.diff(cy_smooth)

    speed = np.sqrt(vx**2 + vy**2)

    # shuttle is almost stationary before hit
    baseline = np.mean(speed[:5])

    threshold = baseline + 3 * np.std(speed[:10])

    candidates = np.where(speed > threshold)[0]

    if len(candidates) == 0:
        hit_index = np.argmax(speed)
    else:
        hit_index = candidates[0]

    predicted_hit = valid_frames[hit_index]

    return predicted_hit