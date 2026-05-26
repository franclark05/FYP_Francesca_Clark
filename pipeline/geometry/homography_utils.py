import cv2
import numpy as np
from config import COURT_POINTS


def compute_homography():

    src = np.array(COURT_POINTS, dtype=np.float32)

    width = 600
    height = 1200

    dst = np.array([
        [0, height],       # near-left
        [width, height],   # near-right
        [width, 0],        # far-right
        [0, 0]             # far-left
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)

    return H, (width, height)


def warp_point(pt, H):
    x, y = pt
    vec = np.array([x, y, 1.0])
    warped = H @ vec
    warped /= warped[2]
    return int(warped[0]), int(warped[1])