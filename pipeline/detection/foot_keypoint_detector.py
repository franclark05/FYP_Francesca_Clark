from ultralytics import YOLO
from config import FOOT_MODEL_PATH


class FootKeypointDetector:

    CONF_THRESHOLD = 0.5

    def __init__(self):
        self.model = YOLO(FOOT_MODEL_PATH)

    def detect(self, frame):

        results = self.model(frame, verbose=False)

        if len(results) == 0:
            return []

        result = results[0]

        if result.keypoints is None:
            return []

        if result.keypoints.data.shape[0] == 0:
            return []

        #Use data instead of xy → includes confidence
        kpts = result.keypoints.data[0]   # shape (4,3)

        if kpts.shape[0] != 4:
            return []

        # -----------------------------------------
        # 0 = left_toe
        # 1 = left_heel
        # 2 = right_toe
        # 3 = right_heel
        # -----------------------------------------

        def extract_point(kpt):
            x, y, conf = kpt

            if conf < self.CONF_THRESHOLD:
                return None  # ignore unreliable point

            return (int(x), int(y))

        left_toe   = extract_point(kpts[0])
        left_heel  = extract_point(kpts[1])
        right_toe  = extract_point(kpts[2])
        right_heel = extract_point(kpts[3])

        return [
            {
                "toe": right_toe,
                "heel": right_heel
            },
            {
                "toe": left_toe,
                "heel": left_heel
            }
        ]