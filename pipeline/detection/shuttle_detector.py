import torch
from config import SHUTTLE_MODEL_PATH


class ShuttleDetector:

    def __init__(self):
        #Load YOLOv5 model
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=SHUTTLE_MODEL_PATH,
            force_reload=False
        )

        self.model.conf = 0.25  # confidence threshold
        self.model.iou = 0.45   # IoU threshold

    def detect(self, frame):

        results = self.model(frame)
        detections = results.xyxy[0]

        if len(detections) == 0:
            return None

        detections = detections.cpu().numpy()

        valid_detections = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det

            #filter confidence
            if conf < 0.5:
                continue

            #remove bottom-of-frame garbage
            # (this is your main issue)
            if y2 > frame.shape[0] * 0.9:
                continue

            
            w = x2 - x1
            h = y2 - y1

            if w > 100 or h > 100:
                continue

            valid_detections.append(det)

        if len(valid_detections) == 0:
            return None

        #pick best from filtered detections
        best = max(valid_detections, key=lambda x: x[4])

        x1, y1, x2, y2, conf, cls = best

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        return (cx, cy)