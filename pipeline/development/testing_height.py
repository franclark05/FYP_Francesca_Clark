import cv2

from pipeline.detection.shuttle_detector import ShuttleDetector
from pipeline.temporal.hit_detector import detect_hit_frame
from pipeline.evaluation.shuttle_height_checker import check_shuttle_height
from pipeline.utils.video_utils import get_frame


VIDEO_PATH = "fixed/P4_V2.mp4"

START_FRAME = 15974
END_FRAME = 16167


def main():

    detector = ShuttleDetector()

    #detect hit frame
    hit_frame = detect_hit_frame(detector, VIDEO_PATH, START_FRAME, END_FRAME)

    print("Hit frame:", hit_frame)

    #Extract frame
    frame = get_frame(VIDEO_PATH, hit_frame)

    #Detect shuttle
    centroid = detector.detect(frame)

    print("Shuttle centroid:", centroid)

    #Check height
    fault = check_shuttle_height(centroid)

    if fault:
        print("FAULT: Shuttle above service height")
    else:
        print("Serve height is legal")

    #Visual debug
    if centroid is not None:

        cx, cy = centroid

        cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)

        cv2.imshow("Hit Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()