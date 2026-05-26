import cv2

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}:", (x, y))


def calibrate(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        print("Could not read video")
        return

    cv2.imshow("Click 4 court corners", frame)
    cv2.setMouseCallback("Click 4 court corners", click)

    while True:
        temp = frame.copy()

        for p in points:
            cv2.circle(temp, p, 5, (0, 0, 255), -1)

        cv2.imshow("Click 4 court corners", temp)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nCopy these into config.py:")
    print(points)


if __name__ == "__main__":
    calibrate("test_videos/P1_V4_S12.mp4")