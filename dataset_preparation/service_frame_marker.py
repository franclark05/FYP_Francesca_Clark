import cv2
import os
import csv

# -------- CONFIG --------
VIDEO_PATH = r"test_videos\P8_V4_S30.mp4"   # change per video
CSV_FILE = r"balala.csv"
END_OFFSET = 50                  # frames after HIT
# ------------------------

LABEL_MAP = {
    "V1": "valid",
    "V2": "too_high",
    "V3": "foot_on_line",
    "V4": "foot_lifted"
}

def get_label_from_video(video_name):
    for key, label in LABEL_MAP.items():
        if key in video_name:
            return label
    return "unlabeled"

def main(video_path, csv_file):

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.basename(video_path)

    label = get_label_from_video(video_name)

    print(f"\nVideo: {video_name}")
    print(f"Total frames: {frame_count} | FPS: {fps:.2f}")
    print(f"Auto label: {label}")

    current_frame = 0
    serve_id = 1
    records = []

    start_frame = None
    hit_frame = None

    print("\nKeyboard controls:")
    print(" d / → : next frame")
    print(" a / ← : previous frame")
    print(" n     : skip -50 frames")
    print(" m     : skip +50 frames")
    print(" b     : skip +240 frames")
    print(" s     : mark START of serve")
    print(" h     : mark HIT frame")
    print(" e     : save serve (END = HIT + 50)")
    print(" q     : quit and save CSV\n")

    window = "Serve Frame Marker"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        display = frame.copy()

        cv2.putText(display, f"Frame: {current_frame}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display, f"Serve ID: {serve_id}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if start_frame is not None:
            cv2.putText(display, f"START @ {start_frame}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        if hit_frame is not None:
            cv2.putText(display, f"HIT @ {hit_frame}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(0) & 0xFF

        # -------- NAVIGATION --------
        if key == ord('q'):
            print("Quitting...")
            break

        elif key == ord('d') or key == 83:
            current_frame = min(current_frame + 1, frame_count - 1)

        elif key == ord('a') or key == 81:
            current_frame = max(current_frame - 1, 0)

        elif key == ord('m'):
            current_frame = min(current_frame + 50, frame_count - 1)

        elif key == ord('n'):
            current_frame = max(current_frame - 50, 0)

        elif key == ord('b'):
            current_frame = min(current_frame + 240, frame_count - 1)

        # -------- MARKING --------


        elif key == ord('h'):
            hit_frame = current_frame
            print(f"Serve {serve_id} HIT = {hit_frame}")

            print(f"Label = {label}")

            records.append([
                serve_id,
                video_name,
                start_frame,
                hit_frame,
                label
            ])

            serve_id += 1
            start_frame = None
            hit_frame = None

        # ignore all other keys

    cap.release()
    cv2.destroyAllWindows()

    # -------- SAVE CSV --------
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "serve_id",
                "video",
                "start_frame",
                "hit_frame",
                "end_frame",
                "label"
            ])
        writer.writerows(records)

    print(f"\nSaved {len(records)} serves to {csv_file}")

if __name__ == "__main__":
    main(VIDEO_PATH, CSV_FILE)
