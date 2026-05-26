import cv2
import os
import pandas as pd

# -------- CONFIG --------
CSV_PATH  = "selected_serves2.csv"
VIDEO_DIR = "fixed"
OUT_DIR   = "final_dataset_selected2"
WINDOW    = 20   # hit ± WINDOW
# ------------------------

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

grouped = df.groupby("video")

total_saved = 0

for video_name, serves in grouped:
    video_path = os.path.join(VIDEO_DIR, video_name)

    if not os.path.exists(video_path):
        print(f"Video not found: {video_name}")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_name}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing {video_name} ({len(serves)} serves)")

    for _, row in serves.iterrows():
        serve_id = row["serve_id"]
        hit = int(row["hit_frame"])
        label = row["label"]
        player = row["player"]

        start_f = max(hit - WINDOW, 0)
        end_f   = min(hit + WINDOW, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        for f in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret:
                break

            out_name = (
                f"{player}_{label}"
                f"_s{serve_id:02d}"
                f"_f{f:06d}.jpg"
            )

            out_path = os.path.join(OUT_DIR, out_name)
            cv2.imwrite(out_path, frame)
            total_saved += 1

    cap.release()

print(f"\nSaved {total_saved} images to {OUT_DIR}")
