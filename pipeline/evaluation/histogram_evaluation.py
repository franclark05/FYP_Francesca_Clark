import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pipeline.detection.shuttle_detector import ShuttleDetector
from pipeline.temporal.hit_detector import detect_hit_frame
from pipeline.utils.video_utils import get_all_frames


#hit refinement
def get_refined_hit(video_path, detector):

    frames = get_all_frames(video_path)

    if len(frames) == 0:
        return None

    raw_hit_idx = detect_hit_frame(
        detector,
        video_path,
        0,
        len(frames) - 1
    )

    if raw_hit_idx is None:
        return None

    #refinement
    HIT_WINDOW = 10
    start = max(0, raw_hit_idx - HIT_WINDOW)
    end = min(len(frames), raw_hit_idx + HIT_WINDOW)

    positions = []
    indices = []

    for i in range(start, end):
        centroid = detector.detect(frames[i])
        if centroid is not None:
            positions.append(centroid)
            indices.append(i)

    refined_hit_idx = raw_hit_idx

    if len(positions) >= 3:
        positions = np.array(positions)

        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        accel = np.diff(velocities)

        if len(accel) > 0:
            best_idx = np.argmax(accel)
            refined_hit_idx = indices[best_idx + 1]

    return refined_hit_idx


#config
CSV_PATH = "histogram_hit.csv"
VIDEO_FOLDER = "test_videos/"

ERROR_THRESHOLD = 5


df = pd.read_csv(CSV_PATH)

detector = ShuttleDetector()


#run evaluation
errors = []

for idx, row in df.iterrows():

    video_name = row["video"]
    gt_hit = int(row["hit_frame"])

    video_path = VIDEO_FOLDER + video_name

    print(f"\nProcessing {video_name}...")

    pred_hit = get_refined_hit(video_path, detector)

    if pred_hit is None:
        print("Skipping (no detection)")
        continue

    error = pred_hit - gt_hit
    errors.append(error)

    print(f"GT: {gt_hit}, Pred: {pred_hit}, Error: {error}")


errors = np.array(errors)


#summary metrics
print("\nSUMMARY:")
print(f"Total samples: {len(errors)}")
print(f"Mean error: {np.mean(errors):.2f}")
print(f"Std error: {np.std(errors):.2f}")
print(f"Within ±3 frames: {np.sum(np.abs(errors) <= 3)}")
print(f"Within ±5 frames: {np.sum(np.abs(errors) <= 5)}")


#filter only big errors
filtered_errors = errors[np.abs(errors) >= ERROR_THRESHOLD]


#custom buckets for plots
bins = [5, 10, 20, 30, 50, 100, 200, 300]  

early_counts = [0] * (len(bins) - 1)
late_counts = [0] * (len(bins) - 1)

for e in filtered_errors:

    abs_e = abs(e)

    for i in range(len(bins) - 1):
        if bins[i] <= abs_e < bins[i + 1]:

            if e < 0:
                early_counts[i] += 1
            else:
                late_counts[i] += 1

            break


#plot
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(x - width/2, early_counts, width, label="Early (Pred < GT)")
plt.bar(x + width/2, late_counts, width, label="Late (Pred > GT)")

plt.xticks(x, labels)
plt.xlabel("Frame Error Range")
plt.ylabel("Number of Videos")
plt.title("Hit Frame Error Distribution (|error| ≥ 5)")
plt.legend()

plt.savefig("hit_error_distribution.png", dpi=300, bbox_inches='tight')

print("Plot saved as hit_error_distribution.png")

plt.show()