import csv
import os
from pipeline.temporal.hit_detector import detect_hit_frame
from pipeline.detection.shuttle_detector import ShuttleDetector


CSV_PATH = "frames.csv"
VIDEO_FOLDER = "fixed/"   # folder where videos are stored


def main():
    detector = ShuttleDetector()

    total_error = 0
    count = 0

    with open(CSV_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:

            serve_id = row["serve_id"]
            video_name = row["video"]
            start_frame = int(row["start_frame"])
            true_hit = int(row["hit_frame"])
            end_frame = int(row["end_frame"])
            label = row["label"]

            video_path = os.path.join(VIDEO_FOLDER, video_name)

            print(f"\nProcessing serve {serve_id} ({video_name})")



            predicted_hit = detect_hit_frame(
                detector,
                video_path,
                start_frame,
                end_frame
            )

            if predicted_hit is None:
                print("No hit detected.")
                continue

            error = abs(predicted_hit - true_hit)

            print("True hit frame:", true_hit)
            print("Predicted hit  :", predicted_hit)
            print("Frame error    :", error)
            print("Serve label    :", label)

            total_error += error
            count += 1

    if count > 0:
        print("\n=================================")
        print("Total evaluated serves:", count)
        print("Average frame error   :", total_error / count)
        print("=================================")
    else:
        print("No valid serves evaluated.")


if __name__ == "__main__":
    main()



