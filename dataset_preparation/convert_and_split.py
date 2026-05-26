import os
import json
import shutil
import random
from collections import defaultdict
from pathlib import Path

# --------------------------
# CONFIG
# --------------------------

JSON_PATH = "result.json"
IMAGE_ROOT = "final_dataset_selected"
OUTPUT_ROOT = "dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

#Load COCO

with open(JSON_PATH) as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

#Clean paths

def clean_filename(name):
    name = name.replace("%5C", "/")
    name = name.replace("\\", "/")
    return os.path.basename(name)

for img in images:
    img["file_name"] = clean_filename(img["file_name"])

#Group image paths

serve_groups = defaultdict(list)

for img in images:
    filename = img["file_name"]
    parts = filename.replace(".jpg", "").split("_")

    player = parts[0]
    serve_number = parts[-2]        
    serve_type = "_".join(parts[1:-2]) 

    full_serve_id = f"{player}_{serve_type}_{serve_number}"
    serve_groups[full_serve_id].append(img)

#Stratify serves by type

serves_by_type = defaultdict(list)

for full_serve_id in serve_groups.keys():
    # format: P5_valid_s28
    serve_type = full_serve_id.split("_")[1]
    serves_by_type[serve_type].append(full_serve_id)

#Split each type separately

splits = {
    "train": [],
    "val": [],
    "test": []
}

for serve_type, serve_list in serves_by_type.items():

    random.shuffle(serve_list)

    n_total = len(serve_list)
    train_cut = int(n_total * TRAIN_RATIO)
    val_cut = int(n_total * (TRAIN_RATIO + VAL_RATIO))

    splits["train"].extend(serve_list[:train_cut])
    splits["val"].extend(serve_list[train_cut:val_cut])
    splits["test"].extend(serve_list[val_cut:])

#Shuffle final splits
for split in splits:
    random.shuffle(splits[split])

print("Split summary:")
for split in splits:
    print(f"{split}: {len(splits[split])} serves")


for model_type in ["shuttle", "pose"]:
    for split in splits.keys():
        Path(f"{OUTPUT_ROOT}_{model_type}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_ROOT}_{model_type}/labels/{split}").mkdir(parents=True, exist_ok=True)

ann_map = defaultdict(list)
for ann in annotations:
    ann_map[ann["image_id"]].append(ann)


for split_name, serve_list in splits.items():

    for full_serve_id in serve_list:

        for img in serve_groups[full_serve_id]:

            img_id = img["id"]
            filename = img["file_name"]
            width = img["width"]
            height = img["height"]

            src_path = os.path.join(IMAGE_ROOT, filename)

            #Copy image to both datasets
            for model_type in ["shuttle", "pose"]:
                dst_path = f"{OUTPUT_ROOT}_{model_type}/images/{split_name}/{filename}"
                shutil.copy(src_path, dst_path)

            shuttle_lines = []
            pose_lines = []

            for ann in ann_map[img_id]:

                #SHUTTLE
                if ann["category_id"] == 0:

                    x, y, w, h = ann["bbox"]

                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height

                    shuttle_lines.append(
                        f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                    )

                #FOOT
                elif ann["category_id"] == 1:

                    x, y, w, h = ann["bbox"]

                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height

                    line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

                    kpts = ann["keypoints"]

                    for i in range(0, len(kpts), 3):
                        kx = kpts[i] / width
                        ky = kpts[i + 1] / height
                        v = kpts[i + 2]
                        line += f" {kx:.6f} {ky:.6f} {v}"

                    pose_lines.append(line)

            #Write shuttle labels
            shuttle_label_path = f"{OUTPUT_ROOT}_shuttle/labels/{split_name}/{filename.replace('.jpg','.txt')}"
            with open(shuttle_label_path, "w") as f:
                f.write("\n".join(shuttle_lines))

            #Write pose labels
            pose_label_path = f"{OUTPUT_ROOT}_pose/labels/{split_name}/{filename.replace('.jpg','.txt')}"
            with open(pose_label_path, "w") as f:
                f.write("\n".join(pose_lines))

print("Dataset creation complete.")