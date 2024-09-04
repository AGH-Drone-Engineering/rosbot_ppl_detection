import os
import shutil
from sklearn.model_selection import train_test_split

images_path = "train/images"
labels_path = "train/labels"

image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg")]

train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)


def move_files(files, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for file in files:
        shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))
        label_file = file.replace(".jpg", ".txt")
        shutil.move(
            os.path.join(labels_path, label_file), os.path.join(target_dir, label_file)
        )


move_files(train_files, images_path, "../datasets/splitted/train/images")
move_files(val_files, images_path, "../datasets/splitted/val/images")
move_files(test_files, images_path, "../datasets/splitted/test/images")
