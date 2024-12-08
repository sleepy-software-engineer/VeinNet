import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import os
import glob
import random
from typing import DefaultDict

from dataprocessor import VeinImageProcessor
from config import *

class SplitData:
    @staticmethod 
    def create_patient_id_mapping(patients):
        return {patient_id: idx for idx, patient_id in enumerate(sorted(patients))} 
    
    @staticmethod 
    def split_images_by_patient(patients, dataset_dir, hand="l", spectrum="940", seed=42):
        random.seed(seed)
        split_data = DefaultDict(list)

        for patient_id in patients:
            pattern = f"{patient_id}_{hand}_{spectrum}_*.jpg"
            image_paths = glob.glob(os.path.join(dataset_dir, pattern))
            assert (
                len(image_paths) == 6
            ), f"Patient {patient_id} does not have exactly 6 images."

            image_paths = sorted(image_paths)

            random.shuffle(image_paths)
            split_data["train"].extend(image_paths[:3]) 
            split_data["val"].extend(image_paths[3:5])
            split_data["test"].extend(image_paths[5:])

        return split_data
class VeinImageDataLoader:
    def __init__(self, split_data, split_name, id_mapping):
        self.image_paths = split_data[split_name]
        self.id_mapping = id_mapping
    
    def generate_data(self):
        for image_path in self.image_paths:
            patient_id = os.path.basename(image_path).split("_")[0]
            label = self.id_mapping[patient_id]  # Convert patient ID to mapped label
            v = VeinImageProcessor()
            vein_image = v.preprocess_image(image_path)

            yield vein_image, label

def test(dataloader, split_name):
    print(f"Testing {split_name} Vein Image DataLoader")
    try:
        patient_image_counts = DefaultDict(int)
        for vein_image, patient_id in dataloader.generate_data():
            print(f"Patient ID: {patient_id}, Vein Image Shape: {vein_image.shape}")
            patient_image_counts[patient_id] += 1

        # Validate counts
        for patient_id, count in patient_image_counts.items():
            if split_name == "train":
                assert (
                    count == 3
                ), f"Expected 3 images for training, got {count} for Patient {patient_id}"
            elif split_name == "val":
                assert (
                    count == 2
                ), f"Expected 2 images for validation, got {count} for Patient {patient_id}"
            elif split_name == "test":
                assert (
                    count == 1
                ), f"Expected 1 image for testing, got {count} for Patient {patient_id}"

        print(f"{split_name} split test passed!")
    except Exception as e:
        print(f"Error testing {split_name} Vein Image DataLoader: {e}")


if __name__ == "__main__":
    data = split_images_by_patient(PATIENTS, DATASET_PATH)
    
    train_loader = VeinImageDataLoader(data, "train")
    val_loader = VeinImageDataLoader(data, "val")
    test_loader = VeinImageDataLoader(data, "test")

    test(train_loader, "train")
    test(val_loader, "val")
    test(test_loader, "test")