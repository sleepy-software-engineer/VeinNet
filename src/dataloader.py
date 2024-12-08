import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from typing import DefaultDict
from dataprocessor import DataProcessor
from utils.functions import split, mapping
from utils.config import PATIENTS, DATASET_PATH

class DataLoader:
    def __init__(self, split_data, split_name, id_mapping):
        self.image_paths = split_data[split_name]
        self.id_mapping = id_mapping
    
    def generate_data(self):
        for image_path in self.image_paths:
            patient_id = os.path.basename(image_path).split("_")[0]
            label = self.id_mapping[patient_id]
            processor = DataProcessor()
            vein_image = processor.preprocess_image(image_path)
            yield vein_image, label

def test(dataloader, split_name):
    try:
        patient_image_counts = DefaultDict(int)
        for vein_image, patient_id in dataloader.generate_data():
            patient_image_counts[patient_id] += 1

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
    split_data = split(PATIENTS, DATASET_PATH)
    mapping_ids = mapping(PATIENTS)
    
    train_loader = DataLoader(split_data, "train", mapping_ids)
    val_loader = DataLoader(split_data, "val", mapping_ids)
    test_loader = DataLoader(split_data, "test", mapping_ids)

    test(train_loader, "train")
    test(val_loader, "val")
    test(test_loader, "test")