import glob
import cv2
import numpy as np
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import os
import glob
from typing import DefaultDict

from dataprocessor import HandGeometryProcessor
from config import *
from utils.utils import create_patient_id_mapping, split_images_by_patient

class HandGeometryDataLoader:
    def __init__(self, split_data, split_name, id_mapping):
        self.image_paths = split_data[split_name]
        self.id_mapping = id_mapping

    def generate_data(self):
        for image_path in self.image_paths:
            patient_id = os.path.basename(image_path).split("_")[0]
            label = self.id_mapping[patient_id]
            hd = HandGeometryProcessor() 
            features = hd.extract_features(image_path)
            yield features, label

def test(dataloader, split_name):
    print(f"Testing {split_name} Hand Geometry DataLoader")
    try:
        patient_feature_counts = DefaultDict(int)
        for features, patient_id in dataloader.generate_data():
            print(f"Patient ID: {patient_id}, Features: {features}")
            assert len(features) == 11, f"Expected 11 features, got {len(features)}"
            patient_feature_counts[patient_id] += 1

        for patient_id, count in patient_feature_counts.items():
            if split_name == "train":
                assert count == 3, f"Expected 3 samples for training, got {count} for Patient {patient_id}"
            elif split_name == "val":
                assert count == 2, f"Expected 2 samples for validation, got {count} for Patient {patient_id}"
            elif split_name == "test":
                assert count == 1, f"Expected 1 sample for testing, got {count} for Patient {patient_id}"

        print(f"{split_name} split test passed!")
    except Exception as e:
        print(f"Error testing {split_name} Hand Geometry DataLoader: {e}")


if __name__ == "__main__":
    id_mapping = create_patient_id_mapping(PATIENTS)
    split_data = split_images_by_patient(PATIENTS, DATASET_PATH)

    # Create dataloaders for each split
    train_loader = HandGeometryDataLoader(split_data, "train", id_mapping)
    val_loader = HandGeometryDataLoader(split_data, "val", id_mapping)
    test_loader = HandGeometryDataLoader(split_data, "test", id_mapping)

    # Test each split
    test(train_loader, "train")
    test(val_loader, "val")
    test(test_loader, "test")