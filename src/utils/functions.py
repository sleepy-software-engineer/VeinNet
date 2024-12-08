import random
from typing import DefaultDict
import glob
import os
import torch

def mapping(patients):
    return {patient_id: idx for idx, patient_id in enumerate(sorted(patients))} 

def split(patients, dataset_dir, hand="l", spectrum="940", seed=42):
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

def prepare(dataloader):
    images = []
    labels = []
    for vein_image, patient_id in dataloader.generate_data():
        images.append(torch.tensor(vein_image, dtype=torch.float32).unsqueeze(0))
        labels.append(int(patient_id)) 
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels