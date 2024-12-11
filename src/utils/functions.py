import random
from typing import DefaultDict
import glob
import os

def mapping(patients: list) -> DefaultDict:
    """
    Creates a mapping of patient IDs to their corresponding indices.

    Args:
        patients (list): A list of patient IDs.

    Returns:
        DefaultDict: A dictionary where the keys are patient IDs and the values are their indices in the sorted list.
    """
    return {patient_id: idx for idx, patient_id in enumerate(sorted(patients))} 

def split(patients: list, dataset_dir: str, hand: str, spectrum: str, seed: int) -> DefaultDict:
    """
    Splits the dataset of patient images into training, validation, and test sets.

    Args:
        patients (list): List of patient IDs.
        dataset_dir (str): Directory where the dataset is stored.
        hand (str): Hand specification (e.g., 'left' or 'right').
        spectrum (str): Spectrum specification (e.g., 'visible' or 'infrared').
        seed (int): Random seed for shuffling.

    Returns:
        DefaultDict: A dictionary with keys 'train', 'val', and 'test', each containing a list of image paths.
    """
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