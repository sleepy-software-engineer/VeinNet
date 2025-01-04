import glob
import os
import random
from typing import DefaultDict, Tuple


def mapping(patients: list) -> DefaultDict[str, int]:
    return {patient_id: idx for idx, patient_id in enumerate(sorted(patients))}


def split_closed(
    patients: list, dataset_dir: str, hand: str, spectrum: str, seed: int
) -> DefaultDict[str, list]:
    random.seed(seed)
    split_data = DefaultDict(list)
    for patient_id in patients:
        pattern = f"{patient_id}_{hand}_{spectrum}_*.jpg"
        image_paths = glob.glob(os.path.join(dataset_dir, pattern))
        image_paths = sorted(image_paths)
        random.shuffle(image_paths)
        split_data["train"].extend(image_paths[:3])
        split_data["val"].extend(image_paths[3:4])
        split_data["test"].extend(image_paths[4:])
    return split_data


def split_open(
    patients: list, dataset_dir: str, hand: str, spectrum: str, seed: int
) -> Tuple[DefaultDict[str, list], DefaultDict[str, list]]:
    random.seed(seed)
    split_data_known = DefaultDict(list)
    split_data_unknown = DefaultDict(list)

    random.shuffle(patients)
    num_known = int(len(patients) * 0.7)
    num_val_unknown = int(len(patients) * 0.1)

    known_patients = patients[:num_known]
    val_unknown_patients = patients[num_known : num_known + num_val_unknown]
    test_unknown_patients = patients[num_known + num_val_unknown :]

    for patient_id in known_patients:
        pattern = f"{patient_id}_{hand}_{spectrum}_*.jpg"
        image_paths = glob.glob(os.path.join(dataset_dir, pattern))
        image_paths = sorted(image_paths)
        random.shuffle(image_paths)
        split_data_known["train"].extend(image_paths[:3])
        split_data_known["val"].append(image_paths[1])
        split_data_known["test"].append(image_paths[2])

    for patient_id in val_unknown_patients:
        pattern = f"{patient_id}_{hand}_{spectrum}_*.jpg"
        image_paths = glob.glob(os.path.join(dataset_dir, pattern))
        split_data_unknown["val"].extend(image_paths)

    for patient_id in test_unknown_patients:
        pattern = f"{patient_id}_{hand}_{spectrum}_*.jpg"
        image_paths = glob.glob(os.path.join(dataset_dir, pattern))
        split_data_unknown["test"].extend(image_paths)

    return split_data_known, split_data_unknown


if __name__ == "__main__":
    # from config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM

    # split_open(
    #     patients=PATIENTS,
    #     dataset_dir=DATASET_PATH,
    #     hand=HAND,
    #     spectrum=SPECTRUM,
    #     seed=SEED,
    # )
    pass
