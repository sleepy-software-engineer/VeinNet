import glob
import os
import random
from typing import DefaultDict, List, Tuple


def mapping(patients: list) -> DefaultDict[str, int]:
    return {patient_id: idx for idx, patient_id in enumerate(sorted(patients))}


# TODO: rename to identification_close
# TODO: split only between train and test and not val and then run the closed set identification
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


# TODO: rename to identification_open
def split_open(
    patients: List[str], dataset_dir: str, hand: str, spectrum: str, seed: int
) -> Tuple[DefaultDict[str, List[str]], DefaultDict[str, List[str]]]:
    random.seed(seed)
    split_data_known = DefaultDict(list)
    split_data_unknown = DefaultDict(list)
    random.shuffle(patients)

    num_known = int(len(patients) * 0.7)
    known_patients = patients[:num_known]
    unknown_patients = patients[num_known:]

    for patient_id in known_patients:
        pattern = f"{patient_id}_{hand}_{spectrum}_*.jpg"
        image_paths = glob.glob(os.path.join(dataset_dir, pattern))
        image_paths = sorted(image_paths)
        random.shuffle(image_paths)
        split_data_known["train"].extend(image_paths[:4])
        split_data_known["test"].extend(image_paths[2:])

    for patient_id in unknown_patients:
        pattern = f"{patient_id}_{hand}_{spectrum}_*.jpg"
        image_paths = glob.glob(os.path.join(dataset_dir, pattern))
        split_data_unknown["test"].extend(image_paths)

    return split_data_known, split_data_unknown


def split_verification_closed(
    patients: list, dataset_dir: str, hand: str, spectrum: str, seed: int
) -> DefaultDict[str, list]:
    random.seed(seed)
    split_data = DefaultDict(list)

    for patient_id in patients:
        pattern = f"{patient_id}_{hand}_{spectrum}_*.jpg"
        image_paths = glob.glob(os.path.join(dataset_dir, pattern))
        image_paths = sorted(image_paths)
        random.shuffle(image_paths)
        split_data["train"].extend(image_paths[:4])
        split_data["test"].extend(image_paths[4:])

    return split_data


if __name__ == "__main__":
    pass
