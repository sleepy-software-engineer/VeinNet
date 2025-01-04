import os
import sys
from typing import DefaultDict

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from dataprocessor import DataProcessor


class DataLoader:
    def __init__(
        self,
        split_data_known: DefaultDict[str, list],
        split_data_unknown: DefaultDict[str, list],
        split_name: str,
        id_mapping: DefaultDict[str, int],
    ) -> None:
        self.image_paths_known = split_data_known[split_name]
        self.image_paths_unknown = split_data_unknown[split_name]
        self.id_mapping = id_mapping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_data(self):
        for image_path in self.image_paths_known:
            vein_image, label, _ = self.generate_image(image_path)
            vein_tensor = (
                torch.tensor(vein_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            label_tensor = torch.tensor([int(label)], dtype=torch.long).to(self.device)
            yield vein_tensor, label_tensor

        for image_path in self.image_paths_unknown:
            vein_image, label, _ = self.generate_image(image_path)
            vein_tensor = (
                torch.tensor(vein_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            label_tensor = torch.tensor([int(-1)], dtype=torch.long).to(self.device)
            yield vein_tensor, label_tensor

    def generate_image(self, image_path: str):
        patient_id = os.path.basename(image_path).split("_")[0]
        label = self.id_mapping[patient_id]
        vein_image = DataProcessor.preprocess_image(image_path)
        return vein_image, label, patient_id


if __name__ == "__main__":
    pass
