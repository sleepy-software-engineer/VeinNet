import os
import sys
from typing import DefaultDict

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from dataprocessor import DataProcessor


class DataLoader:
    """
    DataLoader class for loading and processing biometric data.
    """

    def __init__(
        self,
        split_data: DefaultDict[str, list],
        split_name: str,
        id_mapping: DefaultDict[str, int],
    ) -> None:
        self.image_paths = split_data[split_name]
        self.id_mapping = id_mapping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_data(self):
        """
        Generates data by iterating over image paths and processing each image.
        """
        for image_path in self.image_paths:
            vein_image, label, _ = self.generate_image(image_path)
            vein_tensor = (
                torch.tensor(vein_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            label_tensor = torch.tensor([int(label)], dtype=torch.long).to(self.device)
            yield vein_tensor, label_tensor

    def generate_image(self, image_path: str):
        """
        Generates a preprocessed vein image along with its label and patient ID.
        """
        patient_id = os.path.basename(image_path).split("_")[0]
        label = self.id_mapping[patient_id]
        vein_image = DataProcessor.preprocess_image(image_path)
        return vein_image, label, patient_id
