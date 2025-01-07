import os
import sys
from typing import DefaultDict, Generator, Tuple

import cv2
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from dataprocessor import DataProcessor


class DataLoader:
    def __init__(
        self,
        split_data: DefaultDict[str, list],
        split_name: str,
        id_mapping: DefaultDict[str, int],
    ) -> None:
        self.image_paths = split_data[split_name]
        self.id_mapping = id_mapping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_data(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        for image_path in self.image_paths:
            vein_image, label = self._generate_image(image_path)
            vein_tensor = (
                torch.tensor(vein_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            label_tensor = torch.tensor([int(label)], dtype=torch.long).to(self.device)
            yield vein_tensor, label_tensor

    def _generate_image(self, image_path: str) -> Tuple[cv2.UMat, int]:
        patient_id = os.path.basename(image_path).split("_")[0]
        label = self.id_mapping[patient_id]
        vein_image = DataProcessor.preprocess_image(image_path)
        return vein_image, label
