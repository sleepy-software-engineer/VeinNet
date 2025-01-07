import os
import sys
from typing import DefaultDict, Generator, Tuple

import cv2
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


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
        self.num_classes = len(id_mapping)
        self.imposters = [1, 40, 30, 20]
        self.imposter_index = 0

    def generate_data(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        for image_path in self.image_paths:
            vein_image, genuine_label = self._generate_image(image_path)
            vein_tensor = (
                torch.tensor(vein_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            genuine_label_tensor = torch.tensor(
                [int(genuine_label)], dtype=torch.long
            ).to(self.device)
            is_genuine = torch.tensor([True], dtype=torch.bool).to(self.device)
            yield vein_tensor, genuine_label_tensor, is_genuine

            fake_label = self.imposters[self.imposter_index]
            self.imposter_index = (self.imposter_index + 1) % len(self.imposters)

            if fake_label == genuine_label:
                continue

            fake_label_tensor = torch.tensor([int(fake_label)], dtype=torch.long).to(
                self.device
            )
            is_fake = torch.tensor([False], dtype=torch.bool).to(self.device)
            yield vein_tensor, fake_label_tensor, is_fake

    def _generate_image(self, image_path: str) -> Tuple[cv2.UMat, int]:
        patient_id = os.path.basename(image_path).split("_")[0]
        label = self.id_mapping[patient_id]
        vein_image = DataProcessor.preprocess_image(image_path)
        return vein_image, label
