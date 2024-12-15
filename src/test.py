import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
from torch import nn

from dataloader import DataLoader
from model import Model
from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split


def test(
    model: nn.Module,
    test_loader,
    device: torch.device,
    checkpoint_path: str,
    output_dir: str,
) -> None:
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    all_preds = []
    all_labels = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for vein_image, label in test_loader.generate_data():
            vein_image = (
                torch.tensor(vein_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            label = torch.tensor([int(label)], dtype=torch.long).to(device)

            output = model(vein_image)
            test_loss += criterion(output, label).item()
            pred = torch.argmax(output, dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    mcc = matthews_corrcoef(all_labels, all_preds)
    accuracy = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

    metrics_data = {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Matthews Corr. Coeff.",
        ],
        "Value": [accuracy, precision, recall, f1_score, mcc],
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

    print(f"Test metrics saved in: {output_dir}")


if __name__ == "__main__":
    mapping_ids = mapping(PATIENTS)
    split_data = split(PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED)

    test_loader = DataLoader(split_data, "test", mapping_ids)

    num_classes = len(mapping_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes).to(device)
    test(model, test_loader, device, "./model/model.pth", "./model/test")
