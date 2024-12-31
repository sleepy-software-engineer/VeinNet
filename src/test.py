import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn

from dataloader import DataLoader
from model import Model
from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split


def calculate_cmc(all_preds, all_labels, num_classes):
    cmc_curve = np.zeros(num_classes)
    for label, pred_scores in zip(all_labels, all_preds):
        ranked_preds = np.argsort(pred_scores)[::-1]
        rank = np.where(ranked_preds == label)[0][0]
        cmc_curve[rank:] += 1
    return cmc_curve / len(all_labels)


def calculate_rank_n(all_preds, all_labels, n):
    correct = 0
    for label, pred_scores in zip(all_labels, all_preds):
        top_n_preds = np.argsort(pred_scores)[::-1][:n]
        if label in top_n_preds:
            correct += 1
    return correct / len(all_labels)


def plot_confusion_matrix(conf_matrix, labels, output_path):
    plt.figure(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    disp.plot(cmap="viridis", ax=plt.gca(), xticks_rotation="vertical", colorbar=True)

    if disp.text_ is not None:
        for text in disp.text_.ravel():
            text.set_visible(False)

    plt.xticks(
        ticks=np.arange(len(labels)),
        labels=labels,
        fontsize=8,
        rotation=90,
        ha="center",
    )
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=8, va="center")
    plt.grid(color="black", linestyle="--", linewidth=0.5)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def test(
    model: nn.Module,
    test_loader,
    device: torch.device,
    checkpoint_path: str,
    output_dir: str,
    num_classes: int,
) -> None:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    all_preds = []
    all_labels = []
    all_scores = []
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
            pred_scores = output.cpu().numpy().flatten()

            all_preds.append(np.argmax(pred_scores))
            all_scores.append(pred_scores)
            all_labels.append(label.item())

    # Metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    accuracy = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

    # CMC Curve
    cmc_curve = calculate_cmc(all_scores, all_labels, num_classes)
    rank_1_rate = cmc_curve[0]
    rank_5_rate = calculate_rank_n(all_scores, all_labels, 5)

    # plot CMC Curve
    plt.figure(figsize=(6, 6))
    plt.plot(cmc_curve, marker="o")
    plt.xlabel("Rank")
    plt.ylabel("Identification Rate")
    plt.title("CMC Curve")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "cmc_curve.png"))
    plt.close()

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(
        conf_matrix,
        labels=range(num_classes),
        output_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    # Save Metrics
    metrics_data = {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Rank-1 Rate",
            "Rank-5 Rate",
        ],
        "Value": [accuracy, precision, recall, f1_score, rank_1_rate, rank_5_rate],
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

    print(f"Test metrics and confusion matrix saved in: {output_dir}")


if __name__ == "__main__":
    mapping_ids = mapping(PATIENTS)
    split_data = split(PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED)

    test_loader = DataLoader(split_data, "test", mapping_ids)

    num_classes = len(mapping_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes).to(device)
    test(
        model,
        test_loader,
        device,
        "./model/model.pth",
        "./model/test",
        num_classes,
    )
