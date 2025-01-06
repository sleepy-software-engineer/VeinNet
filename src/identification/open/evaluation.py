import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from dataloader import DataLoader
from model import Model
from torch import device
from torch.nn import CrossEntropyLoss
from torch_optimizer import Lookahead, RAdam

from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split_open

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from scipy.interpolate import interp1d
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve


def test(model, test_loader, device):
    model.eval()
    probabilities = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in test_loader.generate_data():
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)

            probabilities.extend(max_probs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    probabilities = np.array(probabilities)
    labels = np.array(labels)

    thresholds = np.linspace(0, 1, 1000)

    far_list = []
    frr_list = []
    dir_list = []

    for threshold in thresholds:
        far = np.mean((probabilities >= threshold) & (labels == -1))  # False acceptance
        frr = np.mean((probabilities < threshold) & (labels != -1))  # False rejection
        dir_rate = 1 - frr  # Detection Identification Rate (DIR)

        far_list.append(far)
        frr_list.append(frr)
        dir_list.append(dir_rate)

    far_list = np.array(far_list)
    frr_list = np.array(frr_list)
    dir_list = np.array(dir_list)

    eer_index = np.argmin(np.abs(far_list - frr_list))
    eer = (far_list[eer_index] + frr_list[eer_index]) / 2
    best_threshold = thresholds[eer_index]

    # Plot FAR vs. FRR
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far_list, label="FAR (False Acceptance Rate)")
    plt.plot(thresholds, frr_list, label="FRR (False Rejection Rate)")
    plt.axvline(
        x=best_threshold,
        color="r",
        linestyle="--",
        label=f"EER Threshold = {best_threshold:.4f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("FAR vs. FRR")
    plt.legend()
    plt.grid()
    plt.savefig("far_frr_plot.png")

    # Plot ROC Curve for Best Threshold
    fpr, tpr, roc_thresholds = roc_curve(
        (labels != -1).astype(int), probabilities, pos_label=1
    )
    roc_auc = auc(fpr, tpr)

    # Interpolate TPR for the best threshold
    interp_tpr = interp1d(roc_thresholds, tpr, kind="linear", fill_value="extrapolate")
    best_tpr = interp_tpr(best_threshold)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="b", label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.scatter(
        fpr[np.abs(roc_thresholds - best_threshold).argmin()],
        best_tpr,
        color="red",
        label="Best Threshold",
    )
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve for Best Threshold ({best_threshold:.4f})")
    plt.legend()
    plt.grid()
    plt.savefig("roc_curve_best_threshold.png")

    # Plot DET Curve for Best Threshold (Log Scale)
    plt.figure(figsize=(8, 6))
    plt.plot(far_list, frr_list, label="DET Curve")
    plt.scatter(
        far_list[eer_index], frr_list[eer_index], color="red", label="Best Threshold"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(f"DET Curve for Best Threshold ({best_threshold:.4f})")
    plt.grid(which="both")
    plt.legend()
    plt.savefig("det_curve_best_threshold.png")

    # Save FAR, FRR, DIR to CSV
    with open("threshold_metrics.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Threshold", "FAR", "FRR", "DIR"])
        for threshold, far, frr, dir_rate in zip(
            thresholds, far_list, frr_list, dir_list
        ):
            csvwriter.writerow([threshold, far, frr, dir_rate])

    # Compute Confusion Matrix for Best Threshold
    predictions = (probabilities >= best_threshold).astype(int)
    true_labels = (labels != -1).astype(int)  # Known are 1, Unknown are 0
    cm = confusion_matrix(true_labels, predictions, labels=[1, 0])

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Known", "Unknown"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold = {best_threshold:.4f})")
    plt.savefig("confusion_matrix.png")


def train(
    model,
    train_loader: DataLoader,
    num_epochs,
    device,
):
    radam_optimizer = RAdam(model.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer = Lookahead(radam_optimizer, k=10, alpha=0.5)
    criterion = CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader.generate_data():
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.image_paths_known)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")
    return model


if __name__ == "__main__":
    mapping_ids = mapping(PATIENTS)
    split_data_known, split_data_unknown = split_open(
        PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED
    )
    train_loader = DataLoader(
        split_data_known, split_data_unknown, "train", mapping_ids
    )
    test_loader = DataLoader(split_data_known, split_data_unknown, "test", mapping_ids)

    num_classes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes).to(device)

    model = train(
        model,
        train_loader,
        num_epochs=25,
        device=device,
    )

    test(model, test_loader, device)
