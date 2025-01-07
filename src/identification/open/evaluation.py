import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from dataloader import DataLoader
from model import Model
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from torch.nn import CrossEntropyLoss
from torch_optimizer import Lookahead, RAdam

from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split_identification_open

OUTPUT_PATH = "./src/identification/open/"


def plot_confusion_matrix(
    probabilities, labels, far_list, frr_list, thresholds, directory
):
    far_list = np.array(far_list)
    frr_list = np.array(frr_list)
    eer_index = np.argmin(np.abs(far_list - frr_list))
    best_threshold = thresholds[eer_index]
    predictions = (probabilities >= best_threshold).astype(int)
    true_labels = (labels != -1).astype(int)
    cm = confusion_matrix(true_labels, predictions, labels=[1, 0])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Known", "Unknown"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(directory + "out/confusion_matrix.png")
    plt.close()


def save_threshold_metrics(thresholds, far_list, frr_list, dir_list, directory):
    selected_thresholds = np.arange(0.1, 1.1, 0.1)
    with open(directory + "out/threshold_metrics.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Threshold", "FAR", "FRR", "DIR"])
        for selected_threshold in selected_thresholds:
            index = (np.abs(thresholds - selected_threshold)).argmin()
            csvwriter.writerow(
                [
                    thresholds[index],
                    far_list[index],
                    frr_list[index],
                    dir_list[index],
                ]
            )


def plot_far_vs_frr(far_list, frr_list, thresholds, directory):
    far_list = np.array(far_list)
    frr_list = np.array(frr_list)
    eer_index = np.argmin(np.abs(far_list - frr_list))
    best_threshold = thresholds[eer_index]
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far_list, label="FAR (False Acceptance Rate)", color="b")
    plt.fill_between(thresholds, far_list, alpha=0.2, color="b", label="_nolegend_")
    plt.plot(thresholds, frr_list, label="FRR (False Rejection Rate)", color="g")
    plt.fill_between(thresholds, frr_list, alpha=0.2, color="g", label="_nolegend_")
    plt.axvline(
        x=best_threshold,
        color="r",
        linestyle="--",
        label=f"EER Threshold = {best_threshold:.4f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("FAR vs. FRR")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(directory + "out/far_vs_frr.png")
    plt.close()


def plot_roc_curve(labels, probabilities, directory):
    fpr, tpr, _ = roc_curve((labels != -1).astype(int), probabilities, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="b", label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.fill_between(fpr, tpr, alpha=0.2, color="b", label="_no_legend_")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(directory + "out/roc_curve.png")
    plt.close()


def plot_det_curve(far_list, frr_list, directory):
    plt.figure(figsize=(8, 6))
    plt.plot(far_list, frr_list, label="DET Curve", color="b", linewidth=2)
    plt.fill_between(far_list, frr_list, alpha=0.2, color="b", label="_nolegend_")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title("DET Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(directory + "out/det_curve.png")
    plt.close()


def test(model: Model, test_loader: DataLoader, device: torch.device, directory: str):
    model.load_state_dict(
        torch.load(directory + "model/model.pth", map_location=device)
    )
    model.eval()
    probabilities, labels = [], []

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
    far_list, frr_list, dir_list = [], [], []

    for threshold in thresholds:
        far = np.mean((probabilities >= threshold) & (labels == -1))
        frr = np.mean((probabilities < threshold) & (labels != -1))
        dir_rate = 1 - frr

        far_list.append(far)
        frr_list.append(frr)
        dir_list.append(dir_rate)

    plot_far_vs_frr(far_list, frr_list, thresholds, directory)
    plot_roc_curve(labels, probabilities, directory)
    plot_det_curve(far_list, frr_list, directory)
    save_threshold_metrics(thresholds, far_list, frr_list, dir_list, directory)
    plot_confusion_matrix(
        probabilities, labels, far_list, frr_list, thresholds, directory
    )


def train(
    model: Model,
    train_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    directory: str,
):
    radam_optimizer = RAdam(model.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer = Lookahead(radam_optimizer, k=10, alpha=0.5)
    criterion = CrossEntropyLoss()
    checkpoint_path = os.path.realpath(os.path.join(directory, "model/model.pth"))

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

    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    split_data_known, split_data_unknown = split_identification_open(
        PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED
    )
    train_loader = DataLoader(
        split_data_known, split_data_unknown, "train", mapping(PATIENTS)
    )
    test_loader = DataLoader(
        split_data_known, split_data_unknown, "test", mapping(PATIENTS)
    )

    model = Model(len(mapping(PATIENTS))).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # train(
    #     model,
    #     train_loader,
    #     25,
    #     torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #     OUTPUT_PATH,
    # )

    test(
        model,
        test_loader,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        OUTPUT_PATH,
    )
