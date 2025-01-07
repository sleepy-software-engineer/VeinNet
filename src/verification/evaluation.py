import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataloader import DataLoader
from model import Model
from sklearn.metrics import auc, roc_curve
from torch.nn import BCELoss
from torch_optimizer import Lookahead, RAdam

from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split_verification_closed

OUTPUT_PATH = "./src/verification/"


def plot_roc_curve(fpr, tpr, roc_auc, directory):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Parameters:
        fpr (list or numpy array): False Positive Rates.
        tpr (list or numpy array): True Positive Rates.
        roc_auc (float): Area Under the Curve (AUC).
        directory (str): The directory where the ROC plot will be saved.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="b", label=f"ROC Curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.fill_between(fpr, tpr, alpha=0.2, color="b", label="_nolegend_")  # Shaded area
    plt.plot(
        [0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess"
    )  # Diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(directory + "out/roc_curve.png")
    plt.close()


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


def test(
    model: Model, test_loader: DataLoader, device: torch.device, directory: str
) -> None:
    model.load_state_dict(
        torch.load(directory + "model/model.pth", map_location=device)
    )
    model.eval()
    probabilities = []
    labels = []

    with torch.no_grad():
        for images, claims, ground_truths in test_loader.generate_data():
            images = images.to(device)
            claims = claims.to(device)
            ground_truths = ground_truths.to(device).float()
            outputs = model(images, claims)
            probabilities.extend(outputs.cpu().numpy())
            labels.extend(ground_truths.cpu().numpy())

    probabilities = np.array(probabilities)
    labels = np.array(labels)

    thresholds = np.linspace(0, 1, 1000)
    far_list = []
    frr_list = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        far = np.mean((predictions == 1) & (labels == 0))
        frr = np.mean((predictions == 0) & (labels == 1))

        far_list.append(far)
        frr_list.append(frr)

    plot_far_vs_frr(far_list, frr_list, thresholds, directory)
    # Compute ROC curve (FPR, TPR) and AUC
    fpr, tpr, _ = roc_curve(labels, probabilities, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, directory)


def train(
    model: Model,
    train_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    directory: str,
) -> None:
    radam_optimizer = RAdam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    optimizer = Lookahead(radam_optimizer, k=10, alpha=0.5)
    criterion = BCELoss()
    checkpoint_path = os.path.realpath(os.path.join(directory, "model/model.pth"))

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for image, claim, ground_truth in train_loader.generate_data():
            image = image.to(device)
            claim = claim.to(device)
            ground_truth = ground_truth.to(device).float()
            prediction = model(image, claim)
            loss = criterion(prediction, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    split_data = split_verification_closed(PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED)
    train_loader = DataLoader(split_data, "train", mapping(PATIENTS))
    test_loader = DataLoader(split_data, "test", mapping(PATIENTS))

    model = Model(len(mapping(PATIENTS))).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train(
        model,
        train_loader,
        2,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        OUTPUT_PATH,
    )

    test(
        model,
        test_loader,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        OUTPUT_PATH,
    )
