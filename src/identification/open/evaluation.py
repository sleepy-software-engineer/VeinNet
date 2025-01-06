import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib.pyplot as plt
import numpy as np
from dataloader import DataLoader
from model import Model
from torch import device
from torch.nn import CrossEntropyLoss
from torch_optimizer import Lookahead, RAdam

from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split_open


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

    for threshold in thresholds:
        far = np.mean(
            (probabilities >= threshold) & (labels == -1)
        )  # Unknown classified as known
        frr = np.mean(
            (probabilities < threshold) & (labels != -1)
        )  # Known classified as unknown

        far_list.append(far)
        frr_list.append(frr)

    far_list = np.array(far_list)
    frr_list = np.array(frr_list)

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
    plt.savefig("eer.png")

    print(f"EER: {eer:.4f}, Best Threshold: {best_threshold:.4f}")


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
