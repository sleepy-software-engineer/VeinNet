import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


import matplotlib.pyplot as plt
import numpy as np
from dataloader import DataLoader
from model import Model
from torch import device
from torch.nn import BCELoss
from torch_optimizer import Lookahead, RAdam

from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split_verification_closed


def test(model: Model, test_loader: DataLoader, device: torch.device) -> None:
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
        far = np.mean((predictions == 1) & (labels == 0))  # False acceptance
        frr = np.mean((predictions == 0) & (labels == 1))  # False rejection

        far_list.append(far)
        frr_list.append(frr)

    far_list = np.array(far_list)
    frr_list = np.array(frr_list)

    eer_index = np.argmin(np.abs(far_list - frr_list))
    eer = (far_list[eer_index] + frr_list[eer_index]) / 2
    best_threshold = thresholds[eer_index]

    print(f"EER: {eer:.4f}, Best Threshold: {best_threshold:.4f}")

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
    plt.savefig("verification_far_frr.png")

    return eer, best_threshold


def train(
    model: Model,
    train_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
) -> Model:
    radam_optimizer = RAdam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    optimizer = Lookahead(radam_optimizer, k=10, alpha=0.5)
    criterion = BCELoss()

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

    return model


if __name__ == "__main__":
    mapping_ids = mapping(PATIENTS)
    split_data = split_verification_closed(PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED)
    train_loader = DataLoader(split_data, "train", mapping_ids)
    test_loader = DataLoader(split_data, "test", mapping_ids)

    num_classes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes).to(device)

    model = train(
        model,
        train_loader,
        num_epochs=5,
        device=device,
    )

    test(model, test_loader, device)
