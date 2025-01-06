import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import random

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from dataloader import DataLoader
from model import Model
from sklearn.metrics.pairwise import cosine_similarity
from torch import device

from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split_verification_closed


def test(model, test_loader, device, num_classes):
    model.eval()

    distances = []
    labels = []  # True (1) for correct claim, False (0) for fake claim

    with torch.no_grad():
        for image, true_label in test_loader.generate_data():
            image = image.to(device)

            # Randomly decide to make a true or fake claim
            if random.random() < 0.5:
                claimed_label = true_label  # True claim
                labels.append(1)
            else:
                fake_label = random.choice(
                    [i for i in range(num_classes) if i != true_label.item()]
                )
                claimed_label = torch.tensor(fake_label, device=device)  # Fake claim
                labels.append(0)

            # Forward pass through convolutional layers
            x = F.relu(model.conv1(image))
            x = F.max_pool2d(x, 2)
            x = F.relu(model.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(model.conv3(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(model.conv4(x))
            x = x.view(x.size(0), -1)  # Flatten the output

            # Compute distance between features and claimed label vector
            features = F.relu(model.fc1(x)).cpu().numpy()
            claimed_vector = np.zeros((1, features.shape[1]))
            claimed_vector[0, claimed_label.item()] = 1  # One-hot encoding for claim

            distance = 1 - cosine_similarity(features, claimed_vector)
            distances.append(distance[0][0])

    # Threshold Evaluation
    thresholds = np.linspace(0, max(distances), 1000)
    far_list, frr_list = [], []

    for threshold in thresholds:
        # FAR: Fraction of fake claims incorrectly accepted
        far = np.mean(
            [d <= threshold for d, label in zip(distances, labels) if label == 0]
        )
        # FRR: Fraction of true claims incorrectly rejected
        frr = np.mean(
            [d > threshold for d, label in zip(distances, labels) if label == 1]
        )
        far_list.append(far)
        frr_list.append(frr)

    # Determine Equal Error Rate (EER)
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
    plt.title("Verification: FAR vs. FRR")
    plt.legend()
    plt.grid()
    plt.savefig("verification_eer.png")
    print(f"EER: {eer:.4f}, Best Threshold: {best_threshold:.4f}")

    return eer, best_threshold


def __test(model, test_loader, device, num_classes):
    model.eval()

    probabilities = []
    true_labels = []
    claimed_labels = []
    claim_types = []  # Store whether the claim is true or fake

    with torch.no_grad():
        for image, true_label in test_loader.generate_data():
            image = image.to(device)

            # Randomly decide to make a true or fake claim
            if random.random() < 0.5:
                claimed_label = true_label  # True claim
                claim_types.append("true")
            else:
                # Choose a random label that is different from the true label
                fake_label = random.choice(
                    [i for i in range(num_classes) if i != true_label.item()]
                )
                claimed_label = torch.tensor(fake_label, device=device)  # Fake claim
                claim_types.append("fake")

            # Model prediction
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            prob_claimed = probs[0, claimed_label.item()].item()

            # Store results
            probabilities.append(prob_claimed)
            true_labels.append(true_label.item())
            claimed_labels.append(claimed_label.item())

    # Analyze results
    thresholds = np.linspace(0, 1, 1000)
    far_list, frr_list = [], []

    for threshold in thresholds:
        # FAR: Fraction of fake claims incorrectly accepted as true
        far = np.mean(
            [
                p >= threshold
                for p, ctype in zip(probabilities, claim_types)
                if ctype == "fake"
            ]
        )
        # FRR: Fraction of true claims incorrectly rejected as false
        frr = np.mean(
            [
                p < threshold
                for p, ctype in zip(probabilities, claim_types)
                if ctype == "true"
            ]
        )
        far_list.append(far)
        frr_list.append(frr)

    # Determine Equal Error Rate (EER)
    far_list = np.array(far_list)
    frr_list = np.array(frr_list)
    eer_index = np.argmin(np.abs(far_list - frr_list))
    eer = (far_list[eer_index] + frr_list[eer_index]) / 2
    best_threshold = thresholds[eer_index]

    # Print results for fake claims
    fake_claim_indices = [i for i, ctype in enumerate(claim_types) if ctype == "fake"]
    fake_claim_results = [
        (true_labels[i], claimed_labels[i], probabilities[i])
        for i in fake_claim_indices
    ]
    print("Fake Claim Results (True Label, Claimed Label, Probability):")
    for result in fake_claim_results:
        print(result)

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
    plt.title("Verification: FAR vs. FRR")
    plt.legend()
    plt.grid()
    plt.savefig("verification_eer.png")
    print(f"EER: {eer:.4f}, Best Threshold: {best_threshold:.4f}")

    return eer, best_threshold


def train(model, train_loader, num_epochs, device, num_classes=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for image, true_label in train_loader.generate_data():
            # Ensure correct dimensions for image and labels
            image = image.to(device).unsqueeze(0)  # Add batch dimension
            true_label = true_label.to(device)

            # Generate a fake claim
            if random.random() < 0.5:  # 50% chance of generating a fake claim
                fake_label = random.choice(
                    [j for j in range(num_classes) if j != true_label.item()]
                )
                fake_label = torch.tensor(fake_label, device=device)
            else:
                fake_label = true_label

            # Create combined images and labels
            combined_images = torch.cat([image, image], dim=0)  # Shape [2, 1, 128, 128]
            combined_labels = torch.tensor(
                [true_label.item(), fake_label.item()], device=device
            )  # Shape [2]

            # Forward pass
            outputs = model(combined_images)
            loss = criterion(outputs, combined_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")

    return model


# def train(
#     model,
#     train_loader: DataLoader,
#     num_epochs,
#     device,
# ):
#     radam_optimizer = RAdam(model.parameters(), lr=0.001, weight_decay=5e-4)
#     optimizer = Lookahead(radam_optimizer, k=10, alpha=0.5)
#     criterion = CrossEntropyLoss()

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         for images, labels in train_loader.generate_data():
#             images, labels = images.to(device), labels.to(device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#         train_loss /= len(train_loader.image_paths)
#         print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")
#     return model


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
        num_epochs=25,
        device=device,
    )

    test(model, test_loader, device, num_classes)
