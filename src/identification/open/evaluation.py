import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import torch.nn.functional as F
from dataloader import DataLoader
from model import Model
from torch_optimizer import Lookahead, RAdam

from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split_open


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_classes: int,
    epochs: int,
    lr: float,
    device: torch.device,
):
    model.to(device)
    radam_optimizer = RAdam(model.parameters(), lr=lr, weight_decay=1e-3)
    optimizer = Lookahead(radam_optimizer, k=5, alpha=0.5)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0.0

        class_centroids = torch.zeros(
            num_classes, model.embedding_layer.out_features
        ).to(device)
        class_counts = torch.zeros(num_classes).to(device)

        for images, labels in train_dataloader.generate_data():
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)

            for c in range(num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    class_centroids[c] += embeddings[mask].detach().sum(dim=0)
                    class_counts[c] += mask.sum()

            class_centroids /= class_counts[:, None] + 1e-8

            centroid_distances = F.pairwise_distance(
                embeddings[:, None], class_centroids, p=2
            )

            margin = 0.1
            loss = torch.mean(
                torch.clamp(torch.min(centroid_distances, dim=1)[0] - margin, min=0)
            )
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader.image_paths_known)

        model.eval()
        val_loss = 0.0
        known_distances = []
        unknown_distances = []

        with torch.no_grad():
            for images, labels in val_dataloader.generate_data():
                images, labels = images.to(device), labels.to(device)
                embeddings = model(images)

                centroid_distances = F.pairwise_distance(
                    embeddings[:, None], class_centroids, p=2
                )

                known_mask = labels != -1
                unknown_mask = labels == -1

                if known_mask.sum() > 0:
                    known_embeddings = embeddings[known_mask]
                    known_centroid_distances = F.pairwise_distance(
                        known_embeddings[:, None], class_centroids, p=2
                    )
                    loss_known = torch.mean(
                        torch.min(centroid_distances[labels != -1], dim=1)[0]
                    )
                    val_loss += loss_known.item()
                    known_distances.extend(
                        torch.min(known_centroid_distances, dim=1)[0].cpu().numpy()
                    )

                if unknown_mask.sum() > 0:
                    unknown_embeddings = embeddings[unknown_mask]
                    unknown_centroid_distances = F.pairwise_distance(
                        unknown_embeddings[:, None], class_centroids, p=2
                    )
                    loss_unknown = torch.mean(
                        torch.clamp(
                            margin
                            - torch.min(centroid_distances[labels == -1], dim=1)[0],
                            min=0,
                        )
                    )
                    val_loss += loss_unknown.item()
                    unknown_distances.extend(
                        torch.mean(unknown_centroid_distances, dim=1).cpu().numpy()
                    )

        val_loss /= len(val_dataloader.image_paths_known) + len(
            val_dataloader.image_paths_unknown
        )
        print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    mapping_ids = mapping(PATIENTS)
    split_data_known, split_data_unknown = split_open(
        PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED
    )
    train_loader = DataLoader(
        split_data_known, split_data_unknown, "train", mapping_ids
    )
    val_loader = DataLoader(split_data_known, split_data_unknown, "val", mapping_ids)
    test_loader = DataLoader(split_data_known, split_data_unknown, "test", mapping_ids)

    num_classes = 70
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model().to(device)

    history = train(
        model,
        train_loader,
        val_loader,
        num_classes,
        50,
        0.001,
        device,
    )
