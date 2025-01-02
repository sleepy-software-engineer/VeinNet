import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch_optimizer import Lookahead, RAdam

from dataloader import DataLoader
from model import Model
from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    lr: int,
    patience: int,
    checkpoint_dir: str,
) -> None:
    radam_optimizer = RAdam(model.parameters(), lr=lr, weight_decay=5e-4)
    optimizer = Lookahead(radam_optimizer, k=5, alpha=0.5)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "model.pth")

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    epoch = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    while epochs_no_improve < patience:
        model.train()
        train_loss = 0.0
        correct_train, total_train = 0, 0

        for vein_image, label in train_loader.generate_data():
            vein_image = (
                torch.tensor(vein_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            label = torch.tensor([int(label)], dtype=torch.long).to(device)

            optimizer.zero_grad()
            output = model(vein_image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            correct_train += (pred == label).sum().item()
            total_train += label.size(0)

        train_loss /= len(train_loader.image_paths)
        train_accuracy = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val, total_val = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for vein_image, label in val_loader.generate_data():
                vein_image = (
                    torch.tensor(vein_image, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                label = torch.tensor([int(label)], dtype=torch.long).to(device)

                output = model(vein_image)
                val_loss += criterion(output, label).item()
                all_preds.append(output.cpu().numpy().flatten())
                all_labels.append(label.item())
                pred = torch.argmax(output, dim=1)
                correct_val += (pred == label).sum().item()
                total_val += label.size(0)

        val_loss /= len(val_loader.image_paths)
        val_accuracy = correct_val / total_val

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            all_labels,
            np.argmax(all_preds, axis=1),
            average="weighted",
            zero_division=0,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            best_metrics = {
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Train Accuracy": train_accuracy,
                "Val Accuracy": val_accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score,
            }
            best_metrics_df = pd.DataFrame([best_metrics])
            best_metrics_df.to_csv(
                os.path.join(checkpoint_dir, "best_metrics.csv"), index=False
            )

            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            print(f"New best model saved with Val Accuracy: {val_accuracy:.4f}")
        else:
            epochs_no_improve += 1
        epoch += 1

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(
        range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy"
    )
    plt.plot(
        range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, "accuracy_curve.png"))
    plt.close()


if __name__ == "__main__":
    mapping_ids = mapping(PATIENTS)
    split_data = split(PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED)

    train_loader = DataLoader(split_data, "train", mapping_ids)
    val_loader = DataLoader(split_data, "val", mapping_ids)

    num_classes = len(mapping_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes).to(device)
    train(
        model,
        train_loader,
        val_loader,
        device,
        lr=0.001,
        patience=200,
        checkpoint_dir="./model",
    )
