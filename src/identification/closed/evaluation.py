import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dataloader import DataLoader
from model import Model
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch_optimizer import Lookahead, RAdam

from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split_closed


def plot_cmc_curve(all_scores, all_labels, title="CMC Curve"):
    num_classes = all_scores.shape[1]
    num_samples = len(all_labels)
    sorted_indices = np.argsort(-all_scores, axis=1)
    rank_matches = np.zeros(num_classes)

    for i in range(num_samples):
        rank = np.where(sorted_indices[i] == all_labels[i])[0][0]
        rank_matches[rank:] += 1

    cmc = rank_matches / num_samples

    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, num_classes + 1), cmc, linestyle="-", linewidth=3, label="CMC Curve"
    )
    plt.xlabel("Rank")
    plt.ylabel("Prob. of Identification")
    plt.title(title)
    plt.grid()

    rank_1_rate = cmc[0]
    plt.annotate(
        f"Rank-1: {rank_1_rate:.2%}",
        xy=(1, rank_1_rate),
        xytext=(5, rank_1_rate - 0.1),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        fontsize=10,
        ha="center",
    )

    plt.tight_layout()
    plt.savefig("cmc_curve.png")


def plot_confusion_matrix(true_labels, predicted_labels, title="Confusion Matrix"):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    TN, FP, FN, TP = conf_matrix.ravel()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=["Mismatch", "Match"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.savefig("confusion_matrix.png")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: int,
    patience: int,
    train_directory: str,
) -> None:
    radam_optimizer = RAdam(model.parameters(), lr=lr, weight_decay=5e-4)
    optimizer = Lookahead(radam_optimizer, k=10, alpha=0.5)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = os.path.realpath(os.path.join(train_directory, "model.pth"))
    best_val_recognition_rate, epochs_no_improve, epoch = 0.0, 0, 0

    while epochs_no_improve < patience:
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for vein_image, label in train_loader.generate_data():
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
        train_recognition_rate = correct_train / total_train

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for vein_image, label in val_loader.generate_data():
                output = model(vein_image)
                val_loss += criterion(output, label).item()
                all_preds.append(output.cpu().numpy())
                all_labels.append(label.cpu().numpy())
                pred = torch.argmax(output, dim=1)
                correct_val += (pred == label).sum().item()
                total_val += label.size(0)

        val_loss /= len(val_loader.image_paths)
        val_recognition_rate = correct_val / total_val

        if val_recognition_rate > best_val_recognition_rate:
            best_val_recognition_rate = val_recognition_rate

            metrics = {
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Train Recognition Rate": train_recognition_rate,
                "Val Recognition Rate": val_recognition_rate,
            }
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(
                os.path.realpath(os.path.join(train_directory, "metrics.csv")),
                index=False,
            )
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
            print(
                f"New best model saved with Recognition Rate: {val_recognition_rate:.4f}"
            )
        else:
            epochs_no_improve += 1
        epoch += 1


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    train_dir: str,
    test_dir: str,
) -> None:
    model.load_state_dict(torch.load(train_dir, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_scores = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for vein_image, label in test_loader.generate_data():
            vein_image = vein_image.to(device)
            label = label.to(device)

            output = model(vein_image)
            test_loss += criterion(output, label).item()
            pred_scores = output.cpu().numpy()

            all_preds.append(np.argmax(pred_scores, axis=1))
            all_scores.append(pred_scores)
            all_labels.append(label.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_scores = np.vstack(all_scores)
    all_labels = np.concatenate(all_labels, axis=0)

    binary_labels = (all_labels == all_preds).astype(int)
    binary_preds = binary_labels

    plot_confusion_matrix(
        true_labels=binary_labels,
        predicted_labels=binary_preds,
        title="Confusion Matrix",
    )

    plot_cmc_curve(all_scores, all_labels, title="CMC Curve")


if __name__ == "__main__":
    mapping_ids = mapping(PATIENTS)
    split_data = split_closed(PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED)

    train_loader = DataLoader(split_data, "train", mapping_ids)
    val_loader = DataLoader(split_data, "val", mapping_ids)
    test_loader = DataLoader(split_data, "test", mapping_ids)

    num_classes = len(mapping_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes).to(device)

    # train(
    #     model,
    #     train_loader,
    #     val_loader,
    #     0.001,
    #     50,
    #     "./src/identification/closed/model/train",
    # )

    test(
        model,
        test_loader,
        device,
        "./src/identification/closed/model/train/model.pth",
        "./src/identification/closed/model/test",
    )
