import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dataloader import DataLoader
from model import Model
from scipy.interpolate import interp1d
from torch import nn
from torch_optimizer import Lookahead, RAdam

from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import mapping, split


def plot_test_metrics(cmc, ranks, test_directory):
    interp_fn = interp1d(ranks, cmc, kind="quadratic")
    smooth_ranks = np.linspace(min(ranks), max(ranks), 500)
    smooth_cmc = interp_fn(smooth_ranks)

    plt.figure()
    plt.plot(smooth_ranks, smooth_cmc, label="CMC Curve")
    plt.xlabel("Rank")
    plt.ylabel("CMC")
    plt.title("Cumulative Match Characteristic (CMC)")
    plt.savefig(os.path.realpath(os.path.join(test_directory, "cmc.png")))


def plot_training_metrics(
    train_losses,
    val_losses,
    train_recognition_rates,
    val_recognition_rates,
    cmc_values,
    ranks,
    cmc_history,
    train_directory,
):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.realpath(os.path.join(train_directory, "loss.png")))

    plt.figure()
    plt.plot(train_recognition_rates, label="Train Recognition Rate")
    plt.plot(val_recognition_rates, label="Validation Recognition Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Recognition Rate")
    plt.title("Train vs Validation Recognition Rate")
    plt.legend()
    plt.savefig(os.path.realpath(os.path.join(train_directory, "recognition_rate.png")))

    for _, cmc in enumerate(cmc_values):
        plt.figure()
        interp_fn = interp1d(ranks, cmc, kind="quadratic")
        smooth_ranks = np.linspace(min(ranks), max(ranks), 500)
        smooth_cmc = interp_fn(smooth_ranks)
        plt.plot(smooth_ranks, smooth_cmc)
    plt.xlabel("Rank")
    plt.ylabel("CMC")
    plt.title("Cumulative Match Characteristic (CMC)")
    plt.savefig(os.path.realpath(os.path.join(train_directory, "cmc.png")))

    plt.figure()
    for record in cmc_history:
        cmc = record["cmc"]
        interp_fn = interp1d(ranks, cmc, kind="quadratic")
        smooth_ranks = np.linspace(min(ranks), max(ranks), 500)
        smooth_cmc = interp_fn(smooth_ranks)
        plt.plot(smooth_ranks, smooth_cmc)
    plt.xlabel("Rank")
    plt.ylabel("CMC")
    plt.title("CMC Over Epochs")
    plt.savefig(os.path.realpath(os.path.join(train_directory, "cmc_history.png")))


def compute_cmc(predictions, labels, ranks):
    predictions = np.vstack(predictions)
    labels = np.hstack(labels)
    num_samples = len(labels)

    sorted_indices = np.argsort(-predictions, axis=1)

    cmc = np.zeros(len(ranks))

    for i, rank in enumerate(ranks):
        for j in range(num_samples):
            if labels[j] in sorted_indices[j, :rank]:
                cmc[i] += 1

        cmc[i] /= num_samples

    return cmc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: int,
    patience: int,
    train_directory: str,
) -> None:
    radam_optimizer = RAdam(model.parameters(), lr=lr, weight_decay=5e-4)
    optimizer = Lookahead(radam_optimizer, k=5, alpha=0.5)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = os.path.realpath(os.path.join(train_directory, "model.pth"))
    best_val_recognition_rate, epochs_no_improve, epoch = 0.0, 0, 0

    train_losses, val_losses = [], []
    train_recognition_rates, val_recognition_rates = [], []
    cmc_values = []
    cmc_history = []
    ranks = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

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

        if (epoch + 1) % 10 == 0:
            cmc = compute_cmc(all_preds, all_labels, ranks)
            cmc_history.append({"epoch": epoch + 1, "cmc": cmc})

        cmc = compute_cmc(all_preds, all_labels, ranks)
        cmc_values.append(cmc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_recognition_rates.append(train_recognition_rate)
        val_recognition_rates.append(val_recognition_rate)

        if val_recognition_rate > best_val_recognition_rate:
            best_val_recognition_rate = val_recognition_rate

            best_metrics = {
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Train Recognition Rate": train_recognition_rate,
                "Val Recognition Rate": val_recognition_rate,
                "CMC": cmc,
            }
            best_metrics_df = pd.DataFrame([best_metrics])
            best_metrics_df.to_csv(
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

    plot_training_metrics(
        train_losses,
        val_losses,
        train_recognition_rates,
        val_recognition_rates,
        cmc_values,
        ranks,
        cmc_history,
        train_directory,
    )


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
    ranks = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
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

    cmc = compute_cmc(all_scores, all_labels, ranks)

    metrics = {
        "CMC": cmc,
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        os.path.realpath(os.path.join(test_dir, "metrics.csv")),
        index=False,
    )

    plot_test_metrics(cmc, ranks, test_dir)


if __name__ == "__main__":
    mapping_ids = mapping(PATIENTS)
    split_data = split(PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED)

    train_loader = DataLoader(split_data, "train", mapping_ids)
    val_loader = DataLoader(split_data, "val", mapping_ids)
    test_loader = DataLoader(split_data, "test", mapping_ids)

    num_classes = len(mapping_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes).to(device)

    train(
        model,
        train_loader,
        val_loader,
        0.001,
        50,
        "./src/identification/closed/model/train",
    )

    # test(
    #     model,
    #     test_loader,
    #     device,
    #     "./src/identification/closed/model/train/model.pth",
    #     "./src/identification/closed/model/test",
    # )
