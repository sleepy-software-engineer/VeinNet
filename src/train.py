import torch
from torch import nn 
from utils.config import PATIENTS, DATASET_PATH
from utils.functions import split, mapping, prepare
from dataloader import DataLoader 
from model import Model
import os

def train(model, train_loader, val_loader, device, lr=0.001, patience=10, checkpoint_dir="./checkpoints"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_accuracy = 0.0  # Initialize the best validation accuracy
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    epochs_no_improve = 0  # Counter for early stopping
    epoch = 0  # To track the number of epochs

    while epochs_no_improve < patience:
        model.train()
        train_loss = 0.0
        for vein_image, label in train_loader.generate_data():
            vein_image = torch.tensor(vein_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            label = torch.tensor([int(label)], dtype=torch.long).to(device)

            optimizer.zero_grad()
            output = model(vein_image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for vein_image, label in val_loader.generate_data():
                vein_image = torch.tensor(vein_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                label = torch.tensor([int(label)], dtype=torch.long).to(device)

                output = model(vein_image)
                val_loss += criterion(output, label).item()
                pred = torch.argmax(output, dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        train_loss /= len(train_loader.image_paths)
        val_loss /= len(val_loader.image_paths)
        val_accuracy = correct / total

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Check for improvement
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Overwrite the best model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy
            }, best_model_path)
            print(f"New best model saved with Val Accuracy: {val_accuracy:.4f}")
            epochs_no_improve = 0  # Reset the counter
        else:
            epochs_no_improve += 1  # Increment the counter

        # Increment epoch count
        epoch += 1

    # Load the best model
    print(f"Training stopped. Best model loaded with Val Accuracy: {best_val_accuracy:.4f}")
    checkpoint = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

if __name__ == "__main__":
    mapping_ids = mapping(PATIENTS)
    split_data = split(PATIENTS, DATASET_PATH)

    train_loader = DataLoader(split_data, "train", mapping_ids)
    val_loader = DataLoader(split_data, "val", mapping_ids)

    train_images, train_labels = prepare(train_loader)
    val_images, val_labels = prepare(val_loader)

    num_classes = len(mapping_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes).to(device)
    train(model, train_loader, val_loader, device, patience=20)