import torch
from torch_optimizer import RAdam, Lookahead
from torch import nn 
from utils.config import PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED
from utils.functions import split, mapping
from dataloader import DataLoader 
from model import Model
import os

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, lr: int, patience: int, checkpoint_dir: str) -> None:
    radam_optimizer = RAdam(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = Lookahead(radam_optimizer, k=5, alpha=0.5) 
    criterion = nn.CrossEntropyLoss() 

    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_accuracy = 0.0  
    best_model_path = os.path.join(checkpoint_dir, "model.pth")
    epochs_no_improve = 0 
    epoch = 0

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

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'val_accuracy': val_accuracy
            }, best_model_path)
            print(f"New best model saved with Val Accuracy: {val_accuracy:.4f}")
            epochs_no_improve = 0 
        else:
            epochs_no_improve += 1
        epoch += 1

if __name__ == "__main__":
    mapping_ids = mapping(PATIENTS)
    split_data = split(PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED)

    train_loader = DataLoader(split_data, "train", mapping_ids)
    val_loader = DataLoader(split_data, "val", mapping_ids)

    num_classes = len(mapping_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes).to(device)
    train(model, train_loader, val_loader, device, lr=0.001, patience=50, checkpoint_dir="./model")