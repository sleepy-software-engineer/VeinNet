import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from dataloader import VeinImageDataLoader
from dataloader import SplitData

class SimpleBiometricCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleBiometricCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout layer
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x


def prepare_dataset(dataloader):
    images = []
    labels = []
    for vein_image, patient_id in dataloader.generate_data():
        images.append(torch.tensor(vein_image, dtype=torch.float32).unsqueeze(0))  # Add channel dimension
        labels.append(int(patient_id))  # Convert patient ID to integer label
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for vein_image, label in train_loader.generate_data():
            # Prepare inputs
            vein_image = torch.tensor(vein_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # Add batch and channel dims
            label = torch.tensor([int(label)], dtype=torch.long).to(device)  # Single label as a batch

            # Forward pass
            optimizer.zero_grad()
            output = model(vein_image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
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

        # Calculate average losses and accuracy
        train_loss /= len(train_loader.image_paths)
        val_loss /= len(val_loader.image_paths)
        val_accuracy = correct / total

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

def evaluate_model_on_test(model, test_loader, device):
    """
    Evaluate the model on the test set.
    Args:
        model: Trained model.
        test_loader: Test dataloader.
        device: Device (CPU or GPU).
    Returns:
        float: Test accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for vein_image, label in test_loader.generate_data():
            # Move data to the appropriate device
            vein_image = torch.tensor(vein_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            label = int(label)

            # Forward pass
            output = model(vein_image)
            pred = torch.argmax(output, dim=1).item()

            # Collect true and predicted labels
            y_true.append(label)
            y_pred.append(pred)

            # Accuracy
            correct += (pred == label)
            total += 1

    # Calculate accuracy
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=1))

    return test_accuracy 

if __name__ == "__main__":
    s = SplitData()
    id_mapping = s.create_patient_id_mapping(PATIENTS)
    split_data = s.split_images_by_patient(PATIENTS, DATASET_PATH)
    train_loader = VeinImageDataLoader(split_data, "train", id_mapping)
    val_loader = VeinImageDataLoader(split_data, "val", id_mapping)
    test_loader = VeinImageDataLoader(split_data, "test", id_mapping)
    
    # Prepare the dataset
    train_images, train_labels = prepare_dataset(train_loader)
    val_images, val_labels = prepare_dataset(val_loader)
    
    # Initialize the model
    num_classes = len(id_mapping)
    # Select device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleBiometricCNN(num_classes).to(device)
    
    # Train the model
    train_model(model, train_loader, val_loader)
    
    from sklearn.metrics import confusion_matrix, classification_report

    # After validation phase
    y_true = []  # True labels
    y_pred = []  # Predicted labels

    model.eval()
    with torch.no_grad():
        for vein_image, label in val_loader.generate_data():
            vein_image = torch.tensor(vein_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            label = int(label)
            output = model(vein_image)
            pred = torch.argmax(output, dim=1).item()
            y_true.append(label)
            y_pred.append(pred)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    # Classification report
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=1))
       
    test_accuracy = evaluate_model_on_test(model, test_loader, device)