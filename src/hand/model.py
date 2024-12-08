import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataloader import HandGeometryDataLoader
from utils.utils import split_images_by_patient, create_patient_id_mapping
from config import *
from sklearn.preprocessing import StandardScaler
import numpy as np

class HandGeometryModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HandGeometryModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)  # Add batch normalization
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)  # Add batch normalization
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
def train_hand_geometry_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, device="cuda"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_features, train_labels = train_loader
    val_features, val_labels = val_loader

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        output = model(train_features.to(device))
        loss = criterion(output, train_labels.to(device))
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_output = model(val_features.to(device))
            val_loss = criterion(val_output, val_labels.to(device))
            val_accuracy = (val_output.argmax(dim=1) == val_labels.to(device)).float().mean().item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, "
              f"Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

def prepare_normalized_data(dataloader, scaler=None):
    """
    Prepare and normalize data from the dataloader.
    Args:
        dataloader: Dataloader to extract features and labels.
        scaler: Scaler to normalize the data. If None, a new scaler will be created.
    Returns:
        np.ndarray: Normalized features.
        np.ndarray: Labels.
        StandardScaler: Fitted scaler.
    """
    features = []
    labels = []

    for feature, label in dataloader.generate_data():
        features.append(feature)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)  # Fit and transform training data
    else:
        features = scaler.transform(features)  # Transform validation/test data

    return features, labels, scaler 
    
if __name__ == "__main__":
    # Create patient ID mapping
    id_mapping = create_patient_id_mapping(PATIENTS)
    split_data = split_images_by_patient(PATIENTS, DATASET_PATH)

    # Create dataloaders for each split
    train_loader = HandGeometryDataLoader(split_data, "train", id_mapping)
    val_loader = HandGeometryDataLoader(split_data, "val", id_mapping)
    test_loader = HandGeometryDataLoader(split_data, "test", id_mapping)

  # Prepare and normalize data
    print("Preparing and normalizing data...")
    train_features, train_labels, scaler = prepare_normalized_data(train_loader)
    val_features, val_labels, _ = prepare_normalized_data(val_loader, scaler=scaler)
    test_features, test_labels, _ = prepare_normalized_data(test_loader, scaler=scaler)

    # Convert features and labels to PyTorch tensors for the neural network
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_features = torch.tensor(val_features, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Initialize the model
    input_dim = train_features.shape[1]  # Number of features (e.g., 11)
    num_classes = len(id_mapping)  # Number of patients
    model = HandGeometryModel(input_dim, num_classes).to("cuda")

    # Train the model
    train_hand_geometry_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, device="cuda")

    # Evaluate the model on the test set
    # evaluate_hand_geometry_model(model, test_loader, "cuda")
