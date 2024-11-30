import torch
from dataloader import CombinedDataLoader
from hand.dataloader import HandGeometryDataLoader
from palm.dataloader import VeinImageDataLoader

DATASET_PATH = "/home/lucian/University/MSc-Courses/BiometricSystems/data/"
PATIENTS = [f"{i:03}" for i in range(1, 101)] 
HAND = "l"  
SPECTRUM = "940" 

# Example model structure
class MultimodalModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(MultimodalModel, self).__init__()
        # CNN for vein images
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 64 * 64, 128),  # Adjust for input size
            torch.nn.ReLU(),
        )

        # Fully connected layers for hand geometry features
        self.fc_geometry = torch.nn.Sequential(
            torch.nn.Linear(num_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
        )

        # Fusion layer and classification head
        self.fc_combined = torch.nn.Sequential(
            torch.nn.Linear(128 + 128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, vein_images, hand_features):
        # Reshape vein_images to combine batch and image dimensions
        # From [batch_size, 6, 1, 128, 128] to [batch_size * 6, 1, 128, 128]
        vein_images = vein_images.view(-1, 1, 128, 128)

        # Process vein images through the CNN
        vein_features = self.cnn(vein_images)  # Output shape: [batch_size * 6, 128]

        # Process hand geometry features
        # Reshape hand_features to match the CNN output
        # From [batch_size, 6, num_features] to [batch_size * 6, num_features]
        hand_features = hand_features.view(-1, hand_features.size(-1))
        geometry_features = self.fc_geometry(hand_features)  # Output shape: [batch_size * 6, 128]

        # Concatenate features from both branches
        combined = torch.cat((vein_features, geometry_features), dim=1)  # Shape: [batch_size * 6, 256]

        # Final classification
        output = self.fc_combined(combined)  # Shape: [batch_size * 6, num_classes]

        return output


# Initialize the model
model = MultimodalModel(num_features=11, num_classes=10)  # Adjust `num_classes` as needed
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

vein_loader = VeinImageDataLoader(dataset_dir=DATASET_PATH, batch_size=2)
geometry_loader = HandGeometryDataLoader(dataset_dir=DATASET_PATH, batch_size=2)
combined_loader = CombinedDataLoader(vein_loader, geometry_loader, batch_size=2)

for epoch in range(10):  # Example: 10 epochs
    for vein_image_batch, hand_geometry_batch, patient_ids in combined_loader.generate_batches(hand=HAND, spectrum=SPECTRUM):
        # Convert batches to PyTorch tensors
        vein_images = torch.tensor(vein_image_batch, dtype=torch.float32)
        hand_features = torch.tensor(hand_geometry_batch, dtype=torch.float32)
        labels = torch.tensor([0] * len(patient_ids), dtype=torch.long)  # Replace with actual labels

        # Expand labels to match reshaped batch size
        labels = labels.repeat_interleave(6)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(vein_images, hand_features)

        # Compute loss and backpropagate
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

