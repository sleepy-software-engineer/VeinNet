import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    A convolutional neural network model for classification tasks.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initializes the Model class.
        """
        # Call the parent class (nn.Module) constructor
        super(Model, self).__init__()

        # Define the first convolutional layer (input channels: 1, output channels: 16, kernel size: 3x3, stride: 1, padding: 1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        # Define the second convolutional layer (input channels: 16, output channels: 32, kernel size: 3x3, stride: 1, padding: 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Define the third convolutional layer (input channels: 32, output channels: 64, kernel size: 3x3, stride: 1, padding: 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Define the fourth convolutional layer (input channels: 64, output channels: 128, kernel size: 3x3, stride: 1, padding: 1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Define the first fully connected layer (input features: 32*32*32, output features: 128)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)

        # Define a dropout layer with a dropout probability of 0.5
        self.dropout = nn.Dropout(0.5)

        # Define the second fully connected layer (input features: 128, output features: num_classes)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the neural network.
        """
        # Apply the first convolutional layer followed by ReLU activation
        x = F.relu(self.conv1(x))

        # Apply max pooling with a kernel size of 2
        x = F.max_pool2d(x, 2)

        # Apply the second convolutional layer followed by ReLU activation
        x = F.relu(self.conv2(x))

        # Apply max pooling with a kernel size of 2
        x = F.max_pool2d(x, 2)

        # Apply the third convolutional layer followed by ReLU activation
        x = F.relu(self.conv3(x))

        # Apply max pooling with a kernel size of 2
        x = F.max_pool2d(x, 2)

        # Apply the fourth convolutional layer followed by ReLU activation
        x = F.relu(self.conv4(x))

        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the first fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout
        x = self.dropout(x)

        # Apply the second fully connected layer to get the final output
        x = self.fc2(x)

        return x
