import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A convolutional neural network model for classification tasks.

    Args:
        num_classes (int): Number of output classes.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        conv4 (nn.Conv2d): Fourth convolutional layer.
        fc1 (nn.Linear): Fully connected layer.
        dropout (nn.Dropout): Dropout layer.
        fc2 (nn.Linear): Output fully connected layer.

    Methods:
        forward(x):
            Defines the forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after passing through the network.
    """
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.conv1   = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2   = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3   = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4   = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1     = nn.Linear(32 * 32 * 32, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
