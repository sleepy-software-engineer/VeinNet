import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.fc_image = nn.Linear(32 * 32 * 32, 128)
        self.label_embedding = nn.Embedding(num_classes, 128)
        self.fc_final = nn.Linear(128 + 128, 1)

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(image))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        image_features = F.relu(self.fc_image(x))
        label_features = self.label_embedding(label)
        combined_features = torch.cat((image_features, label_features), dim=1)
        logits = self.fc_final(combined_features)

        return torch.sigmoid(logits).view(-1)
