import torch.nn as nn
import torch.nn.functional as F


class GeoGuessrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 256, 5)
        self.conv2 = nn.Conv2d(256, 128, 3)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 4)

        """
        The model outputs 4 neurons; each signifying a confidence score for where the image is in the corresponding country:
        0: Ghana,
        1: Japan,
        2: Malaysia
        3: United States
        """

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = F.sigmoid(self.conv2(x))

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)

        return x