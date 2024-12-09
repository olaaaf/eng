import torch
import torch.nn as nn
import torch.nn.functional as F
from train.helpers import Reward


class SimpleModel(nn.Module):
    def __init__(self, config: Reward | None = None, random_weights=True):
        super(SimpleModel, self).__init__()
        if not config:
            # Default values
            conv1_channels = 16
            conv2_channels = 32
            fc1_size = 128
            input_size = 3840  # 60 * 64
        else:
            settings = config.to_dict()
            conv1_channels = settings["conv1_channels"]
            conv2_channels = settings["conv2_channels"]
            fc1_size = settings["fc1_size"]
            input_size = settings["input_size"]

        # Conv layers - input is single frame [batch, 1, 60, 64]
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_channels)

        # Calculate output size after convolutions
        self.fc1 = nn.Linear(32 * 30 * 32, fc1_size)  # Adjusted for single frame
        self.fc2 = nn.Linear(fc1_size, 6)

        if random_weights:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [batch, 3840]
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 60, 64)  # Reshape to [batch, channels=1, height=60, width=64]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = x.view(batch_size, -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model
