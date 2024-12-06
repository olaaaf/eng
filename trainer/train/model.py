import torch
import torch.nn as nn
import random


class SimpleModel(nn.Module):
    def __init__(self, input_size=3840, random_weights=True):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 6)  # 6 possible actions

        if random_weights:
            nn.init.kaiming_normal_(self.fc1.weight)
            nn.init.kaiming_normal_(self.fc2.weight)
            nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model
