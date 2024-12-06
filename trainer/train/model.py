import torch
import torch.nn as nn
import random


class SimpleModel(nn.Module):
    def __init__(self, fc1_size=256, fc2_size=64, input_size=3840, random_weights=True):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 6)  # 6 possible actions

        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

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
