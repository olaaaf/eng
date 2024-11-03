import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, random_weights=True):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3840, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)
        if random_weights:
            nn.init.xavier_uniform(self.fc1.weight)
            nn.init.xavier_uniform(self.fc2.weight)
            nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return [1 if a > 0.5 else 0 for a in torch.sigmoid(self.fc3(x))]

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model
