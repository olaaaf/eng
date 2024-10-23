from cynes import (
    NES,
    NES_INPUT_START,
    NES_INPUT_RIGHT,
    NES_INPUT_LEFT,
    NES_INPUT_UP,
    NES_INPUT_DOWN,
    NES_INPUT_A,
    NES_INPUT_B,
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from numpy import astype
from eval import Step
from typing import List
from torch import tensor
from torch import float32 as tf32


class Runner:
    def __init__(self, size=(64, 60), frame_skip=2, rom_path="mario.nes"):
        self.nes = NES(rom_path)
        self.nes.step(frames=40)
        self.nes.controller = NES_INPUT_START
        self.nes.step(frames=85)
        self.nes.controller = 0
        self.nes.step(frames=85)
        self.size = size
        self.frame_skip = frame_skip

        self.step = Step()

    def next(self, controller: List[int] = [0, 0, 0, 0, 0, 0]):
        c = self.__convert_input(controller)
        self.__frame(c)
        self.__scale_down()
        self.get_metrics()
        return self.tensor

    def get_metrics(self):
        lives = self.nes[0x75A]
        x_horizontal = self.nes[0x006D]
        x_on_screen = self.nes[0x0086]
        horizontal_speed = self.nes[0x0057]
        y_position_on_screen = self.nes[0x00CE]
        x_position = (x_horizontal << 8) | x_on_screen
        self.step.step(
            x_position, y_position_on_screen, horizontal_speed, self.frame_skip, lives
        )

    def __scale_down(self):
        self.buffer = cv2.cvtColor(
            cv2.resize(self.buffer, self.size), cv2.COLOR_RGB2GRAY
        )

        self.tensor = tensor(self.buffer, dtype=tf32).flatten()
        self.tensor /= 255.0

    def __frame(self, controller: int):
        self.nes.controller = controller
        self.buffer = self.nes.step(frames=self.frame_skip)

    def __convert_input(self, controller: List[int]) -> int:
        return (
            controller[0] * NES_INPUT_RIGHT
            | controller[1] * NES_INPUT_LEFT
            | controller[2] * NES_INPUT_DOWN
            | controller[3] * NES_INPUT_UP
            | controller[4] * NES_INPUT_A
            | controller[5] * NES_INPUT_B
        )

    def controller_to_text(self, controller):
        text = ""
        if controller[1]:
            text += "←"
        if controller[0]:
            text += "→"
        if controller[2]:
            text += "↓"
        if controller[3]:
            text += "↑"
        if controller[4]:
            text += "A"
        if controller[5]:
            text += "B"
        return text


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


r = Runner()
model = SimpleModel()
input = r.next()
print(input.size())
output = model.forward(input)
while not r.step.died:
    input = r.next(output)
    output = model.forward(input)

    print(f"output: {r.controller_to_text(output)}")

