from typing import List

import cv2
import numpy as np
from cynes import (
    NES,
    NES_INPUT_A,
    NES_INPUT_B,
    NES_INPUT_DOWN,
    NES_INPUT_LEFT,
    NES_INPUT_RIGHT,
    NES_INPUT_START,
    NES_INPUT_UP,
)
from torch import float32 as tf32
from torch import tensor

from game.eval import Step


class Runner:
    max_frames = 6000

    def __init__(self, size=(64, 60), record=False, frame_skip=4, rom_path="mario.nes"):
        self.rom_path = rom_path
        self.record = record
        self.size = size
        self.frame_skip = frame_skip
        self.reset()

    def reset(self):
        self.nes = NES(self.rom_path)
        self.nes.step(frames=40)
        self.nes.controller = NES_INPUT_START
        self.nes.step(frames=85)
        self.nes.controller = 0
        self.nes.step(frames=85)
        self.alive = True
        self.step = Step()
        if self.record:
            self.frames = np.ndarray(
                (self.size[0], self.size[1], Runner.max_frames), dtype=int
            )
            self.current_frame = 0
        return self.next()

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
        if lives != 2:
            self.alive = False

    def __scale_down(self):
        self.buffer = cv2.cvtColor(
            cv2.resize(self.buffer, self.size), cv2.COLOR_RGB2GRAY
        )
        if self.record:
            if self.current_frame < Runner.max_frames:
                self.frames[self.current_frame] = self.buffer
                self.current_frame += 1
        self.tensor = tensor(self.buffer, dtype=tf32).flatten()
        self.tensor /= 255.0

    def __frame(self, controller: int):
        self.nes.controller = controller
        self.buffer = self.nes.step(frames=self.frame_skip)

    def __convert_input(self, controller: List[int]) -> int:
        controller = [int(np.ceil(x)) for x in controller]
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

    def get_reward(self):
        # Base reward from position progress
        position_delta = (
            self.step.x_pos[-1] - self.step.x_pos[-2] if len(self.step.x_pos) > 1 else 0
        )
        reward = 0

        # Reward for moving right
        reward += position_delta * 0.1

        # Penalty for moving left or not moving
        if position_delta <= 0:
            reward -= 0.1

        # Speed bonus
        if self.step.horizontal_speed[-1] > 0:
            reward += 0.05

        # Large penalty for death
        if not self.alive:
            reward -= 10

        # Penalty for taking too long
        if self.step.time > 8000:
            reward -= 5

        return reward
