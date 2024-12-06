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

    def __init__(
        self, device, size=(64, 60), record=False, frame_skip=4, rom_path="mario.nes"
    ):
        self.rom_path = rom_path
        self.record = record
        self.device = device
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
        self.done = False
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
        # print(f"{self.step.time}: c\t{self.step.x_pos[-1]}")
        return self.tensor.to(self.device)

    def get_metrics(self):
        lives = self.nes[0x75A]
        x_horizontal = self.nes[0x006D]
        x_on_screen = self.nes[0x0086]
        horizontal_speed = self.nes[0x0057]
        y_position_on_screen = self.nes[0x00CE]
        x_position = (x_horizontal << 8) | x_on_screen
        score_bcd = [
            self.nes[0x07DD],  # 1000000 and 100000 place
            self.nes[0x07DE],  # 10000 and 1000 place
            self.nes[0x07DF],  # 100 and 10 place
            self.nes[0x07E0],  # 1 place (if applicable)
            self.nes[0x07E1],  # 1 place (if applicable)
            self.nes[0x07E2],  # 1 place (if applicable)
        ]
        level = self.nes[0x0760]
        # Convert BCD to integer score
        score = 0
        for byte in score_bcd:
            score = score * 100 + ((byte >> 4) * 10) + (byte & 0x0F)

        self.step.step(
            x_position,
            y_position_on_screen,
            horizontal_speed,
            self.frame_skip,
            lives,
            score,
            level,
        )
        if lives != 2:
            self.alive = False
            self.done = True
        if level == 1:
            self.done = True

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
        return_controller = 0

        if controller[0] > 0 and controller[0] > controller[1]:
            return_controller |= NES_INPUT_RIGHT
        if controller[1] > 0 and controller[1] > controller[0]:
            return_controller |= NES_INPUT_LEFT
        if controller[2] > 0 and controller[2] > controller[3]:
            return_controller |= NES_INPUT_DOWN
        if controller[3] > 0 and controller[3] > controller[2]:
            return_controller |= NES_INPUT_UP
        if controller[4] > 0:
            return_controller |= NES_INPUT_A
        if controller[5] > 0:
            return_controller |= NES_INPUT_B

        return return_controller

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
