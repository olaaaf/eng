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
        self,
        device,
        size=(64, 60),
        record=False,
        frame_skip=4,
        rom_path="mario.nes",
        video_save_path=".",
        video_prefix="",
    ):
        self.rom_path = rom_path
        self.record = record
        self.device = device
        self.size = size
        self.frame_skip = frame_skip
        self.frames = []  # Store frames in memory
        self.record = record
        if self.record:
            self.video_save_path = video_save_path
            self.video_prefix = video_prefix
            # Store frames at original NES resolution (256x240)
            self.frames = np.ndarray((Runner.max_frames, 240, 256, 3), dtype=np.uint8)
            self.current_frame = 0
        self.reset()

    def reset(self):
        record_tmp = self.record
        self.record = False
        self.nes = NES(self.rom_path)
        self.nes.step(frames=40)
        self.nes.controller = NES_INPUT_START
        self.nes.step(frames=85)
        self.nes.controller = 0
        self.nes.step(frames=85)
        self.record = record_tmp
        self.alive = True
        self.step = Step()
        self.done = False
        if self.record:
            self.frames = np.ndarray((Runner.max_frames, 240, 256, 3), dtype=np.uint8)
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
        lives, level = self.step.step(self.nes, self.frame_skip)

        if lives != 2:
            self.alive = False
            self.done = True
            if self.record:
                self.save_video()
        if level == 1:
            self.done = True
            if self.record:
                self.save_video()

    def __scale_down(self):
        # Store original size frame for recording
        if self.record:
            frame_rgb = cv2.cvtColor(self.buffer, cv2.COLOR_RGB2BGR)
            self.frames[self.current_frame] = frame_rgb
            self.current_frame += 1

        # Scale down for ML processing
        frame_rgb = cv2.cvtColor(cv2.resize(self.buffer, self.size), cv2.COLOR_RGB2BGR)
        self.buffer = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        self.tensor = tensor(self.buffer, dtype=tf32)
        self.tensor /= 255.0

    def save_video(self):
        if not self.record:
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Fix dimensions to match NES resolution (256x240)
        out = cv2.VideoWriter(
            f"{self.video_save_path}/{self.video_prefix}_{max(self.step.x_pos)}_gameplay.mp4",
            fourcc,
            30.0,
            (256, 240),
        )

        for frame in range(self.current_frame):
            out.write(self.frames[frame])

        out.release()
        self.frames = []  # Clear memory

    def __frame(self, controller: int):
        self.nes.controller = controller
        self.buffer = self.nes.step(frames=self.frame_skip)

    def __convert_input(self, controller: List[float]) -> int:
        return_controller = 0

        # Use meaningful thresholds to determine action
        if controller[0] > 0.5:  # Right
            return_controller |= NES_INPUT_RIGHT
        if controller[1] > 0.5:  # Left
            return_controller |= NES_INPUT_LEFT
        if controller[2] > 0.5:  # Down
            return_controller |= NES_INPUT_DOWN
        if controller[3] > 0.5:  # Up
            return_controller |= NES_INPUT_UP
        if controller[4] > 0.5:  # A button
            return_controller |= NES_INPUT_A
        if controller[5] > 0.5:  # B button
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
