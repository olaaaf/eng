from nes import NES
import matplotlib.pyplot as plt
import cv2


class Runner:
    def __init__(self, size=(64, 60), frame_skip=10, rom_path="../../mario.nes"):
        self.n = NES(rom_path, verbose=False, headless=True)
        self.n.run_frame_headless(
            run_frames=40, controller1_state=[0, 0, 0, 0, 0, 0, 0, 0]
        )
        self.n.run_frame_headless(
            run_frames=85, controller1_state=[0, 0, 0, 1, 0, 0, 0, 0]
        )
        self.buffer = n.run_frame_headless(
            run_frames=85, controller1_state=[0, 0, 0, 0, 0, 0, 0, 0]
        )
        self.size = size
        self.frame_skip = frame_skip

    def next(self, controller):
        self.__frame(controller)
        self.__scale_down()
        yield self.buffer

    def __scale_down(self):
        self.buffer = cv2.cvtColor(
            cv2.resize(self.buffer, self.size), cv2.COLOR_RGB2GRAY
        )

    def __frame(self, controller=[0, 0, 0, 0, 0, 0, 0, 0]):
        self.buffer = self.n.run_frame_headless(
            run_frames=self.frame_skip, controller1_state=controller
        )
