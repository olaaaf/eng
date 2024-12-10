from cynes.windowed import WindowedNES
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
import sdl2
import time
import cv2
from PIL import Image

frame_count = 0

with WindowedNES("mario.nes") as nes:
    nes.step(frames=40)
    nes.controller = NES_INPUT_START
    nes.step(frames=85)
    nes.controller = 0
    nes.step(frames=85)
    last_x = 40
    current_x = 40

    while not nes.should_close:
        frame = nes.step()
        frame_count += 1
        if frame_count % 10 == 0:
            frameee = cv2.cvtColor(cv2.resize(frame, (64, 60)), cv2.COLOR_RGB2GRAY)
            img = Image.fromarray(frame)
            filename = f"frame_{frame_count}.png"
            filenameee = f"frameee_{frame_count}.png"
            img.save(filename)
            cv2.imwrite(filenameee, frameee)
            frame_count += 1
