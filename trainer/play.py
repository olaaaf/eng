from time import sleep
import torch
import numpy as np
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
from util.db_handler import DBHandler
from train.model import SimpleModel
import cv2

# Initialize database handler
db_handler = DBHandler()

model_id = int(input("Enter the model ID to use: "))
# List available model archives
model_archives = db_handler.get_model_archives(model_id)

print("Available model archives:")
for archive in model_archives:
    print(f"ID: {archive[0]}")

archive_id = int(input("Enter the archive ID to use: "))

# Load the model
model, _ = db_handler.load_model_arhive(archive_id)
model.eval()


# Function to convert model output to NES controller input
def convert_output_to_controller(output):
    return [
        int(output[0]),  # NES_INPUT_RIGHT
        int(output[1]),  # NES_INPUT_LEFT
        int(output[2]),  # NES_INPUT_DOWN
        int(output[3]),  # NES_INPUT_UP
        int(output[4]),  # NES_INPUT_A
        int(output[5]),  # NES_INPUT_B
    ]


# Function to select action without randomness
def select_action(state: torch.Tensor) -> list:
    with torch.no_grad():
        actions = model.forward(state)
        return actions.squeeze().tolist()


# Function to preprocess the frame
def preprocess_frame(frame, size=(64, 60)):
    frame = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_RGB2GRAY)
    frame = torch.tensor(frame, dtype=torch.float32).flatten() / 255.0
    return frame


def controller_to_text(controller):
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


with WindowedNES("mario.nes") as nes:
    nes.step(frames=40)
    nes.controller = NES_INPUT_START
    nes.step(frames=85)
    nes.controller = 0
    nes.step(frames=85)
    while not nes.should_close:
        lives = nes[0x75A]
        level = nes[0x0760]
        x_horizontal = nes[0x006D]
        x_on_screen = nes[0x0086]
        horizontal_speed = nes[0x0057]
        y_position_on_screen = nes[0x00CE]
        x_position = (x_horizontal << 8) | x_on_screen

        # Get the current frame buffer and preprocess it
        frame = nes.step()
        frame = np.array(frame)
        frame = preprocess_frame(frame)

        # Get the model's action
        action = select_action(frame)
        controller_input = convert_output_to_controller(action)

        # Set the controller input
        nes.controller = (
            controller_input[0] * NES_INPUT_RIGHT
            | controller_input[1] * NES_INPUT_LEFT
            | controller_input[2] * NES_INPUT_DOWN
            | controller_input[3] * NES_INPUT_UP
            | controller_input[4] * NES_INPUT_A
            | controller_input[5] * NES_INPUT_B
        )

        sleep(1 / 60)  # 60fps

        score_bcd = [
            nes[0x07DD],  # 1000000 and 100000 place
            nes[0x07DE],  # 10000 and 1000 place
            nes[0x07DF],  # 100 and 10 place
            nes[0x07E0],  # 1 place (if applicable)
            nes[0x07E1],  # 1 place (if applicable)
            nes[0x07E2],  # 1 place (if applicable)
        ]

        # Convert BCD to integer score
        score = 0
        for byte in score_bcd:
            score = score * 100 + ((byte >> 4) * 10) + (byte & 0x0F)

        print(
            f"{controller_to_text(action)},pos: {x_position}, level: {level}, score: {score}, horizontal_speed: {horizontal_speed}"
        )
