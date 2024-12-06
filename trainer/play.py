from time import sleep
import torch
import numpy as np
import wandb
import os
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
from typing import List
from train.model import SimpleModel
import cv2
import matplotlib.pyplot as plt

plt.ion()  # Turn on interactive mode

# Initialize database handler
db_handler = DBHandler()

model_id = int(input("Enter the model ID to use: "))
source = input("Load from (1) Local DB or (2) Wandb? Enter 1 or 2: ")

model = None
if source == "1":
    # List available model archives
    model_archives = db_handler.get_model_archives(model_id)

    print("Available model archives:")
    for archive in model_archives:
        print(f"ID: {archive[0]}")

    archive_id = int(input("Enter the archive ID to use: "))

    # Load model from local DB
    model, _ = db_handler.load_model_arhive(archive_id)

elif source == "2":
    # Initialize wandb
    run = wandb.init()
    # Download the artifact
    version = input("input version: ")
    artifact = run.use_artifact(
        f"olafercik/mario_advanced_dqn/advanced_model_checkpoint_{model_id}:v{version}",
        type="model",
    )
    # artifact = run.use_artifact(
    #     "olafercik/mario_advanced_dqn/advanced_model_checkpoint_6:v28", type="model"
    # )
    artifact_dir = artifact.download()
    # Load the model
    filee = os.listdir(artifact_dir)[0]
    checkpoint = torch.load(f"{artifact_dir}/{filee}", map_location=torch.device("cpu"))
    model = SimpleModel()
    model.load_state_dict(checkpoint["model_state_dict"])

if model is None:
    print("Failed to load model")
    exit(1)

model.eval()


def convert_output_to_controller(controller: List[int]) -> int:
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


# Function to select action without randomness
def select_action(state: torch.Tensor) -> List[float]:
    with torch.no_grad():
        state = state.view(1, -1)
        # Use get_actions to match original implementation
        actions = model.forward(state)
        print(actions)

        return (
            (actions).float().squeeze().cpu().tolist()
        )  # Element-wise comparison to produce 0s and 1s


# Function to preprocess the frame
def preprocess_frame(frame, size=(64, 60)):
    # Scale down and convert to grayscale
    scaled_frame = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_RGB2GRAY)

    # Convert to tensor and normalize
    frame_tensor = torch.tensor(scaled_frame, dtype=torch.float32).flatten() / 255.0

    # Display frame using OpenCV
    cv2.namedWindow("Model Input", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Model Input", 320, 300)  # Larger window for better visibility
    cv2.imshow("Model Input", scaled_frame)
    cv2.waitKey(1)  # Small delay to update window

    return frame_tensor, scaled_frame


def controller_to_text(controller):
    text = ""
    if controller & NES_INPUT_LEFT:
        text += "←"
    if controller & NES_INPUT_RIGHT:
        text += "→"
    if controller & NES_INPUT_DOWN:
        text += "↓"
    if controller & NES_INPUT_UP:
        text += "↑"
    if controller & NES_INPUT_A:
        text += "A"
    if controller & NES_INPUT_B:
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
        frame = nes.step()
        frame = nes.step()
        frame = nes.step()
        frame, model_input_frame = preprocess_frame(frame)

        # Get the model's action
        action = select_action(frame)
        nes.controller = convert_output_to_controller(action)

        # Set the controller input
        if lives != 2:
            nes.should_close = True

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
            f"{controller_to_text(nes.controller)},pos: {x_position}, level: {level}, score: {score}, horizontal_speed: {horizontal_speed}"
        )

cv2.destroyAllWindows()
