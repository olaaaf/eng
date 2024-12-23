from train.helpers import *
import json

import logging

logger = logging.getLogger("ConfigFileReward")
logger.setLevel(logging.INFO)


def create_default(model_id):
    config_path = "rewards.json"
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
    except FileNotFoundError:
        logger.info(f"{config_path} not found. Creating a new one.")
        config_data = {}
    if str(model_id) in config_data:
        logger.info(f"{model_id} in config already exists, omitting")
        return

    default_config = {
        "position_delta": 0.5,
        "score_delta": 0.05,
        "speed": 0.00125,
        "finish": 10,
        "death": -5,
        "position_reward": 1,
        "time_penalty": 0.00,
        "time_penalty_start": 9000,
        "stomp": 1,
        "beat_x_highscore": 50,
        "beat_score_highscore": 20,
        "beat_time_highscore": 150,
        "config": {
            "gamma": 0.97,
            "batch_size": 64,
            "epsilon_end": 0.01,
            "weight_decay": 1e-5,
            "epsilon_decay": 0.999,
            "tau": 0.01,
            "learning_rate": 1e-4,
            "fc1_size": 128,
            "conv1_channels": 16,
            "conv2_channels": 32,
            "input_size": 3840,
        },
    }
    logger.info(f"Adding new configuration for model_id: {model_id}")
    config_data[model_id] = default_config
    with open(config_path, "w") as file:
        json.dump(config_data, file, indent=4)

    logger.info(f"Configuration for model_id {model_id} saved to {config_path}")


if __name__ == "__main__":
    model_id = input("Enter model_id: ").strip()
    create_default(model_id)
