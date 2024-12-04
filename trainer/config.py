from train.helpers import *
import json

if __name__ == "__main__":
    import logging

    logger = logging.getLogger("ConfigFileReward")
    logger.setLevel(logging.INFO)

    default_config = {
        "position_delta": 0.05,
        "score_delta": 0.005,
        "speed": 0.001,
        "finish": 200,
        "death": 40,
        "time_penalty": 0.05,
        "time_penalty_start": 5000,
    }

    model_id = input("Enter model_id: ").strip()

    config_path = "rewards.json"
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
    except FileNotFoundError:
        logger.info(f"{config_path} not found. Creating a new one.")
        config_data = {}
    if model_id in config_data:
        logger.info(f"Updating configuration for model_id: {model_id}")
    else:
        logger.info(f"Adding new configuration for model_id: {model_id}")
    config_data[model_id] = default_config
    with open(config_path, "w") as file:
        json.dump(config_data, file, indent=4)

    logger.info(f"Configuration for model_id {model_id} saved to {config_path}")
