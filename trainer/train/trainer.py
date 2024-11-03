import logging

import torch
import torch.optim as optim

from game.runner import Runner
from train.model import SimpleModel


class Trainer:
    def __init__(
        self,
        model_id: int,
        runner: Runner,
        model: SimpleModel,
        optimizer,
        db_handler,
        logger,
    ):
        self.model: SimpleModel = model
        self.runner = runner
        self.model_id = model_id
        self.optimizer = optimizer
        self.db_handler = db_handler
        self.logger: logging.Logger = logger
        self.episode_data = []

    def store_step(self, state, action):
        self.episode_data.append((state, action))

    async def evaluate(self):
        self.runner.reset()
        self.logger.info(f"Evaluating model {self.model_id}")
        # first frame without input
        inputs = [0 for _ in range(6)]
        tensor = self.runner.next(inputs)
        while self.runner.alive and self.runner.unfinished:
            inputs = self.model.forward(tensor)
            tensor = self.runner.next(inputs)
        self.runner.step.save_to_db(self.model_id, self.db_handler)

    async def train(self, final_score):
        self.logger.info(f"Training model {self.model_id}")

        # Prepare data for training
        states = torch.cat([data[0] for data in self.episode_data])
        actions = torch.cat([data[1] for data in self.episode_data])
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Perform training
        self.optimizer.zero_grad()
        action_probs = self.model(states)
        loss = -torch.sum(torch.log(action_probs) * rewards.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        # Save model and optimizer state
        self.db_handler.save_model(self.model_id, self.model, self.optimizer)

        # Clear episode data
        self.episode_data = []

        self.logger.info(
            f"Completed training for model {self.model_id}. Loss: {loss.item()}"
        )
