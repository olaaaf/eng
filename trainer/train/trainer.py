import logging

import torch

from game.runner import Runner
from train.model import SimpleModel
from util.db_handler import DBHandler


class Trainer:
    def __init__(
        self,
        model_id: int,
        runner: Runner,
        model: SimpleModel,
        optimizer,
        db_handler: DBHandler,
        logger,
        random_weight_threshold=5,
    ):
        self.model: SimpleModel = model
        self.runner = runner
        self.model_id = model_id
        self.optimizer = optimizer
        self.db_handler = db_handler
        self.logger: logging.Logger = logger
        self.random_weight_threshold = (
            random_weight_threshold  # Threshold for applying random weights
        )
        self.episode_data = []

    def store_step(self, state, action, reward):
        self.episode_data.append((state, action, reward))

    def apply_random_weight_changes(self):
        for param in self.model.parameters():
            if param.requires_grad:
                # Apply small random noise to weights
                param.data += (
                    torch.randn_like(param) * 0.01
                )  # Adjust the multiplier for desired randomness

    async def evaluate(self):
        self.runner.reset()
        self.logger.info(f"Evaluating model {self.model_id}")
        # first frame without input
        inputs = [0 for _ in range(6)]
        tensor = self.runner.next(inputs)
        while self.runner.alive:
            inputs = self.model.forward(tensor)
            reward = (
                self.runner.get_reward()
            )  # Calculate reward based on the current state
            self.store_step(tensor, inputs, reward)
            tensor = self.runner.next(inputs.tolist())
        self.runner.step.save_to_db(self.model_id, self.db_handler)

    def apply_random_weight_changes(self, train_count):
        """
        Apply random weight changes with fading intensity based on training count.

        Args:
            train_count (int): Current number of training iterations
        """
        # Calculate fading factor (50% to 2% over 200 runs)
        max_random = 0.5  # 50% at start
        min_random = 0.02  # 2% at end
        fade_iterations = 200

        if train_count >= fade_iterations:
            current_random = min_random
        else:
            # Linear interpolation between max_random and min_random
            current_random = max_random - (max_random - min_random) * (
                train_count / fade_iterations
            )

        self.logger.info(
            f"Applying random weight changes with intensity: {current_random:.3f}"
        )

        # Apply random changes to each parameter
        with torch.no_grad():
            for param in self.model.parameters():
                # Generate random noise with the same shape as the parameter
                noise = torch.randn_like(param) * current_random
                # Add noise to the parameter
                param.add_(noise)

    async def train(self):
        # Check current training count from the database
        train_count = self.db_handler.get_train_count(self.model_id)
        self.logger.info(f"Training count for model {self.model_id}: {train_count}")

        # Always apply random weight changes, but with fading intensity
        self.apply_random_weight_changes(train_count)

        self.logger.info(f"Training model {self.model_id}")

        # Prepare data for training
        states = torch.stack([data[0] for data in self.episode_data])
        rewards = torch.tensor(
            [data[2] for data in self.episode_data],
            requires_grad=True,
            dtype=torch.float32,
        )

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        # Perform training
        self.optimizer.zero_grad()
        action_probs = self.model(states)
        loss = -torch.sum(torch.log(action_probs + 1e-10) * rewards.unsqueeze(1))
        loss.backward()

        # Apply gradients
        self.optimizer.step()

        # Increment training count
        self.db_handler.increase_train_count(self.model_id)

        # Clear episode data for the next run
        self.episode_data = []

        self.logger.info(
            f"Completed training for model {self.model_id}. "
            f"Loss: {loss.item():.4f}, "
            f"Training Count: {train_count + 1}"
        )

    def get_exploration_rate(self, train_count):
        """
        Calculate the current exploration rate based on training count.
        This can be used to adjust the model's exploration behavior alongside the random weights.

        Args:
            train_count (int): Current number of training iterations

        Returns:
            float: Current exploration rate
        """
        max_explore = 0.5  # 50% at start
        min_explore = 0.02  # 2% at end
        fade_iterations = 200

        if train_count >= fade_iterations:
            return min_explore

        # Linear interpolation between max_explore and min_explore
        return max_explore - (max_explore - min_explore) * (
            train_count / fade_iterations
        )

    async def ttrain(self):
        # Check current training count from the database
        train_count = self.db_handler.get_train_count(self.model_id)
        self.logger.info(f"Training count for model {self.model_id}: {train_count}")

        # Apply random weight changes if the training count is below the threshold
        if train_count < self.random_weight_threshold:
            self.logger.info(
                "Applying random weight changes due to low training count."
            )
            self.apply_random_weight_changes()

        self.logger.info(f"Training model {self.model_id}")

        # Prepare data for training
        states = torch.stack([data[0] for data in self.episode_data])
        # actions = torch.tensor(
        #     [data[1] for data in self.episode_data], dtype=torch.float32
        # )
        rewards = torch.tensor(
            [data[2] for data in self.episode_data],
            requires_grad=True,
            dtype=torch.float32,
        )

        # Normalize rewards (optional)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        # Perform training
        self.optimizer.zero_grad()
        action_probs = self.model(states)
        loss = -torch.sum(torch.log(action_probs + 1e-10) * rewards.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        # Increment training count
        self.db_handler.increase_train_count(self.model_id)

        # Clear episode data for the next run
        self.episode_data = []

        self.logger.info(
            f"Completed training for model {self.model_id}. Loss: {loss.item()}"
        )

    async def tttrain(self):
        self.logger.info(f"Training model {self.model_id}")

        # Prepare data for training
        # States and actions are already stored in `self.episode_data` by `store_step()`
        states = torch.stack([data[0] for data in self.episode_data])
        # actions = torch.tensor(
        #     [data[1] for data in self.episode_data], dtype=torch.float32
        # )
        rewards = torch.tensor(
            [data[2] for data in self.episode_data],
            requires_grad=True,
            dtype=torch.float32,
        )

        # Normalize rewards (optional, but can help with stable training)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        # Perform training
        self.optimizer.zero_grad()
        action_probs = self.model(states)

        # Compute the loss using a policy gradient-like approach
        loss = -torch.sum(torch.log(action_probs + 1e-10) * rewards.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        # Save model and optimizer state
        self.db_handler.save_model(self.model_id, self.model, self.optimizer)
        self.db_handler.increase_train_count(self.model_id)

        # Clear episode data for the next run
        self.episode_data = []

        self.logger.info(
            f"Completed training for model {self.model_id}. Loss: {loss.item()}"
        )
