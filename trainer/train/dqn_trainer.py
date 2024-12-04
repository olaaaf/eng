# train/dqn_trainer.py
import asyncio
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import wandb
from game.runner import Runner
from util.db_handler import DBHandler
from train.helpers import Reward


class DQNTrainer:
    def __init__(
        self,
        model_id: int,
        runner: Runner,
        model,
        optimizer,
        db_handler: DBHandler,
        reward_handler: Reward,
        epsilon_start: float = 1.0,
        episode: int = 0,
    ):
        # get the settings
        self.batch_size = reward_handler.to_dict()["batch_size"]
        self.gamma = reward_handler.to_dict()["gamma"]
        self.epsilon_end = reward_handler.to_dict()["epsilon_end"]
        self.epsilon_decay = reward_handler.to_dict()["epsilon_decay"]
        # load the epsilon value from earlier
        self.epsilon = epsilon_start

        self.run = wandb.init(
            project="mario_b",
            name=f"model_{model_id}",
            id=f"run_{model_id}",
            config={
                "model_id": model_id,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
            },
            resume="allow",
        )
        self.model_id = model_id
        self.runner = runner
        self.model = model
        self.optimizer = optimizer
        self.db_handler = db_handler
        self.logger = logging.getLogger(f"trainer_{model_id}")
        self.reward_handler = reward_handler

        self.episode_count = episode
        self.total_steps = 0

        # Add device initialization
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

    def select_action(self, state: torch.Tensor) -> List[float]:
        if torch.rand(1) < self.epsilon:
            return (
                (torch.rand(6) > 0.5).float().tolist()
            )  # Random actions for 6 buttons

        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.model(state.unsqueeze(0))
            # Threshold at 0.5 to get binary actions
            actions = (q_values.squeeze(0) > 0.5).float()
            return actions.cpu().tolist()  # Conv

    async def evaluate(self):
        """Single training episode"""
        episode_reward = 0
        state = self.runner.reset()
        episode_experiences = []  # Store experiences for this episode

        while not self.runner.done:
            # Select and perform action
            action = self.select_action(state)
            next_state = self.runner.next(controller=action)
            reward = self.reward_handler.get_reward(self.runner.step)
            done = not self.runner.alive

            # Store experience
            episode_experiences.append((state, action, reward, next_state, done))
            episode_reward += reward
            state = next_state
            self.total_steps += 1

            # Allow other tasks to run
            await asyncio.sleep(0)

        # Episode ended - process all experiences
        self.train(episode_experiences)

        # Update epsilon and save metrics
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

        # Log metrics
        metrics = {
            "reward": episode_reward,
            "finished": (1.0 if self.runner.alive else 0.0),
            "time": self.runner.step.time,
            "score": self.runner.step.score[-1],
            "epsilon": self.epsilon,
            "x_pos_dev": np.std(self.runner.step.x_pos),
            "horizontal_speed_avg": sum(self.runner.step.horizontal_speed)
            / len(self.runner.step.horizontal_speed),
            "horizontal_speed_dev": np.std(self.runner.step.horizontal_speed),
            "max_x": max(self.runner.step.x_pos),
        }
        self.run.log(metrics)

        if self.episode_count % 10 == 0:
            try:
                model_save_path = f"model_episode_{self.episode_count}.pt"
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    model_save_path,
                )
                artifact = wandb.Artifact(
                    name=f"model_checkpoint_{self.model_id}",
                    type="model",
                    description=f"Model checkpoint at episode {self.episode_count}",
                    metadata={"episode": self.episode_count},
                )
                artifact.add_file(model_save_path)
                self.run.log_artifact(artifact)

                os.remove(model_save_path)
            except Exception as e:
                self.logger.error(f"Failed to save model to wandb: {e}")

            try:
                self.db_handler.save_model_archive(
                    self.model_id,
                    self.model,
                    self.optimizer,
                )
            except Exception as e:
                self.logger.error(e)
            self.db_handler.increase_train_count(self.model_id)

    def train(self, experiences):
        """Train on all experiences from completed episode"""
        total_loss = 0

        for state, action, reward, next_state, done in experiences:
            # Convert to tensors
            state = torch.FloatTensor(state.to(torch.device("cpu"))).to(self.device)
            next_state = torch.FloatTensor(next_state.to(torch.device("cpu"))).to(
                self.device
            )
            reward = torch.FloatTensor([reward]).to(self.device)
            action = torch.FloatTensor(action).to(self.device)

            # Get current Q values
            current_q_values = self.model(state)
            current_q_value = (current_q_values * action).sum()  # For multi-action

            # Get next state values
            with torch.no_grad():
                next_q_values = self.model(next_state)
                next_q_value = next_q_values.max()
                if done:
                    next_q_value = 0.0

                expected_q_value = reward + (self.gamma * next_q_value)

            # Compute loss
            loss = F.smooth_l1_loss(current_q_value, expected_q_value)
            total_loss += loss.item()

            # Optimize
            self.optimizer.zero_grad()
            loss.requires_grad = True
            loss.backward()
            self.optimizer.step()

        # Log average loss for episode
        if experiences:
            self.run.log({"loss": total_loss / len(experiences)})

    def cleanup(self):
        """Cleanup resources"""
        if self.run:
            self.run.finish()
        self.db_handler.save_model(
            self.epsilon, self.model_id, self.model, self.optimizer, self.episode_count
        )
