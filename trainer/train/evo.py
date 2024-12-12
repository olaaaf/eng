import numpy as np
import torch
import torch.nn.functional as F
import wandb
from game.runner import Runner
from train.model import SimpleModel
from train.helpers import Reward
from util.db_handler import DBHandler
import logging
import os


class EvolutionaryStrategyTrainer:
    def __init__(
        self,
        model_id: int,
        runner: Runner,
        model: SimpleModel,
        db_handler: DBHandler,
        reward_handler: Reward,
        episode: int = 0,
    ):
        self.runner = runner
        self.model = model
        self.db_handler = db_handler
        self.reward_handler = reward_handler
        self.population_size = reward_handler.to_dict()["population_size"]
        self.sigma = reward_handler.to_dict()["sigma"]
        self.learning_rate = reward_handler.to_dict()["learning_rate"]
        self.logger = logging.getLogger(f"trainer_{model_id}")

        # Wandb logging
        self.run = wandb.init(
            project="mario_evo",
            name=f"model_{model_id}",
            id=f"model_{model_id}_{self.learning_rate}",
            config={
                "model_id": model_id,
                "population_size": self.population_size,
                "sigma": self.sigma,
                "learning_rate": self.learning_rate,
            },
            resume="allow",
        )

        # Device setup
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = model.to(self.device)
        self.model_id = model_id
        self.episode_count = episode
        self.total_steps = 0

    def play(self):
        """Evaluate the model and return the total reward"""
        episode_reward = 0
        state = self.runner.reset()
        self.model.eval()

        while not self.runner.done:
            with torch.no_grad():
                action = self.model.forward(state.to(self.device).unsqueeze(0))
            next_state = self.runner.next(controller=action.squeeze().cpu().tolist())
            reward = self.reward_handler.get_reward(self.runner.step)
            episode_reward += reward
            state = next_state
        self.run.log(self.reward_handler.get_sum())
        return episode_reward

    def evaluate(self):
        """Train the model using Evolutionary Strategy"""
        for _ in range(self.population_size):
            # Generate noise
            noise = [torch.randn_like(param) for param in self.model.parameters()]

            # Create perturbed models
            perturbed_models = []
            for i in range(self.population_size):
                perturbed_model = SimpleModel(random_weights=False).to(self.device)
                perturbed_model.load_state_dict(self.model.state_dict())
                for param, n in zip(perturbed_model.parameters(), noise):
                    param.data += self.sigma * n
                perturbed_models.append(perturbed_model)

            # Evaluate perturbed models
            rewards = []
            for perturbed_model in perturbed_models:
                self.model = perturbed_model
                reward = self.play()
                rewards.append(reward)

            # Compute weighted sum of noise
            rewards = np.array(rewards)
            normalized_rewards = (rewards - rewards.mean()) / rewards.std()
            weighted_sum = [
                torch.zeros_like(param) for param in self.model.parameters()
            ]
            for i, perturbed_model in enumerate(perturbed_models):
                for j, param in enumerate(perturbed_model.parameters()):
                    weighted_sum[j] += normalized_rewards[i] * noise[j]

            # Update model parameters
            for param, w in zip(self.model.parameters(), weighted_sum):
                param.data += (
                    self.learning_rate / (self.population_size * self.sigma) * w
                )

            # Logging
            self.episode_count += 1
            metrics = {
                "episode_count": self.episode_count,
                "reward": rewards.mean(),
                "max_reward": rewards.max(),
                "min_reward": rewards.min(),
            } | self.reward_handler.get_sum()
            self.run.log(metrics)

            # Periodic model saving
            if self.episode_count % 30 == 0:
                self.save_model_checkpoint()

    def save_model_checkpoint(self):
        """Save model checkpoint to wandb and local database"""
        try:
            model_save_path = f"model_episode_{self.episode_count}.pt"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "episode_count": self.episode_count,
                },
                model_save_path,
            )

            artifact = wandb.Artifact(
                name=f"evolutionary_model_checkpoint_{self.model_id}",
                type="model",
                description=f"Evolutionary Strategy model checkpoint at episode {self.episode_count}",
                metadata={"episode": self.episode_count},
            )
            artifact.add_file(model_save_path)
            self.run.log_artifact(artifact)

            os.remove(model_save_path)
        except Exception as e:
            self.logger.error(f"Failed to save model to wandb: {e}")

        try:
            self.db_handler.save_model(
                0,
                self.model_id,
                self.model,
                None,
                self.episode_count,
            )
        except Exception as e:
            self.logger.error(e)
        self.db_handler.increase_train_count(self.model_id)

    def cleanup(self):
        """Cleanup resources"""
        if self.run:
            self.run.finish()
        self.db_handler.save_model(
            0,
            self.model_id,
            self.model,
            None,
            self.episode_count,
        )
