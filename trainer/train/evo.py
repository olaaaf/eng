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

        m = {
            "episode_count": self.episode_count,
            "reward": episode_reward,
            "finished": (1.0 if self.runner.alive else 0.0),
            "max_x": max(self.runner.step.x_pos),
            "avg_speed": sum(self.runner.step.horizontal_speed)
            / len(self.runner.step.horizontal_speed),
            "time": self.runner.step.time,
            "score": self.runner.step.score[-1],
        } | self.reward_handler.get_sum()
        self.run.log(m)
        return episode_reward

    def evaluate(self):
        """Train the model using Evolutionary Strategy"""
        base_model_state = self.model.state_dict()
        
        try:
            for _ in range(self.population_size):
                # Generate noise once for all parameters
                noise = [torch.randn_like(param) for param in self.model.parameters()]
                
                # Evaluate perturbed models
                rewards = torch.zeros(self.population_size, device=self.device)
                for i in range(self.population_size):
                    # Create perturbation
                    perturbed_model = SimpleModel(random_weights=False).to(self.device)
                    perturbed_model.load_state_dict(base_model_state)
                    
                    # Apply noise
                    for param, n in zip(perturbed_model.parameters(), noise):
                        param.data += self.sigma * n
                    
                    # Evaluate
                    self.model = perturbed_model
                    rewards[i] = self.play()

                # Normalize rewards
                if rewards.std() != 0:
                    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                else:
                    normalized_rewards = rewards - rewards.mean()

                # Update base model parameters
                self.model.load_state_dict(base_model_state)
                for param, n in zip(self.model.parameters(), noise):
                    update = (
                        self.learning_rate
                        / (self.population_size * self.sigma)
                        * torch.sum(normalized_rewards.view(-1, 1) * n)
                    )
                    param.data += update

                # Logging and checkpointing 
                self.episode_count += 1
                metrics = {
                    "episode_count": self.episode_count,
                    "reward_mean": rewards.mean().item(),
                    "reward_max": rewards.max().item(),
                    "reward_min": rewards.min().item(),
                    "reward_std": rewards.std().item(),
                } | self.reward_handler.get_sum()
                self.run.log(metrics)

                if self.episode_count % 30 == 0:
                    self.save_model_checkpoint()
                    
        except Exception as e:
            self.logger.error(f"Error in evaluate: {e}")
            self.model.load_state_dict(base_model_state)  # Restore original model state
            raise

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
