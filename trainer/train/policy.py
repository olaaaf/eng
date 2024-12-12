import logging
import os
import torch
import torch.nn.functional as F
import wandb
from game.runner import Runner
from train.model import SimpleModel
from train.helpers import Reward
from util.db_handler import DBHandler


class PolicyGradientTrainer:
    def __init__(
        self,
        model_id: int,
        runner: Runner,
        model: SimpleModel,
        optimizer,
        db_handler: DBHandler,
        reward_handler: Reward,
        episode: int = 0,
    ):
        self.runner = runner
        self.model = model
        self.optimizer = optimizer
        self.db_handler = db_handler
        self.reward_handler = reward_handler
        self.logger = logging.getLogger(f"trainer_{model_id}")

        # Wandb logging
        self.run = wandb.init(
            project="mario_shpeed",
            name=f"model_{model_id}",
            id=f"model_{model_id}",
            config={
                "model_id": model_id,
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

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select action based on policy"""
        state = state.to(self.device).unsqueeze(0)
        probs = F.softmax(self.model(state), dim=1)
        action = torch.multinomial(probs, num_samples=1).item()
        return action

    def play(self):
        """Evaluate the model and return the total reward"""
        episode_reward = 0
        state = self.runner.reset()
        self.model.eval()
        log_probs = []
        rewards = []

        while not self.runner.done:
            action = self.select_action(state)
            log_prob = torch.log(
                F.softmax(self.model(state.to(self.device).unsqueeze(0)), dim=1)[
                    0, action
                ]
            )
            log_probs.append(log_prob)
            next_state = self.runner.next(controller=[action])
            reward = self.reward_handler.get_reward(self.runner.step)
            rewards.append(reward)
            episode_reward += reward
            state = next_state

        return episode_reward, log_probs, rewards

    def evaluate(self):
        """Train the model using Policy Gradient"""
        self.model.train()
        episode_reward, log_probs, rewards = self.play()

        # Compute discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = (
                reward + self.reward_handler.to_dict()["gamma"] * cumulative_reward
            )
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards).to(self.device)
        log_probs = torch.stack(log_probs)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-5
        )

        # Compute loss
        loss = -torch.sum(log_probs * discounted_rewards)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging
        self.episode_count += 1
        metrics = {
            "episode_count": self.episode_count,
            "reward": episode_reward,
            "loss": loss.item(),
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
                name=f"policy_gradient_model_checkpoint_{self.model_id}",
                type="model",
                description=f"Policy Gradient model checkpoint at episode {self.episode_count}",
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
