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
        # Optimizer
        self.optimizer = (
            optimizer
            if optimizer
            else torch.optim.Adam(
                self.model.parameters(),  # Fix: use self.model instead of self.online_model
                lr=reward_handler.to_dict()["learning_rate"],
                weight_decay=reward_handler.to_dict()["weight_decay"],
            )
        )

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
                "lr": reward_handler.to_dict()["learning_rate"],
                "weight_decay": reward_handler.to_dict()["weight_decay"],
                "gamma": reward_handler.to_dict()["gamma"],
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

    def select_action(self, state: torch.Tensor) -> list[float]:
        """Select action for continuous action space"""
        state = state.to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_means = self.model(state)
        # Sample from normal distribution
        action = torch.normal(action_means, torch.ones_like(action_means) * 0.1)
        return action.squeeze().cpu().tolist()

    def evaluate(self):
        """Evaluate the model with improved reward handling"""
        episode_reward = 0
        state = self.runner.reset()
        self.model.eval()
        log_probs = []
        rewards = []

        try:
            while not self.runner.done:
                action = self.select_action(state)
                next_state = self.runner.next(controller=action)
                reward = self.reward_handler.get_reward(self.runner.step)

                # Calculate log probability
                action_means = self.model(state.to(self.device).unsqueeze(0))
                dist = torch.distributions.Normal(
                    action_means, torch.ones_like(action_means) * 0.1
                )
                log_prob = dist.log_prob(torch.tensor(action).to(self.device)).sum()

                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward
                state = next_state

            return episode_reward, log_probs, rewards
        except Exception as e:
            self.logger.error(f"Error in evaluate: {e}")
            raise

    def train(self):
        """Train with improved stability"""
        self.model.train()
        try:
            episode_reward, log_probs, rewards = self.evaluate()

            # Compute discounted rewards with improved stability
            discounted_rewards = []
            cumulative_reward = 0
            gamma = self.reward_handler.to_dict()["gamma"]
            for reward in reversed(rewards):
                cumulative_reward = reward + gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)

            # Convert to tensor and normalize
            discounted_rewards = torch.tensor(discounted_rewards).to(self.device)
            if len(discounted_rewards) > 1:  # Only normalize if more than one reward
                discounted_rewards = (
                    discounted_rewards - discounted_rewards.mean()
                ) / (discounted_rewards.std() + 1e-8)

            # Compute loss with improved stability
            loss = 0
            for log_prob, reward in zip(log_probs, discounted_rewards):
                loss -= log_prob * reward

            # Optimize with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Log metrics
            self.episode_count += 1
            metrics = {
                "episode_count": self.episode_count,
                "reward": episode_reward,
                "loss": loss.item(),
                "finished": (1.0 if self.runner.alive else 0.0),
                "max_x": max(self.runner.step.x_pos),
                "avg_speed": sum(self.runner.step.horizontal_speed)
                / len(self.runner.step.horizontal_speed),
                "time": self.runner.step.time,
                "score": self.runner.step.score[-1],
                "gradient_norm": torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float("inf")
                ).item(),
            } | self.reward_handler.get_sum()
            self.run.log(metrics)

            if self.episode_count % 30 == 0:
                self.save_model_checkpoint()

        except Exception as e:
            self.logger.error(f"Error in training: {e}")
            raise

    def save_model_checkpoint(self):
        """Save model checkpoint to wandb and local database"""
        try:
            model_save_path = f"model_episode_{self.episode_count}.pt"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
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
                None,
                self.model_id,
                self.model,
                self.optimizer,
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
            None,
            self.model_id,
            self.model,
            self.optimizer,
            self.episode_count,
        )
