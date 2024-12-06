import asyncio
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import wandb
from game.runner import Runner
from train.model import SimpleModel
from train.helpers import Reward
from util.db_handler import DBHandler


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization is used
        self.beta = beta  # Importance sampling weight

        self.buffer = []
        self.priorities = []
        self.position = 0

    def add(self, experience, priority: float):
        """Add experience to the buffer with a given priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Sample experiences based on their priorities"""
        if len(self.buffer) == 0:
            return [], [], []

        # Convert priorities to probabilities
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            p=sampling_probabilities,
        )

        # Sample experiences
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance sampling weights
        weights = (len(self.buffer) * sampling_probabilities[indices]) ** -self.beta
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities for the sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class DQNTrainer:
    def __init__(
        self,
        model_id: int,
        runner: Runner,
        model: SimpleModel,
        optimizer,
        db_handler: DBHandler,
        reward_handler: Reward,
        epsilon_start: float = 1.0,
        episode: int = 0,
    ):
        # Hyperparameters
        self.batch_size = reward_handler.to_dict()["batch_size"]
        self.gamma = reward_handler.to_dict()["gamma"]
        self.epsilon_end = reward_handler.to_dict()["epsilon_end"]
        self.epsilon_decay = reward_handler.to_dict()["epsilon_decay"]
        learning_rate = reward_handler.to_dict()["learning_rate"]
        self.runner = runner

        # Prioritized Replay and Target Network Parameters
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000)
        self.target_update_frequency = 100  # Update target network every 100 episodes
        self.logger = logging.getLogger(f"trainer_{model_id}")

        # Initialization
        self.epsilon = epsilon_start

        # Wandb logging
        self.run = wandb.init(
            project="mario_advanced_dqn",
            name=f"model_{model_id}",
            config={
                "model_id": model_id,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "target_update_freq": self.target_update_frequency,
                "learning_rate": learning_rate,
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

        # Model setup with target network
        self.online_model = model.to(self.device)
        self.target_model = SimpleModel(random_weights=True).to(self.device)
        self.target_model.load_state_dict(self.online_model.state_dict())

        # Optimizer
        self.optimizer = (
            optimizer
            if optimizer
            else torch.optim.Adam(self.online_model.parameters(), lr=learning_rate)
        )

        # Other configurations
        self.db_handler = db_handler
        self.model_id = model_id
        self.reward_handler = reward_handler

        # Tracking variables
        self.episode_count = episode
        self.total_steps = 0

    def select_action(self, state: torch.Tensor) -> List[float]:
        """Epsilon-greedy action selection"""
        if torch.rand(1) < self.epsilon:
            return (torch.rand(6)).float().tolist()

        with torch.no_grad():
            state = state.to(self.device).view(1, -1)
            # Use get_actions to match original implementation
            actions = self.online_model.forward(state)

            return (
                (actions).float().squeeze().cpu().tolist()
            )  # Element-wise comparison to produce 0s and 1s

    async def evaluate(self):
        """Advanced training episode with experience replay"""
        episode_reward = 0
        state = self.runner.reset()

        while not self.runner.done:
            # Select and perform action
            action = self.select_action(state)
            next_state = self.runner.next(controller=action)
            reward = self.reward_handler.get_reward(self.runner.step)
            done = not self.runner.alive

            # Compute TD Error for Prioritized Experience Replay
            with torch.no_grad():
                state_tensor = (
                    torch.FloatTensor(state.cpu()).unsqueeze(0).to(self.device)
                )
                next_state_tensor = (
                    torch.FloatTensor(next_state.cpu()).unsqueeze(0).to(self.device)
                )

                current_q = self.online_model(state_tensor)
                next_q = self.target_model(next_state_tensor)

                # Compute TD Error (absolute difference)
                current_action_q = (
                    current_q * torch.FloatTensor(action).to(self.device)
                ).sum()
                target_q = reward + (self.gamma * next_q.max() if not done else 0)
                td_error = abs(target_q - current_action_q).item()

            # Add to prioritized replay buffer
            self.replay_buffer.add(
                (state, action, reward, next_state, done),
                priority=td_error + 1e-5,  # Small epsilon to prevent zero priority
            )

            episode_reward += reward
            state = next_state
            self.total_steps += 1

            # Perform training if buffer is sufficiently populated
            if len(self.replay_buffer.buffer) >= self.batch_size:
                self.train()

            # Periodic target network update
            if self.total_steps % (self.target_update_frequency * 10) == 0:
                self.update_target_network()

            await asyncio.sleep(0)

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

        # Logging and checkpointing
        metrics = {
            "reward": episode_reward,
            "finished": (1.0 if self.runner.alive else 0.0),
            "max_x": max(self.runner.step.x_pos),
            "avg_speed": sum(self.runner.step.horizontal_speed)
            / len(self.runner.step.horizontal_speed),
            "time": self.runner.step.time,
            "score": self.runner.step.score[-1],
            "epsilon": self.epsilon,
        } | self.reward_handler.get_sum()
        self.run.log(metrics)

        # Periodic model saving
        if self.episode_count % 30 == 0:
            self.save_model_checkpoint()

    def train(self):
        """Batch training with Prioritized Experience Replay"""
        # Sample from replay buffer
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)

        if not batch:
            return

        # Prepare batches
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Double DQN: Decouple action selection and evaluation
        current_q_values = self.online_model(states)
        next_q_values_online = self.online_model(next_states)
        next_q_values_target = self.target_model(next_states)

        # Get current Q values for taken actions
        current_q_value = (current_q_values * actions).sum(dim=1)

        # Compute target Q values
        max_next_action = next_q_values_online.argmax(dim=1)
        max_next_q_values = next_q_values_target[
            torch.arange(len(max_next_action)), max_next_action
        ]
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute loss with importance sampling weights
        loss = F.smooth_l1_loss(current_q_value, target_q_values, reduction="none")
        weighted_loss = (loss * weights).mean()

        # Update priorities based on TD error
        td_errors = torch.abs(current_q_value - target_q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Optimize
        self.optimizer.zero_grad()
        # weighted_loss.requires_grad = True
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Log metrics
        self.run.log(
            {
                "loss": weighted_loss.item(),
                "current_q_value": current_q_value.mean().item(),
                "target_q_value": target_q_values.mean().item(),
            }
        )

    def update_target_network(self):
        """Soft update of target network"""
        tau = 0.005  # Soft update parameter
        for target_param, online_param in zip(
            self.target_model.parameters(), self.online_model.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def save_model_checkpoint(self):
        """Save model checkpoint to wandb and local database"""
        try:
            model_save_path = f"model_episode_{self.episode_count}.pt"
            torch.save(
                {
                    "model_state_dict": self.online_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    "episode_count": self.episode_count,
                },
                model_save_path,
            )

            artifact = wandb.Artifact(
                name=f"advanced_model_checkpoint_{self.model_id}",
                type="model",
                description=f"Advanced DQN model checkpoint at episode {self.episode_count}",
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
                self.online_model,
                self.optimizer,
            )
        except Exception as e:
            self.logger.error(e)
        self.db_handler.increase_train_count(self.model_id)

    def cleanup(self):
        """Cleanup resources"""
        if self.run:
            self.run.finish()
        self.db_handler.save_model(
            self.epsilon,
            self.model_id,
            self.online_model,
            self.optimizer,
            self.episode_count,
        )
