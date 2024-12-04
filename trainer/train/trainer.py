import torch
import torch.nn.functional as F

import wandb
from game.runner import Runner
from train.replay_buffer import ReplayBuffer
from util.db_handler import DBHandler
from util.logger import setup_logger


class Trainer:
    def __init__(
        self,
        model_id: int,
        runner: Runner,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        db: DBHandler,
    ):
        self.model_id = model_id
        self.runner = runner
        self.model = model
        self.optimizer = optimizer
        self.db = db
        self.logger = setup_logger(db, f"trainer_{model_id}")
        self.episodes = 0

        self.replay_buffer = ReplayBuffer(100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # Initialize wandb
        wandb.init(
            project="mario_b",
        )

    async def train(self):
        state = self.runner.reset()
        episode_reward = 0
        episode_experiences = []  # Store experiences for this episode
        steps = 0
        max_x = float("-inf")

        # Run episode and collect experiences
        while self.runner.alive:
            # Select action
            action = self.model.forward(state)

            # Take action
            next_state = self.runner.next(action)
            reward = self.runner.get_reward()
            done = not self.runner.alive

            # Store experience
            episode_experiences.append((state, action, reward, next_state, done))

            episode_reward += reward
            state = next_state
            steps += 1

            # Track max position
            max_x = max(max_x, self.runner.step.x_pos[-1])

        # After episode ends, add all experiences to replay buffer
        for experience in episode_experiences:
            self.replay_buffer.push(*experience)

        # Perform multiple training iterations if we have enough samples
        episode_loss = 0
        if len(self.replay_buffer) >= self.batch_size:
            # Do multiple training iterations after episode
            n_training_iterations = 32  # Adjust this number as needed
            for _ in range(n_training_iterations):
                loss = self._train_step()
                episode_loss += loss

            episode_loss /= n_training_iterations

            # Update target network after episode
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return episode_reward, episode_loss, steps, max_x

    def _train_step(self):
        # Get random batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Get current Q values
        current_q_values = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        # Get max Q values for next states from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)

            # Compute target Q values using Bellman equation
            target_q_values = (
                reward_batch.unsqueeze(1)
                + (1 - done_batch) * self.gamma * max_next_q_values
            )

        # Compute loss using Huber loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def cleanup(self):
        wandb.finish()
