import torch
import torch.optim as optim

class Trainer:
    def __init__(self, model, optimizer, db_handler, logger):
        self.model = model
        self.optimizer = optimizer
        self.db_handler = db_handler
        self.logger = logger
        self.episode_data = []

    def store_step(self, state, action):
        self.episode_data.append((state, action))

    async def train(self, model_id, final_score):
        self.logger.info(f"Training model {model_id}")
        
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
        self.db_handler.save_model(model_id, self.model, self.optimizer)
        
        # Clear episode data
        self.episode_data = []
        
        self.logger.info(f"Completed training for model {model_id}. Loss: {loss.item()}")