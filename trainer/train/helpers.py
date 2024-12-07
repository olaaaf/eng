from abc import abstractmethod
from logging import Logger
from typing import Dict
import json
from game.eval import Step


class Reward:
    logger: Logger

    def __init__(self, _logger: Logger, model_id: int):
        Reward.logger = _logger

    @abstractmethod
    def to_dict(self) -> Dict:
        Reward.logger.warning("Reward.to_dict not overriden")
        return {}

    @abstractmethod
    def get_reward(self, steps: Step) -> float:
        Reward.logger.warning("Reward.get_reward not overriden")

    @abstractmethod
    def get_sum(self) -> Dict:
        Reward.logger.warning("Reward.get_reward not overriden")


class ConfigFileReward(Reward):
    def __init__(
        self, _logger: Logger, model_id: int, config_path: str = "rewards.json"
    ):
        super().__init__(_logger, model_id)
        self.settings: Dict
        self.sum = {}
        self.punished_for_death = False
        self.rewarded_for_finish = False

        try:
            with open(config_path) as file:
                settings = json.loads(file.read())
                self.settings = settings[str(model_id)]
        except FileNotFoundError:
            super().logger.error(f"Config file {config_path} does not exist")
        except Exception as e:
            super().logger.error(f"Analyzing rewards configuration json: {e}")

    def to_dict(self) -> Dict:
        return self.settings["config"]

    def get_reward(self, steps: Step) -> float:
        reward = 0

        position_delta = (
            steps.x_pos[-1] - steps.x_pos[-2] if len(steps.x_pos) > 1 else 0
        )
        score_delta = steps.score[-1] - steps.score[-2] if len(steps.score) > 1 else 0

        if position_delta < 1e-9:
            # penalty for moving left
            reward -= position_delta * self.settings["position_delta"]
        else:
            # reward for moving right
            reward += position_delta * self.settings["position_delta"]

        rewards = {
            "rewards_for_position": position_delta * self.settings["position_delta"],
            "rewards_for_score_delta": 0,
            "rewards_for_horizontal_speed": 0,
            "rewards_for_level": 0,
            "rewards_for_death": 0,
            "rewards_for_time": 0,
        }

        # reward for gaining score
        if score_delta > 0:
            reward += score_delta * self.settings["score_delta"]
            rewards["rewards_for_score_delta"] = (
                score_delta * self.settings["score_delta"]
            )

        # reward for speed!!
        if steps.horizontal_speed[-1] > -1e-9:
            reward += steps.horizontal_speed[-1] * self.settings["speed"]
            rewards["rewards_for_horizontal_speed"] = (
                steps.horizontal_speed[-1] * self.settings["speed"]
            )

        if steps.level == 1 and not self.rewarded_for_finish:
            reward += self.settings["finish"]
            rewards["rewards_for_level"] = self.settings["finish"]
            self.rewarded_for_finish = True

        # Large penalty for death
        if steps.died and not self.punished_for_death:
            reward -= self.settings["death"]
            self.punished_for_death = True
            rewards["rewards_for_death"] = -self.settings["death"]

        max_time = 9832
        time_penalty_scale = self.settings["time_penalty"]
        if steps.time > self.settings["time_penalty_start"]:
            # Compute penalty as a function of time
            time_over = steps.time - self.settings["time_penalty_start"]
            penalty = (
                time_penalty_scale
                * (time_over / (max_time - self.settings["time_penalty_start"])) ** 2
            )
            rewards["rewards_for_time"] = -penalty
            reward -= penalty

        for key, value in rewards.items():
            if key in self.sum:
                self.sum[key] += value
            else:
                self.sum[key] = value
        return reward

    def get_sum(self):
        s = self.sum
        self.rewarded_for_finish = False
        self.punished_for_death = False
        self.sum = {}
        return s
