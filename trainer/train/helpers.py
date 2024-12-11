from abc import abstractmethod
from logging import Logger
from typing import Dict
import json
from game.eval import Step

MARIO_MAX_SHPEED = 40.0
MARIO_MAX_DELTA_X = 10


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
        self, _logger: Logger, model_id: int, config_path: str = "rewards.json", db=None
    ):
        super().__init__(_logger, model_id)
        self.model_id = model_id
        self.db = db
        self.highscore_cache = None
        self.rewarded_for_x_highscore = False
        self.rewarded_for_score_highscore = False
        self.rewarded_for_time_highscore = False
        self.settings: Dict
        self.sum = {}
        self.punished_for_death = False
        self.rewarded_for_finish = False
        self.run_max_x = 40

        self.reward_mean = 0
        self.reward_std = 1
        self.step_count = 0
        self.sum_rewards = 0
        self.sum_rewards_squared = 0

        try:
            with open(config_path) as file:
                settings = json.loads(file.read())
                self.settings = settings[str(model_id)]
        except FileNotFoundError:
            super().logger.error(f"Config file {config_path} does not exist")
        except Exception as e:
            super().logger.error(f"Analyzing rewards configuration json: {e}")

    def update_running_stats(self, reward):
        """Update running statistics for mean and std of rewards."""
        self.step_count += 1
        self.sum_rewards += reward
        self.sum_rewards_squared += reward**2

        self.reward_mean = self.sum_rewards / self.step_count
        variance = self.sum_rewards_squared / self.step_count - self.reward_mean**2
        self.reward_std = max(variance**0.5, 1e-5)  # Avoid zero division

    def normalize_reward(self, reward):
        """Normalize reward using running mean and std."""
        return (reward - self.reward_mean) / self.reward_std

    def _get_cached_highscore(self):
        if self.highscore_cache is None and self.db:
            self.highscore_cache = self.db.get_highscore(self.model_id)
        return self.highscore_cache

    def to_dict(self) -> Dict:
        return self.settings["config"]

    def get_reward(self, steps: Step) -> float:
        reward = 0

        score_delta = steps.score[-1] - steps.score[-2] if len(steps.score) > 1 else 0

        score_reward = (score_delta / 100) * self.settings.get("score_delta", 0.1)
        speed_reward = 0
        if steps.horizontal_speed[-1] <= 0:
            speed_reward = -self.settings.get("speed", 1.0 / MARIO_MAX_SHPEED)
        position_reward = 0
        level_reward = 0
        death_reward = 0
        time_over_reward = 0
        stomp_reward = 0

        if steps.level == 1 and not self.rewarded_for_finish:
            level_reward = self.settings.get("finish", 10)
            self.rewarded_for_finish = True

        # Large penalty for death
        if steps.died and not self.punished_for_death:
            death_reward = self.settings.get("death", -10)
            self.punished_for_death = True

        max_time = 9832
        time_penalty_scale = self.settings["time_penalty"]
        if steps.time > self.settings["time_penalty_start"]:
            # Compute penalty as a function of time
            time_over = steps.time - self.settings["time_penalty_start"]
            penalty = (
                time_penalty_scale
                * (time_over / (max_time - self.settings["time_penalty_start"])) ** 2
            )
            time_over_reward = -penalty

        if steps.just_stomped:
            stomp_reward = self.settings.get("stomp", 1)

        if steps.x_pos[-1] > self.run_max_x:
            self.run_max_x = steps.x_pos[-1]
            position_reward_function = (1.0 / 3220.0 * steps.x_pos[-1]) - 2.0 / 161.0
            position_reward = (
                self.settings.get("position_reward", 1) * position_reward_function
            )

        # Highscore rewards
        if False:
            highscore = self._get_cached_highscore()
            if highscore:
                _, high_x, high_score, high_time = highscore

                # X position highscore
                if not self.rewarded_for_x_highscore and steps.x_pos[-1] > high_x:
                    reward += self.settings.get("beat_x_highscore", 50)
                    rewards["rewards_for_x_highscore"] = self.settings.get(
                        "beat_x_highscore", 50
                    )
                    self.rewarded_for_x_highscore = True

                # Score highscore
                if (
                    not self.rewarded_for_score_highscore
                    and steps.score[-1] > high_score
                ):
                    reward += self.settings.get("beat_score_highscore", 10)
                    rewards["rewards_for_score_highscore"] = self.settings.get(
                        "beat_score_highscore", 100
                    )
                    self.rewarded_for_score_highscore = True

                # Time highscore (only when completing level)
                if (
                    steps.level == 1
                    and not self.rewarded_for_time_highscore
                    and steps.time[-1] < high_time
                    and steps.time[-1] > 0
                ):
                    reward += self.settings.get("beat_time_highscore", 150)
                    rewards["rewards_for_time_highscore"] = self.settings.get(
                        "beat_time_highscore", 150
                    )
                    self.rewarded_for_time_highscore = True

        rewards = {
            "position_reward": position_reward,
            "score_reward": score_reward,
            "speed_reward": speed_reward,
            "level_reward": level_reward,
            "death_reward": death_reward,
            "time_over_reward": time_over_reward,
            "stomp_reward": stomp_reward,
        }

        for key, value in rewards.items():
            reward += value
            if key in self.sum:
                self.sum[key] += value
            else:
                self.sum[key] = value

        self.update_running_stats(reward)
        normalized_reward = self.normalize_reward(reward)
        return max(min(normalized_reward, 5.0), -5.0)

    def get_sum(self):
        s = self.sum
        self.reward_mean = 0
        self.reward_std = 1
        self.step_count = 0
        self.sum_rewards = 0
        self.sum_rewards_squared = 0
        self.rewarded_for_finish = False
        self.rewarded_for_score_highscore = False
        self.rewarded_for_time_highscore = False
        self.rewarded_for_x_highscore = False
        self.punished_for_death = False
        self.run_max_x = 40
        self.sum = {}
        return s
