from typing import Dict, Any
from runners import base_runner

from rl_nav import constants


class EpisodicRunner(base_runner.BaseRunner):
    def __init__(self, model, environment):

        super().__init__(model=model, environment=environment)

    def train(self, num_episodes: int):
        for i in range(num_episodes):
            episode_logging_dict = self._train_episode(episode=i)
            print(episode_logging_dict)

    def _train_episode(self, episode: int) -> Dict[str, Any]:
        """Perform single training loop.

        Args:
            episode: index of episode

        Returns:
            logging_dict: dictionary of items to log (e.g. episode reward).
        """
        episode_reward = 0

        state = self._environment.reset_environment(train=True)

        while self._environment.active:

            epsilon = 0.1

            action = self._model.select_behaviour_action(state, epsilon=epsilon)
            reward, new_state = self._environment.step(action)

            self._model.step(
                state=state,
                action=action,
                reward=reward,
                new_state=new_state,
                active=self._environment.active,
            )
            state = new_state
            episode_reward += reward

        logging_dict = {
            constants.TRAIN_EPISODE_REWARD: episode_reward,
            constants.TRAIN_EPISODE_LENGTH: self._environment.episode_step_count,
        }

        return logging_dict