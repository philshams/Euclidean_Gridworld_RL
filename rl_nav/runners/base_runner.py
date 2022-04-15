import abc
import os
from typing import Any, Dict

from rl_nav import constants
from rl_nav.environments import escape_env, visualisation_env
from run_modes import base_runner


class BaseRunner(base_runner.BaseRunner):
    def __init__(self, config, unique_id: str):
        self._environment = self._setup_environment(config=config)
        self._model = self._setup_model(config=config)

        self._epsilon = config.epsilon
        self._num_steps = config.num_steps
        self._step_count = 0
        self._rollout_frequency = config.rollout_frequency
        self._next_rollout_step = config.rollout_frequency
        self._visualisation_frequency = config.visualisation_frequency
        self._next_visualisation_step = config.visualisation_frequency

        super().__init__(config=config, unique_id=unique_id)

        # logging setup
        self._rollout_folder_path = os.path.join(
            self._checkpoint_path, constants.ROLLOUTS
        )
        self._visualisations_folder_path = os.path.join(
            self._checkpoint_path, constants.VISUALISATIONS
        )
        os.makedirs(name=self._rollout_folder_path, exist_ok=True)
        os.makedirs(name=self._visualisations_folder_path, exist_ok=True)

    @abc.abstractmethod
    def train(self):
        pass

    def _setup_environment(self, config):
        """Initialise environment specified in configuration."""
        environment_args = self._get_environment_args(config=config)

        if config.environment == constants.ESCAPE_ENV:
            environment = escape_env.EscapeEnv(**environment_args)

        environment = visualisation_env.VisualisationEnv(environment)

        return environment

    def _get_environment_args(self, config) -> Dict[str, Any]:
        """Get arguments needed to pass to environment."""
        if config.environment == constants.ESCAPE_ENV:
            env_args = {
                constants.MAP_PATH: config.map_path,
                constants.REPRESENTATION: config.representation,
                constants.REWARD_POSITIONS: config.reward_positions,
                constants.START_POSITION: config.start_position,
                constants.EPISODE_TIMEOUT: config.episode_timeout
            }

            reward_attr_dict = {constants.AVAILABILITY: config.availability}
            if config.statistics == constants.GAUSSIAN:
                reward_attr_dict[constants.TYPE] = constants.GAUSSIAN
                reward_attr_parameter_dict = {
                    constants.MEAN: config.gaussian_mean,
                    constants.VARIANCE: config.gaussian_variance,
                }
            reward_attr_dict[constants.PARAMETERS] = reward_attr_parameter_dict
            env_args[constants.REWARD_ATTRIBUTES] = reward_attr_dict

        return env_args

    @abc.abstractmethod
    def _setup_model(self, config):
        """Instantiate model specified in configuration."""
        pass
