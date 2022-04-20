import abc
import os
from typing import Any, Dict

import numpy as np
from rl_nav import constants
from rl_nav.environments import escape_env, visualisation_env
from rl_nav.models import q_learning
from rl_nav.utils import model_utils
from run_modes import base_runner


class BaseRunner(base_runner.BaseRunner):
    def __init__(self, config, unique_id: str):
        self._train_environment = self._setup_train_environment(config=config)
        self._test_environments = self._setup_test_environments(config=config)

        test_env_excess_states = [
            set(test_env.state_space) - set(self._train_environment.state_space)
            for test_env in self._test_environments
        ]
        self._excess_state_mapping = [
            {
                state: [
                    s
                    for s in self._train_environment.state_space
                    if max(np.array(s) - np.array(state)) <= 1
                ]
                for state in excess_states
            }
            for excess_states in test_env_excess_states
        ]

        self._model = self._setup_model(config=config)

        self._epsilon = config.epsilon
        self._num_steps = config.num_steps
        self._step_count = 0
        self._rollout_frequency = config.rollout_frequency
        self._next_rollout_step = config.rollout_frequency
        self._visualisation_frequency = config.visualisation_frequency
        self._next_visualisation_step = config.visualisation_frequency
        self._test_frequency = config.test_frequency
        self._next_test_step = 0

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

    def _setup_train_environment(self, config):
        """Initialise environment specified in configuration."""

        environment_args = self._get_environment_args(config=config, train=True)

        if config.train_env_name == constants.ESCAPE_ENV:
            environment = escape_env.EscapeEnv(**environment_args)

        environment = visualisation_env.VisualisationEnv(environment)

        return environment

    def _setup_test_environments(self, config):
        """Initialise environment specified in configuration."""

        environment_args = self._get_environment_args(config=config, train=False)

        environments = []

        if config.test_env_name == constants.ESCAPE_ENV:
            for map_path in config.test_map_paths:
                environment_args[constants.MAP_PATH] = map_path
                environment = escape_env.EscapeEnv(**environment_args)
                environments.append(environment)

        environments = [visualisation_env.VisualisationEnv(env) for env in environments]

        return environments

    def _get_environment_args(self, config, train: bool) -> Dict[str, Any]:
        """Get arguments needed to pass to environment."""
        if train:
            env_name = config.train_env_name
            config_key_prefix = constants.TRAIN
        else:
            env_name = config.test_env_name
            config_key_prefix = constants.TEST

        if env_name == constants.ESCAPE_ENV:
            env_args = {
                constants.TRAINING: train,
                constants.REPRESENTATION: getattr(
                    config, f"{config_key_prefix}_{constants.REPRESENTATION}"
                ),
                constants.REWARD_POSITIONS: getattr(
                    config, f"{config_key_prefix}_{constants.REWARD_POSITIONS}"
                ),
                constants.START_POSITION: getattr(
                    config, f"{config_key_prefix}_{constants.START_POSITION}"
                ),
                constants.EPISODE_TIMEOUT: getattr(
                    config, f"{config_key_prefix}_{constants.EPISODE_TIMEOUT}"
                ),
            }

            reward_attr_dict = {
                constants.AVAILABILITY: getattr(
                    config, f"{config_key_prefix}_{constants.AVAILABILITY}"
                )
            }
            if (
                getattr(config, f"{config_key_prefix}_{constants.STATISTICS}")
                == constants.GAUSSIAN
            ):
                reward_attr_dict[constants.TYPE] = constants.GAUSSIAN
                reward_attr_parameter_dict = {
                    constants.MEAN: getattr(
                        config, f"{config_key_prefix}_{constants.GAUSSIAN_MEAN}"
                    ),
                    constants.VARIANCE: getattr(
                        config, f"{config_key_prefix}_{constants.GAUSSIAN_VARIANCE}"
                    ),
                }
            reward_attr_dict[constants.PARAMETERS] = reward_attr_parameter_dict
            env_args[constants.REWARD_ATTRIBUTES] = reward_attr_dict

        if train:
            env_args[constants.MAP_PATH] = config.train_map_path

        return env_args

    def _setup_model(self, config):
        """Instantiate model specified in configuration."""
        initialisation_strategy = model_utils.get_initialisation_strategy(config)
        model = q_learning.QLearner(
            action_space=self._train_environment.action_space,
            state_space=self._train_environment.state_space,
            behaviour=config.behaviour,
            target=config.target,
            initialisation_strategy=initialisation_strategy,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
            imputation_method=config.imputation_method,
        )
        return model

    def _test(self):
        """Test rollout."""
        self._model.eval()

        test_logging_dict = {}

        for i, test_env in enumerate(self._test_environments):
            episode_reward = 0

            state = test_env.reset_environment()

            while test_env.active:

                action = self._model.select_target_action(
                    state, excess_state_mapping=self._excess_state_mapping[i]
                )
                reward, state = test_env.step(action)

                episode_reward += reward

            test_logging_dict[f"{constants.TEST_EPISODE_REWARD}_{i}"] = episode_reward
            test_logging_dict[
                f"{constants.TEST_EPISODE_LENGTH}_{i}"
            ] = test_env.episode_step_count

        self._model.train()

        return test_logging_dict
