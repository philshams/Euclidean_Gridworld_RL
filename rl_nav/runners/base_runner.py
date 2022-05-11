import abc
import copy
import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
from rl_nav import constants
from rl_nav.environments import (escape_env_cardinal, escape_env_diagonal,
                                 visualisation_env)
from rl_nav.models import (a_star, dyna, linear_features, q_learning,
                           successor_representation)
from rl_nav.utils import model_utils
from run_modes import base_runner


class BaseRunner(base_runner.BaseRunner):
    def __init__(self, config, unique_id: str):

        self._train_environment = self._setup_train_environment(config=config)
        self._test_environments = self._setup_test_environments(config=config)

        test_env_excess_states = [
            set(test_env.state_space) - set(self._train_environment.state_space)
            for test_env in self._test_environments.values()
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

        super().__init__(config=config, unique_id=unique_id)

        self._epsilon = config.epsilon
        self._num_steps = config.num_steps
        self._step_count = 0
        self._rollout_frequency = config.rollout_frequency
        self._visualisation_frequency = config.visualisation_frequency
        self._test_frequency = config.test_frequency
        self._checkpoint_frequency = config.checkpoint_frequency
        self._one_dim_blocks = config.one_dim_blocks

        self._visualisations = config.visualisations

        # logging setup
        self._rollout_folder_path = os.path.join(
            self._checkpoint_path, constants.ROLLOUTS
        )
        self._visualisations_folder_path = os.path.join(
            self._checkpoint_path, constants.VISUALISATIONS
        )
        self._env_skeletons_path = os.path.join(
            self._checkpoint_path, constants.ENV_SKELETON
        )
        os.makedirs(name=self._rollout_folder_path, exist_ok=True)
        os.makedirs(name=self._visualisations_folder_path, exist_ok=True)
        os.makedirs(name=self._env_skeletons_path, exist_ok=True)

        for map_name, test_env in self._test_environments.items():
            test_env.save_as_array(
                save_path=os.path.join(
                    self._env_skeletons_path, f"{map_name}_{constants.ENV_SKELETON}"
                )
            )

    def _get_data_columns(self):
        columns = [
            constants.STEP,
        ]
        for map_name in self._test_environments.keys():
            columns.append(f"{constants.TEST_EPISODE_REWARD}_{map_name}")
            columns.append(f"{constants.TEST_EPISODE_LENGTH}_{map_name}")
            columns.append(
                f"{constants.TEST_EPISODE_REWARD}_{map_name}_{constants.FINAL_REWARD_RUN}"
            )
            columns.append(
                f"{constants.TEST_EPISODE_LENGTH}_{map_name}_{constants.FINAL_REWARD_RUN}"
            )
            columns.append(
                f"{constants.TEST_EPISODE_REWARD}_{map_name}_{constants.FIND_SHELTER_RUN}"
            )
            columns.append(
                f"{constants.TEST_EPISODE_LENGTH}_{map_name}_{constants.FIND_SHELTER_RUN}"
            )

        return columns + self._get_runner_specific_data_columns()

    @abc.abstractmethod
    def _get_runner_specific_data_columns(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    def _train_step(self, state):

        self._step_count += 1

        logging_dict = {}
        if (
            self._step_count % self._visualisation_frequency == 0
            and self._step_count != 1
        ):
            self._generate_visualisations()
        if self._step_count % self._test_frequency == 0:
            logging_dict = self._perform_tests(
                rollout=(self._step_count % self._rollout_frequency == 0),
                planning=self._planner,
            )

        state, reward = self._model_train_step(state)

        return state, reward, logging_dict

    @abc.abstractmethod
    def _model_train_step(self, state):
        pass

    def _setup_train_environment(self, config):
        """Initialise environment specified in configuration."""

        environment_args = self._get_environment_args(config=config, train=True)

        if config.train_env_name == constants.ESCAPE_ENV:
            environment = escape_env_cardinal.EscapeEnvCardinal(**environment_args)
        elif config.train_env_name == constants.ESCAPE_ENV_DIAGONAL:
            environment = escape_env_diagonal.EscapeEnvDiagonal(**environment_args)

        environment = visualisation_env.VisualisationEnv(environment)

        return environment

    def _setup_test_environments(self, config):
        """Initialise environment specified in configuration."""

        environment_args = self._get_environment_args(config=config, train=False)

        environments = {}

        if config.test_env_name == constants.ESCAPE_ENV:
            for map_path in list(set(config.test_map_paths + [config.train_map_path])):
                map_name = map_path.split("/")[-1].rstrip(".txt")
                environment_args[constants.MAP_PATH] = map_path
                environment = escape_env_cardinal.EscapeEnvCardinal(**environment_args)
                environments[map_name] = visualisation_env.VisualisationEnv(environment)
        if config.test_env_name == constants.ESCAPE_ENV_DIAGONAL:
            for map_path in list(set(config.test_map_paths + [config.train_map_path])):
                map_name = map_path.split("/")[-1].rstrip(".txt")
                environment_args[constants.MAP_PATH] = map_path
                environment = escape_env_diagonal.EscapeEnvDiagonal(**environment_args)
                environments[map_name] = visualisation_env.VisualisationEnv(environment)

        return environments

    def _get_environment_args(self, config, train: bool) -> Dict[str, Any]:
        """Get arguments needed to pass to environment."""
        if train:
            env_name = config.train_env_name
            config_key_prefix = constants.TRAIN
        else:
            env_name = config.test_env_name
            config_key_prefix = constants.TEST

        if env_name in [constants.ESCAPE_ENV, constants.ESCAPE_ENV_DIAGONAL]:
            env_args = {
                constants.TRAINING: train,
                constants.REPRESENTATION: getattr(
                    config, f"{config_key_prefix}_{constants.REPRESENTATION}"
                ),
                constants.REWARD_POSITIONS: getattr(
                    config, f"{config_key_prefix}_{constants.REWARD_POSITIONS}"
                ),
                constants.STEP_COST_FACTOR: getattr(config, constants.STEP_COST_FACTOR),
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
        if config.model == constants.Q_LEARNING:
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
        elif config.model == constants.SUCCESSOR_REP:
            model = successor_representation.SuccessorRepresentation(
                action_space=self._train_environment.action_space,
                state_space=self._train_environment.state_space,
                behaviour=config.behaviour,
                target=config.target,
                initialisation_strategy=initialisation_strategy,
                learning_rate=config.learning_rate,
                gamma=config.discount_factor,
                imputation_method=config.imputation_method,
            )
        elif config.model == constants.DYNA:
            model = dyna.Dyna(
                action_space=self._train_environment.action_space,
                state_space=self._train_environment.state_space,
                behaviour=config.behaviour,
                target=config.target,
                initialisation_strategy=initialisation_strategy,
                learning_rate=config.learning_rate,
                gamma=config.discount_factor,
                imputation_method=config.imputation_method,
                plan_steps_per_update=config.plan_steps_per_update,
            )
        elif config.model == constants.UNDIRECTED_DYNA:
            model = dyna.Dyna(
                action_space=self._train_environment.action_space,
                state_space=self._train_environment.state_space,
                behaviour=config.behaviour,
                target=config.target,
                initialisation_strategy=initialisation_strategy,
                learning_rate=config.learning_rate,
                gamma=config.discount_factor,
                imputation_method=config.imputation_method,
                plan_steps_per_update=config.plan_steps_per_update,
                inverse_actions=self._train_environment.inverse_action_mapping,
            )
        elif config.model == constants.A_STAR:
            model = a_star.AStar(
                action_space=self._train_environment.action_space,
                state_space=self._train_environment.state_space,
            )
        elif config.model == constants.LINEAR_FEATURES:
            model = linear_features.LinearFeatureLearner(
                features=config.features,
                action_space=self._train_environment.action_space,
                state_space=self._train_environment.state_space,
                behaviour=config.behaviour,
                target=config.target,
                initialisation_strategy=initialisation_strategy,
                learning_rate=config.learning_rate,
                gamma=config.discount_factor,
                imputation_method=config.imputation_method,
            )
        else:
            raise ValueError(f"Model {config.model} not recogised.")
        return model

    def _write_scalar(
        self,
        tag: str,
        step: int,
        scalar: Union[float, int],
        df_tag: Optional[str] = None,
    ):
        """If specified, log scalar."""
        df_tag = df_tag or tag
        self._data_logger.write_scalar(tag=df_tag, step=step, scalar=scalar)

    def _log_episode(self, step: int, logging_dict: Dict[str, float]) -> None:
        """Write scalars for all quantities collected in logging dictionary.

        Args:
            step: current step.
            logging_dict: dictionary of items to be logged collected during training.
        """
        for tag, scalar in logging_dict.items():
            self._write_scalar(tag=tag, step=step, scalar=scalar)

    def _generate_visualisations(self):
        if constants.VALUE_FUNCTION in self._visualisations:
            averaged_values = (
                self._train_environment.average_values_over_positional_states(
                    self._model.state_action_values
                )
            )
            averaged_max_values = {p: max(v) for p, v in averaged_values.items()}
            self._train_environment.plot_heatmap_over_env(
                heatmap=averaged_max_values,
                save_name=os.path.join(
                    self._visualisations_folder_path,
                    f"{self._step_count}_{constants.VALUES_PDF}",
                ),
            )
        if constants.VISITATION_COUNTS in self._visualisations:
            averaged_visitation_counts = (
                self._train_environment.average_values_over_positional_states(
                    self._model.state_visitation_counts
                )
            )
            self._train_environment.plot_heatmap_over_env(
                heatmap=averaged_visitation_counts,
                save_name=os.path.join(
                    self._visualisations_folder_path,
                    f"{self._step_count}_{constants.VISITATION_COUNTS_PDF}",
                ),
            )

        self._runner_specific_visualisations()

    def _test_rollout(self, save_name_base: str):
        for i, (map_name, test_env) in enumerate(self._test_environments.items()):
            test_env.visualise_episode_history(
                save_path=os.path.join(
                    self._rollout_folder_path,
                    f"{save_name_base}_{map_name}_{self._step_count}.mp4",
                ),
                history=constants.TEST,
            )
            test_env.save_history(
                save_path=os.path.join(
                    self._rollout_folder_path,
                    f"{save_name_base}_{map_name}_{self._step_count}",
                )
            )

    def _perform_tests(self, rollout: bool, planning: bool):
        find_shelter_logging_dict = self._find_shelter_test(
            rollout=rollout, planning=planning
        )
        find_reward_logging_dict = self._find_reward_test(
            rollout=rollout, planning=planning
        )
        plain_logging_dict = self._test(
            test_model=self._model, rollout=rollout, planning=planning
        )
        logging_dict = {
            **plain_logging_dict,
            **find_reward_logging_dict,
            **find_shelter_logging_dict,
        }
        return logging_dict

    def _find_reward_test(self, rollout: bool, planning: bool):
        """Allow agent one more 'period' of exploration to find the reward
        (in the test environment), before test rollout."""
        test_logging_dict = {}

        for i, (map_name, test_env) in enumerate(self._test_environments.items()):

            state = test_env.reset_environment(episode_timeout=np.inf)

            model_copy = copy.deepcopy(self._model)
            if not self._one_dim_blocks:
                model_copy.allow_state_instantiation = True

            while test_env.active:

                # TODO: unclear which behaviour policy to use here...
                action = model_copy.select_behaviour_action(
                    state,
                    epsilon=self._epsilon,
                    excess_state_mapping=self._excess_state_mapping[i],
                )
                reward, new_state = test_env.step(action)

                model_copy.step(
                    state=state,
                    action=action,
                    reward=reward,
                    new_state=new_state,
                    active=test_env.active,
                )
                state = new_state

            model_copy.allow_state_instantiation = False
            model_copy.eval()

            if planning:
                reward, length = self._single_planning_test(
                    test_model=model_copy,
                    test_env=test_env,
                    excess_state_mapping=self._excess_state_mapping[i],
                    retain_history=True,
                )
            else:
                reward, length = self._single_test(
                    test_model=model_copy,
                    test_env=test_env,
                    excess_state_mapping=self._excess_state_mapping[i],
                    retain_history=True,
                )

            test_logging_dict[
                f"{constants.TEST_EPISODE_REWARD}_{map_name}_{constants.FINAL_REWARD_RUN}"
            ] = reward
            test_logging_dict[
                f"{constants.TEST_EPISODE_LENGTH}_{map_name}_{constants.FINAL_REWARD_RUN}"
            ] = length

        if rollout:
            self._test_rollout(
                save_name_base=f"{constants.INDIVIDUAL_TEST_RUN}_{constants.FINAL_REWARD_RUN}"
            )

        return test_logging_dict

    def _find_shelter_test(self, rollout: bool, planning: bool):
        """Agent starts in shelter and explores/learns in the time until it first reaches
        the threat zone at which point a test rollout is triggered."""
        test_logging_dict = {}

        for i, (map_name, test_env) in enumerate(self._test_environments.items()):

            temporary_start_state = random.choice(test_env.reward_positions)
            state = test_env.reset_environment(
                episode_timeout=np.inf,
                start_position=temporary_start_state,
                reward_availability=constants.INFINITE,
            )

            model_copy = copy.deepcopy(self._model)
            if not self._one_dim_blocks:
                model_copy.allow_state_instantiation = True

            while state != tuple(test_env.starting_xy):

                # TODO: unclear which behaviour policy to use here...
                action = model_copy.select_behaviour_action(
                    state,
                    epsilon=self._epsilon,
                    excess_state_mapping=self._excess_state_mapping[i],
                )
                reward, new_state = test_env.step(action)

                model_copy.step(
                    state=state,
                    action=action,
                    reward=reward,
                    new_state=new_state,
                    active=test_env.active,
                )
                state = new_state

            model_copy.allow_state_instantiation = False
            model_copy.eval()

            if planning:
                reward, length = self._single_planning_test(
                    test_model=model_copy,
                    test_env=test_env,
                    excess_state_mapping=self._excess_state_mapping[i],
                    retain_history=True,
                )
            else:
                reward, length = self._single_test(
                    test_model=model_copy,
                    test_env=test_env,
                    excess_state_mapping=self._excess_state_mapping[i],
                    retain_history=True,
                )

            test_logging_dict[
                f"{constants.TEST_EPISODE_REWARD}_{map_name}_{constants.FIND_SHELTER_RUN}"
            ] = reward
            test_logging_dict[
                f"{constants.TEST_EPISODE_LENGTH}_{map_name}_{constants.FIND_SHELTER_RUN}"
            ] = length

        if rollout:
            self._test_rollout(
                save_name_base=f"{constants.INDIVIDUAL_TEST_RUN}_{constants.FIND_SHELTER_RUN}"
            )

        return test_logging_dict

    def _test(self, test_model, rollout: bool, planning: bool):
        """Test rollout."""
        test_model.eval()

        test_logging_dict = {}

        for i, (map_name, test_env) in enumerate(self._test_environments.items()):

            if planning:
                reward, length = self._single_planning_test(
                    test_model=test_model,
                    test_env=test_env,
                    excess_state_mapping=self._excess_state_mapping[i],
                )

            else:
                reward, length = self._single_test(
                    test_model=test_model,
                    test_env=test_env,
                    excess_state_mapping=self._excess_state_mapping[i],
                )

            test_logging_dict[f"{constants.TEST_EPISODE_REWARD}_{map_name}"] = reward
            test_logging_dict[f"{constants.TEST_EPISODE_LENGTH}_{map_name}"] = length

        if rollout:
            self._test_rollout(save_name_base=f"{constants.INDIVIDUAL_TEST_RUN}")

        test_model.train()

        return test_logging_dict

    def _single_test(
        self,
        test_model,
        test_env,
        excess_state_mapping,
        retain_history: Optional[bool] = False,
    ):
        """Test rollout with single model on single environment."""
        episode_reward = 0

        state = test_env.reset_environment(retain_history=retain_history)

        while test_env.active:

            action = test_model.select_target_action(
                state, excess_state_mapping=excess_state_mapping
            )
            reward, state = test_env.step(action)

            episode_reward += reward

        return episode_reward, test_env.episode_step_count

    def _single_planning_test(
        self,
        test_model,
        test_env,
        excess_state_mapping,
        retain_history: Optional[bool] = False,
    ):
        """Test rollout with single model on single environment."""
        episode_reward = 0

        state = test_env.reset_environment(retain_history=retain_history)
        planned_path = test_model.plan(state)

        if planned_path is not None:
            planned_path_deltas = [
                tuple(delta)
                for delta in (np.array(planned_path[1:]) - np.array(planned_path[:-1]))
            ]
            planned_path_actions = [
                test_env.delta_actions[d] for d in planned_path_deltas
            ]

            for action in planned_path_actions:

                reward, state = test_env.step(action)

                episode_reward += reward

        return episode_reward, test_env.episode_step_count
