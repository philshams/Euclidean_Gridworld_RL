from rl_nav.runners import episodic_runner, lifelong_runner
import numpy as np
from rl_nav import constants
import os
import copy
import random


class LifelongQLearningHierarchyRunner(lifelong_runner.LifelongRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._planner = False

        self._train_step_cost_factor = config.train_super_step_cost_factor
        self._test_step_cost_factor = config.train_super_step_cost_factor

    def get_high_state(self, state):
        state_index = self._train_environment._env._inverse_partition_mapping[state]
        high_state = self._train_environment._env._state_position_mapping[state_index]
        return high_state

    def _model_train_step(self, state) -> float:
        """Perform single training step."""
        high_state = self.get_high_state(state)
        reward, new_state = self._train_environment.step(action=None)
        new_high_state = self.get_high_state(new_state)

        if high_state != new_high_state:
            action = self._train_environment.position_state_mapping[new_high_state]

            self._model.step(
                state=high_state,
                action=action,
                reward=reward,
                new_state=new_high_state,
                active=self._train_environment.active,
            )

            self._train_environment._env.cum_reward = 0

        return new_state, reward

    def _find_threat_test(self, rollout: bool, planning: bool):
        """Overrides base runner method.

        Agent starts in shelter and explores/learns in the time until it first reaches
        the threat zone at which point a test rollout is triggered."""
        test_logging_dict = {}

        for i, (map_name, test_env) in enumerate(self._test_environments.items()):

            model_copy = copy.deepcopy(self._model)
            model_copy.env_transition_matrix = test_env.transition_matrix

            if not self._one_dim_blocks:
                model_copy.allow_state_instantiation = True

            for t in range(self._test_num_trials):
                temporary_start_state = random.choice(test_env.reward_positions)
                state = test_env.reset_environment(
                    episode_timeout=np.inf,
                    start_position=temporary_start_state,
                    reward_availability=constants.INFINITE,
                    retain_history=(t != 0),
                )
                # state = self._start_with_reward(model_copy, test_env, t)
                self._train_environment._env.cum_reward = 0

                while state != tuple(test_env.starting_xy):
                    high_state = self.get_high_state(state)
                    reward, new_state = self._train_environment.step(action=None)
                    new_high_state = self.get_high_state(new_state)
                            
                    if high_state != new_high_state:
                        action = test_env.position_state_mapping[new_high_state]

                        model_copy.step(
                            state=high_state,
                            action=action,
                            reward=reward,
                            new_state=new_high_state,
                            active=test_env.active,
                        )
                        self._train_environment._env.cum_reward = 0
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

                # Pad trials with a (0,0) to facilitate post-hoc analysis
                test_env._env._test_episode_position_history.append((0,0))

                test_logging_dict[
                    f"{constants.TEST_EPISODE_REWARD}_{map_name}_{constants.FIND_THREAT_RUN}_{t}"
                ] = reward
                test_logging_dict[
                    f"{constants.TEST_EPISODE_LENGTH}_{map_name}_{constants.FIND_THREAT_RUN}_{t}"
                ] = length

                if (
                    self._visualisations is not None
                    and constants.VALUE_FUNCTION in self._visualisations
                ):
                    try:
                        averaged_values = self._train_environment.average_values_over_positional_states(
                            model_copy.state_action_values
                        )
                        plot_values = {p: max(v) for p, v in averaged_values.items()}
                    except AttributeError:
                        plot_values = model_copy.state_values

                    np.save(
                        os.path.join(
                            self._visualisations_folder_path,
                            f"{self._step_count}_{constants.PRE_TEST}_{constants.VALUES}_{t}",
                        ),
                        plot_values,
                    )

                    self._train_environment.plot_heatmap_over_env(
                        heatmap=plot_values,
                        save_name=os.path.join(
                            self._visualisations_folder_path,
                            f"{self._step_count}_{constants.PRE_TEST}_{t}_{constants.VALUES_PDF}",
                        ),
                    )

        self._test_rollout(
            visualise=rollout,
            save_name_base=f"{constants.INDIVIDUAL_TEST_RUN}_{constants.FIND_THREAT_RUN}",
        )

        return test_logging_dict

    def _runner_specific_visualisations(self):
        if (
            self._visualisations is not None
            and constants.HIERARCHICAL_VALUE_FUNCTION in self._visualisations
        ):
            high_state_value_function = self._model.state_action_values
            low_state_value_function = {}

            state_position_mapping = (
                self._train_environment._env._state_position_mapping
            )
            partitions = self._train_environment._env._partitions
            for (
                centroid,
                coordinates,
            ) in partitions.items():
                for coordinate in coordinates:
                    low_state_value_function[coordinate] = high_state_value_function[
                        state_position_mapping[centroid]
                    ]

            try:
                averaged_values = (
                    self._train_environment.average_values_over_positional_states(
                        low_state_value_function
                    )
                )
                plot_values = {p: max(v) for p, v in averaged_values.items()}

            except AttributeError:
                plot_values = self._model.state_values

            np.save(
                os.path.join(
                    self._visualisations_folder_path,
                    f"{self._step_count}_{constants.VALUES}",
                ),
                plot_values,
            )

            self._train_environment.plot_heatmap_over_env(
                heatmap=plot_values,
                save_name=os.path.join(
                    self._visualisations_folder_path,
                    f"{self._step_count}_{constants.VALUES_PDF}",
                ),
            )


class EpisodicQLearningHierarchyRunner(episodic_runner.EpisodicRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._planner = False

        self._train_step_cost_factor = config.train_super_step_cost_factor
        self._test_step_cost_factor = config.train_super_step_cost_factor

        # make no-op actions infinitely low-value
        # these should then never be updated
        for action in self._train_environment.action_space:
            self._model._state_action_values[action][action] = -np.inf

    def _model_train_step(self, state) -> float:
        """Perform single training step."""
        reward, new_state = self._train_environment.step(action=None)

        action = self._train_environment.position_state_mapping[new_state]

        self._model.step(
            state=state,
            action=action,
            reward=reward,
            new_state=new_state,
            active=self._train_environment.active,
        )

        return new_state, reward

    def _find_threat_test(self, rollout: bool, planning: bool):
        """Overrides base runner method.

        Agent starts in shelter and explores/learns in the time until it first reaches
        the threat zone at which point a test rollout is triggered."""
        test_logging_dict = {}

        for i, (map_name, test_env) in enumerate(self._test_environments.items()):

            model_copy = copy.deepcopy(self._model)
            model_copy.env_transition_matrix = test_env.transition_matrix

            if not self._one_dim_blocks:
                model_copy.allow_state_instantiation = True

            for t in range(self._test_num_trials):

                temporary_start_state = random.choice(test_env.reward_positions)
                state = test_env.reset_environment(
                    episode_timeout=np.inf,
                    start_position=temporary_start_state,
                    reward_availability=constants.INFINITE,
                    retain_history=(t != 0),
                )

                test_env._env._fine_tuning = True

                while state != tuple(test_env.starting_xy):

                    reward, new_state = test_env.step(action=None)

                    action = test_env.position_state_mapping[new_state]

                    model_copy.step(
                        state=state,
                        action=action,
                        reward=reward,
                        new_state=new_state,
                        active=test_env.active,
                    )
                    state = new_state

                test_env._env._fine_tuning = False

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

                # Pad trials with a (0,0) to facilitate post-hoc analysis
                test_env._env._test_episode_position_history.append((0,0))

                test_logging_dict[
                    f"{constants.TEST_EPISODE_REWARD}_{map_name}_{constants.FIND_THREAT_RUN}_{t}"
                ] = reward
                test_logging_dict[
                    f"{constants.TEST_EPISODE_LENGTH}_{map_name}_{constants.FIND_THREAT_RUN}_{t}"
                ] = length

                if (
                    self._visualisations is not None
                    and constants.VALUE_FUNCTION in self._visualisations
                ):
                    try:
                        averaged_values = self._train_environment.average_values_over_positional_states(
                            model_copy.state_action_values
                        )
                        plot_values = {p: max(v) for p, v in averaged_values.items()}
                    except AttributeError:
                        plot_values = model_copy.state_values

                    np.save(
                        os.path.join(
                            self._visualisations_folder_path,
                            f"{self._step_count}_{constants.PRE_TEST}_{constants.VALUES}_{t}",
                        ),
                        plot_values,
                    )

                    self._train_environment.plot_heatmap_over_env(
                        heatmap=plot_values,
                        save_name=os.path.join(
                            self._visualisations_folder_path,
                            f"{self._step_count}_{constants.PRE_TEST}_{t}_{constants.VALUES_PDF}",
                        ),
                    )

        self._test_rollout(
            visualise=rollout,
            save_name_base=f"{constants.INDIVIDUAL_TEST_RUN}_{constants.FIND_THREAT_RUN}",
        )

        return test_logging_dict

    def _find_reward_test(self, rollout: bool, planning: bool):
        """Overrides base runner method.

        Allow agent one more 'period' of exploration to find the reward
        (in the test environment), before test rollout."""
        test_logging_dict = {}

        for i, (map_name, test_env) in enumerate(self._test_environments.items()):

            state = test_env.reset_environment(episode_timeout=np.inf)

            model_copy = copy.deepcopy(self._model)
            model_copy.env_transition_matrix = test_env.transition_matrix

            if not self._one_dim_blocks:
                model_copy.allow_state_instantiation = True

            while test_env.active:

                reward, new_state = test_env.step(action=None)

                action = test_env.position_state_mapping[new_state]

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

        self._test_rollout(
            visualise=rollout,
            save_name_base=f"{constants.INDIVIDUAL_TEST_RUN}_{constants.FINAL_REWARD_RUN}",
        )

        return test_logging_dict

    def _runner_specific_visualisations(self):
        if (
            self._visualisations is not None
            and constants.HIERARCHICAL_VALUE_FUNCTION in self._visualisations
        ):
            high_state_value_function = self._model.state_action_values
            low_state_value_function = {}

            state_position_mapping = (
                self._train_environment._env._state_position_mapping
            )
            partitions = self._train_environment._env._partitions
            for (
                centroid,
                coordinates,
            ) in partitions.items():
                for coordinate in coordinates:
                    low_state_value_function[coordinate] = high_state_value_function[
                        state_position_mapping[centroid]
                    ]

            try:
                averaged_values = (
                    self._train_environment.average_values_over_positional_states(
                        low_state_value_function
                    )
                )
                plot_values = {p: max(v) for p, v in averaged_values.items()}

            except AttributeError:
                plot_values = self._model.state_values

            np.save(
                os.path.join(
                    self._visualisations_folder_path,
                    f"{self._step_count}_{constants.VALUES}",
                ),
                plot_values,
            )

            self._train_environment.plot_heatmap_over_env(
                heatmap=plot_values,
                save_name=os.path.join(
                    self._visualisations_folder_path,
                    f"{self._step_count}_{constants.VALUES_PDF}",
                ),
            )
