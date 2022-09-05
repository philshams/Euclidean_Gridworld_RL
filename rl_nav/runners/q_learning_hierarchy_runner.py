from rl_nav.runners import episodic_runner, lifelong_runner
import numpy as np


class LifelongQLearningHierarchyRunner(lifelong_runner.LifelongRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._planner = False

        self._train_step_cost_factor = config.train_super_step_cost_factor
        self._test_step_cost_factor = config.train_super_step_cost_factor

    def _model_train_step(self, state) -> float:
        """Perform single training step."""
        original_partition = self._train_environment.get_partition(state)
        new_state = state

        reward = 0

        while self._train_environment.get_partition(new_state) == original_partition:
            action = np.random.choice(self._train_environment.sub_action_space)
            sub_reward, new_state = self._train_environment.step(action)
            reward += sub_reward

        new_partition = self._train_environment.get_partition(new_state)

        original_centroid = self._train_environment.get_centroid(original_partition)
        new_centroid = self._train_environment.get_centroid(new_partition)

        distance = np.sqrt(((original_centroid - new_centroid) ** 2).sum())
        reward -= self._train_step_cost_factor * distance

        self._model.step(
            state=original_partition,
            action=new_partition,
            reward=reward,
            new_state=new_partition,
            active=self._train_environment.active,
        )

        return new_state, reward

    def _runner_specific_visualisations(self):
        pass


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
        # original_partition = self._train_environment.get_partition(state)
        # new_state = state

        # reward = 0
        # while (
        #     self._train_environment.get_partition(new_state) == original_partition
        #     and self._train_environment.active
        # ):
        #     action = self._model.select_behaviour_action(
        #         state, epsilon=self._epsilon.value
        #     )
        #     # action = np.random.choice(self._train_environment.sub_action_space)
        #     sub_reward, new_state = self._train_environment.step(action)
        #     reward += sub_reward

        # new_partition = self._train_environment.get_partition(new_state)

        # import pdb

        # pdb.set_trace()

        # original_centroid = np.array(
        #     self._train_environment.get_centroid(original_partition)
        # )
        # new_centroid = np.array(self._train_environment.get_centroid(new_partition))

        # distance = np.sqrt(((original_centroid - new_centroid) ** 2).sum())
        # reward -= self._train_step_cost_factor * distance

        # action = self._model.select_behaviour_action(state, epsilon=self._epsilon.value)
        reward, new_state = self._train_environment.step(action=None)

        action = self._train_environment.position_state_mapping[new_state]

        print(state, action, new_state)

        self._model.step(
            state=state,
            action=action,
            reward=reward,
            new_state=new_state,
            active=self._train_environment.active,
        )

        return new_state, reward

    def _runner_specific_visualisations(self):
        pass
