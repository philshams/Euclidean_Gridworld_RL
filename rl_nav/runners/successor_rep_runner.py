import os

from rl_nav import constants
from rl_nav.runners import episodic_runner, lifelong_runner


class LifelongSRRunner(lifelong_runner.LifelongRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

    def _model_train_step(self, state) -> float:
        """Perform single training step."""
        action = self._model.select_behaviour_action(state, epsilon=self._epsilon)
        reward, new_state = self._train_environment.step(action)

        self._model.step(
            state=state,
            action=action,
            reward=reward,
            new_state=new_state,
            active=self._train_environment.active,
        )

        return new_state, reward

    def _runner_specific_visualisations(self):
        # reward function visualisation
        self._train_environment.plot_heatmap_over_env(
            heatmap=self._model.reward_function,
            save_name=os.path.join(
                self._visualisations_folder_path,
                f"{self._step_count}_{constants.SR_REWARD_FUNCTION_PDF}",
            ),
        )
        # place field of reward state
        for i, reward_pos in enumerate(self._train_environment.reward_positions):
            place_field = self._model.get_place_field(state=reward_pos)

            self._train_environment.plot_heatmap_over_env(
                heatmap=place_field,
                save_name=os.path.join(
                    self._visualisations_folder_path,
                    f"{self._step_count}_{i}_{constants.SR_REWARD_POS_PLACE_FIELD_PDF}",
                ),
            )


class EpisodicSRRunner(episodic_runner.EpisodicRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

    def _model_train_step(self, state) -> float:
        """Perform single training step."""
        action = self._model.select_behaviour_action(state, epsilon=self._epsilon)
        reward, new_state = self._train_environment.step(action)

        self._model.step(
            state=state,
            action=action,
            reward=reward,
            new_state=new_state,
            active=self._train_environment.active,
        )

        return new_state, reward

    def _runner_specific_visualisations(self):
        self._train_environment.plot_heatmap_over_env(
            heatmap=averaged_max_values,
            save_name=os.path.join(
                self._visualisations_folder_path,
                f"{self._step_count}_{constants.VALUES_PDF}",
            ),
        )
