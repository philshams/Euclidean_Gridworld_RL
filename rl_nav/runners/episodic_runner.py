import os
from typing import Any, Dict, Optional, Union

from rl_nav import constants
from rl_nav.models import q_learning
from rl_nav.runners import base_runner
from rl_nav.utils import model_utils


class EpisodicRunner(base_runner.BaseRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._episode_count = 0

    def _get_data_columns(self):
        columns = [
            constants.STEP,
            constants.TRAIN_EPISODE_REWARD,
            constants.TRAIN_EPISODE_LENGTH,
        ]
        return columns

    def _setup_model(self, config):
        """Instantiate model specified in configuration."""
        initialisation_strategy = model_utils.get_initialisation_strategy(config)
        model = q_learning.QLearner(
            action_space=self._environment.action_space,
            state_space=self._environment.state_space,
            behaviour=config.behaviour,
            target=config.target,
            initialisation_strategy=initialisation_strategy,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
        )
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
        averaged_values = (
            self._environment.average_values_over_positional_states(
                self._model.state_action_values
            )
        )
        averaged_visitation_counts = (
            self._environment.average_values_over_positional_states(
                self._model.state_visitation_counts
            )
        )

        averaged_max_values = {p: max(v) for p, v in averaged_values.items()}
        
        self._environment.plot_heatmap_over_env(
            heatmap=averaged_max_values,
            save_name=os.path.join(
                self._visualisations_folder_path,
                f"{self._step_count}_{constants.VALUES_PDF}",
            ),
        )
    
        self._environment.plot_heatmap_over_env(
            heatmap=averaged_visitation_counts,
            save_name=os.path.join(
                self._visualisations_folder_path,
                f"{self._step_count}_{constants.VISITATION_COUNTS_PDF}",
            ),
        )
        while self._next_visualisation_step <= self._step_count:
            self._next_visualisation_step += self._visualisation_frequency

    def _generate_rollout(self):
        self._environment.visualise_episode_history(
            save_path=os.path.join(
                self._rollout_folder_path,
                f"{constants.INDIVIDUAL_TRAIN_RUN}_{self._step_count}.mp4",
            )
        )
        while self._next_rollout_step <= self._step_count:
            self._next_rollout_step += self._rollout_frequency

    def train(self):
        while self._step_count < self._num_steps:
            if self._step_count > self._next_rollout_step:
                self._generate_rollout()
            if self._step_count > self._next_visualisation_step:
                self._generate_visualisations()
            episode_logging_dict = self._train_episode()
            self._log_episode(
                step=self._step_count, logging_dict=episode_logging_dict
            )
            # import pdb; pdb.set_trace()
            self._data_logger.checkpoint()

    def _train_episode(self) -> Dict[str, Any]:
        """Perform single training loop.

        Args:
            episode: index of episode

        Returns:
            logging_dict: dictionary of items to log (e.g. episode reward).
        """
        self._episode_count += 1

        episode_reward = 0

        state = self._environment.reset_environment(train=True)

        while self._environment.active and self._step_count < self._num_steps:

            action = self._model.select_behaviour_action(state, epsilon=self._epsilon)
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

            self._step_count += 1

        logging_dict = {
            constants.STEP: self._step_count,
            constants.TRAIN_EPISODE_REWARD: episode_reward,
            constants.TRAIN_EPISODE_LENGTH: self._environment.episode_step_count,
        }

        return logging_dict
