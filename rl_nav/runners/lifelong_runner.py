from rl_nav import constants
from rl_nav.runners import base_runner


class LifelongRunner(base_runner.BaseRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

    def _get_runner_specific_data_columns(self):
        return [constants.TRAIN_REWARD]

    def train(self):

        state = self._train_environment.reset_environment()
        train_reward = 0

        while self._step_count < self._num_steps:

            state, reward, logging_dict = self._train_step(state=state)
            train_reward += reward

            logging_dict[constants.STEP] = self._step_count
            logging_dict[constants.TRAIN_REWARD] = train_reward
            self._log_episode(step=self._step_count, logging_dict=logging_dict)
            if (
                self._step_count % self._checkpoint_frequency == 0
                and self._step_count != 1
            ):
                self._data_logger.checkpoint()
