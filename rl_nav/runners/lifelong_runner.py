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
            # if self._step_count >= self._next_rollout_step:
            #     self._test_rollout(save_name_base=constants.INDIVIDUAL_TEST_RUN)
            if self._step_count >= self._next_visualisation_step:
                self._generate_visualisations()
            if self._step_count >= self._next_test_step:
                plain_logging_dict = self._test(self._model)
                find_reward_logging_dict = self._find_reward_test()
                logging_dict = {**plain_logging_dict, **find_reward_logging_dict}
            else:
                logging_dict = {}
            if self._step_count >= self._next_checkpoint_step:
                self._data_logger.checkpoint()
                while self._next_checkpoint_step <= self._step_count:
                    self._next_checkpoint_step += self._checkpoint_frequency

            state, reward = self._train_step(state=state)
            train_reward += reward

            logging_dict[constants.STEP] = self._step_count
            logging_dict[constants.TRAIN_REWARD] = train_reward
            self._log_episode(step=self._step_count, logging_dict=logging_dict)
