from rl_nav import constants
from rl_nav.runners import base_runner


class LifelongRunner(base_runner.BaseRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

    def _get_data_columns(self):
        columns = [
            constants.STEP,
            constants.TRAIN_REWARD,
        ]
        for i in range(len(self._test_environments)):
            columns.append(f"{constants.TEST_EPISODE_REWARD}_{i}")
            columns.append(f"{constants.TEST_EPISODE_LENGTH}_{i}")
        return columns

    def train(self):

        state = self._train_environment.reset_environment()
        train_reward = 0

        while self._step_count < self._num_steps:
            if self._step_count >= self._next_rollout_step:
                self._test_rollout()
            if self._step_count >= self._next_visualisation_step:
                self._generate_visualisations()
            if self._step_count >= self._next_test_step:
                logging_dict = self._test()
            else:
                logging_dict = {}
            if self._step_count >= self._next_checkpoint_step:
                self._data_logger.checkpoint()
                while self._next_checkpoint_step <= self._step_count:
                    self._next_checkpoint_step += self._checkpoint_frequency

            action = self._model.select_behaviour_action(state, epsilon=self._epsilon)
            reward, new_state = self._train_environment.step(action)

            self._model.step(
                state=state,
                action=action,
                reward=reward,
                new_state=new_state,
                active=self._train_environment.active,
            )
            state = new_state
            train_reward += reward

            self._step_count += 1

            logging_dict[constants.STEP] = self._step_count
            logging_dict[constants.TRAIN_REWARD] = train_reward
            self._log_episode(step=self._step_count, logging_dict=logging_dict)
