from rl_nav.runners import episodic_runner, lifelong_runner


class LifelongQLearningRunner(lifelong_runner.LifelongRunner):
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

        self._step_count += 1

        return new_state, reward


class EpisodicQLearningRunner(episodic_runner.EpisodicRunner):
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

        self._step_count += 1

        return new_state, reward
