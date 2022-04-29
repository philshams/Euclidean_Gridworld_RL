from rl_nav.runners import episodic_runner, lifelong_runner


class LifelongAStarRunner(lifelong_runner.LifelongRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._deltas = self._train_environment.action_deltas
        self._model.deltas = self._deltas

        self._planner = True

    def _model_train_step(self, state) -> float:
        """Perform single training step."""
        action = self._model.select_behaviour_action(state, epsilon=1)
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
        pass


class EpisodicAStarRunner(episodic_runner.EpisodicRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._deltas = self._train_environment.action_deltas
        self._model.deltas = self._deltas

        self._planner = True

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
        pass
