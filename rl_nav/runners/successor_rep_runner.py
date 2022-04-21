from rl_nav.runners import episodic_runner, lifelong_runner


class LifelongSRRunner(lifelong_runner.LifelongRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

    def _train_step(self, state) -> float:
        """Perform single training step."""
        pass


class EpisodicSRRunner(episodic_runner.EpisodicRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

    def _train_step(self, state) -> float:
        """Perform single training step."""
        pass
