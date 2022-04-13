import abc


class BaseRunner(abc.ABC):
    def __init__(self, model, environment):
        self._model = model
        self._environment = environment

    @abc.abstractmethod
    def train(self):
        pass
