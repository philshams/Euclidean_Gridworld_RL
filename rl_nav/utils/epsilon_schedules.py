import abc
from typing import Union


class EpsilonSchedule(abc.ABC):
    def __init__(self, value: Union[int, float]):
        self._value = value

    @abc.abstractmethod
    def __next__(self):
        pass

    @property
    def value(self):
        assert (
            self._value <= 1 and self._value >= 0
        ), f"Epsilon must be between 0 and 1, currently is {self._value}"
        return self._value


class ConstantEpsilon(EpsilonSchedule):
    def __init__(self, value: Union[int, float]):
        super().__init__(value=value)

    def __next__(self):
        pass


class LinearDecayEpsilon(EpsilonSchedule):
    def __init__(
        self,
        initial_value: Union[int, float],
        final_value: Union[int, float],
        anneal_duration: int,
    ):

        self._step_size = (initial_value - final_value) / anneal_duration
        self._final_value = final_value

        super().__init__(value=initial_value)

    def __next__(self):
        if self._value > self._final_value:
            self._value -= self._step_size
