import abc
from typing import Union, List
import numpy as np


class LearningRateSchedule(abc.ABC):
    def __init__(self, value: Union[int, float]):
        self._value = value

    @abc.abstractmethod
    def __next__(self):
        pass

    @property
    def value(self):
        assert (
            self._value > 0
        ), f"Learning rate must be positive, currently is {self._value}"
        return self._value


class ConstantLearningRate(LearningRateSchedule):
    def __init__(self, value: Union[int, float]):
        super().__init__(value=value)

    def __next__(self):
        pass


class LinearDecayLearningRate(LearningRateSchedule):
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


class HardCodedLearningRate(LearningRateSchedule):
    def __init__(self, values: List[Union[int, float]], timesteps: List[int]):

        assert (
            len(timesteps) == len(values) - 1
        ), "number of timestep changes must be one fewer than number of values."

        self._values = iter(values)
        self._timesteps = iter(timesteps)
        self._next_change = next(self._timesteps)

        self._step_count = 0

        super().__init__(value=next(self._values))

    def __next__(self):
        self._step_count += 1
        if self._step_count == self._next_change:
            self._value = next(self._values)

            try:
                self._next_change = next(self._timesteps)
            except StopIteration:
                self._next_change = np.inf
