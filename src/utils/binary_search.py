from abc import ABC, abstractmethod
import numpy as np

class BinarySearchFunctor(ABC):
    @abstractmethod
    def __call__(self, values: np.ndarray) -> np.ndarray:
        """
        Compute the function that the binary search will use.
        This method should be overridden by subclasses.
        """
        pass

    @abstractmethod
    def get_initial_bounds(self, size: int) -> (np.ndarray, np.ndarray):
        """
        Generate the initial low and high bounds for the binary search.
        This method should be overridden by subclasses.
        """
        pass


def parallel_binary_search(functor: BinarySearchFunctor, size: int, num_iterations: int = 20) -> np.ndarray:
    low, high = functor.get_initial_bounds(size)

    for _ in range(num_iterations):
        mid = (low + high) / 2
        is_above_target = functor(mid)
        low = np.where(is_above_target, low, mid)
        high = np.where(is_above_target, mid, high)

    return (low + high) / 2

