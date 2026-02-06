"""Abstract base class for task datasets."""
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class TaskDataset(ABC, Dataset):
    """Interface for classification task datasets.

    Subclasses must implement __len__ and __getitem__, and expose
    `num_classes` for the model to know output dimensionality.
    """

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        ...
