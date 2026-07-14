from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn
from torch import Tensor


T = TypeVar("T")


class Processor(nn.Module, ABC):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        tokens: Tensor,
        use_checkpoint: bool = True,
    ) -> Tensor:
        pass

