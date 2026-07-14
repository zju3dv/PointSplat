from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import nn
from torch import Tensor
from ...dataset.types import BatchedViews, DataShim
from ..types import Gaussians

from dataclasses import dataclass
T = TypeVar("T")

@dataclass
class EncoderOutput:
    context: BatchedViews | None = None
    gaussians: Gaussians | None = None
    gaussians_mask: Float[Tensor, "batch gaussian 1"] | None = None
    depth: Float[Tensor, "batch view height width"] | None = None
    pose: dict[str, Float[Tensor, "batch view 4 4"] | Float[Tensor, "batch view 3 3"]] | None = None


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
    ) -> EncoderOutput:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
