from dataclasses import dataclass, field

from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    # Image shape can be either [H, W] or a single int for square images
    image_shape: list[int] | int
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg

    def __post_init__(self):
        """Convert image_shape to [H, W] format if it's a single int."""
        if isinstance(self.image_shape, int):
            self.image_shape = [self.image_shape, self.image_shape]
