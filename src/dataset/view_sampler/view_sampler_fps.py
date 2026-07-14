from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import random
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerFPSCfg:
    name: Literal["fps"]
    num_context_views: int
    num_target_views: int


class ViewSamplerFPS(ViewSampler[ViewSamplerFPSCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        """Arbitrarily sample context and target views."""
        num_views, _, _ = extrinsics.shape

        # Sort cameras based on Euclidean distance from camera 0
        # camera_0_pos = extrinsics[0, :3, 3]  # Position of camera 0
        camera_0_pos = torch.tensor([ 0.5725,  0.3874, -0.2389], device=extrinsics.device)
        distances = torch.norm(extrinsics[:, :3, 3] - camera_0_pos, dim=-1)  # Distance from camera 0 to each camera
        sorted_indices = torch.argsort(distances)  # Sort by distance (camera 0 will be first)

        if self.cfg.num_context_views == 1:
            input_index = [num_views // 2]
        else:
            step = (num_views - 1) / (self.cfg.num_context_views - 1) if self.cfg.num_context_views > 1 else 1
            input_index = [round(i * step) for i in range(self.cfg.num_context_views)]
        index_context_at_sorted = input_index
        available_target_indices = list(range(num_views))
        index_target_at_sorted = random.sample(available_target_indices, self.cfg.num_target_views)

        index_context = sorted_indices[index_context_at_sorted]
        index_target = sorted_indices[index_target_at_sorted]

        return index_context, index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
