from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from typing import Optional
from torch import Tensor, nn
import torch.nn.functional as F


from .gaussians import build_covariance, quaternion_to_matrix, matrix_to_quaternion
from ...types import Gaussians
from ....utils.graphics_utils import get_world_rays,rotate_sh

@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def scale_activation(self,
                         scales: Float[Tensor, "batch view r srf 1 3"],
                         intrinsics: Float[Tensor, "batch view 1 1 1 3 3"],
                         depths: Float[Tensor, "batch view r 1 1"],
                         image_shape: tuple[int, int]) -> Float[Tensor, "batch view r srf 1 3"]:
        return torch.clamp(F.softplus(scales - 4.),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            ) # [b, v, r, srf, 1, 3]

    def build_covariance(self,
                         scales: Float[Tensor, "*#batch 3"],
                         rotations: Float[Tensor, "*#batch 4"],
                         ) -> Float[Tensor, "*#batch 3 3"]:
        return build_covariance(scales, rotations)

    def forward(
        self,
        extrinsics: Float[Tensor, "batch view 1 1 1 4 4"],
        intrinsics: Float[Tensor, "batch view 1 1 1 3 3"] | None,
        coordinates: Float[Tensor, "batch view r srf 1 2"],
        depths: Float[Tensor, "batch view r 1 1"] | None,
        opacities: Float[Tensor, "batch view r srf 1"],
        raw_gaussians: Float[Tensor, "batch view r srf 1 c"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        scales = self.scale_activation(scales, intrinsics, depths, image_shape) # [b, v, r, srf, 1, 3]

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps) # [b, v, r, srf, 1, 4]

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask # [b, v, r, srf, 1, 3, d_sh]

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Create world-space covariance matrices.
        covariances = self.build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3] # [b, v, 1, 1, 1, 3, 3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2) # [b, v, r, srf, 1, 3, 3]

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None] # [b, v, r, srf, 1, 3]

        # rotations = rotations.broadcast_to((*scales.shape[:-1], 4)) # [b, v, r, srf, 1, 4]

        # Convert rotations from camera space to world space
        # 1. Convert quaternions to rotation matrices
        rotation_matrices = quaternion_to_matrix(rotations)  # [b, v, r, srf, 1, 3, 3]
        # 2. Apply camera-to-world rotation (broadcasting will handle dimension alignment)
        world_rotation_matrices = c2w_rotations @ rotation_matrices  # [b, v, r, srf, 1, 3, 3]
        # 3. Convert back to quaternions
        world_rotations = matrix_to_quaternion(world_rotation_matrices)  # [b, v, r, srf, 1, 4]

        # These are already in world space
        gaussians = Gaussians(
            rearrange(
                means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                rotate_sh(sh, c2w_rotations[..., None, :, :]),
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            scales=rearrange(
                scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            ),
            rotations=rearrange(
                world_rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            ),
        )
        return gaussians

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh

