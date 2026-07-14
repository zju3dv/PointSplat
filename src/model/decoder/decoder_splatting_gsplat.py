from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import math

from ...dataset import DatasetCfg
from ..types import Gaussians
from .decoder import Decoder, DecoderOutput
from ..encoder import EncoderOutput

def render_gsplat(
    render_func,
    xyz: Float[Tensor, "batch gaussian 3"],
    color: Float[Tensor, "batch gaussian *C"],
    scale: Float[Tensor, "batch gaussian 3"],
    rotation: Float[Tensor, "batch gaussian 4"],
    opacity: Float[Tensor, "batch gaussian"],
    test_c2ws: Float[Tensor, "batch 4 4"],
    test_intr: Float[Tensor, "batch 3 3"],
    W: int,
    H: int,
    near_plane: float = 0.01,
    far_plane: float = 100000,
    sh_degree: Optional[int] = None,
    use_for_loop: bool = False,
    backgrounds: Float[Tensor, "batch 3"] | None = None,
) -> Float[Tensor, "batch 1 height width 3"]:
    B = test_intr.shape[0]
    N = xyz.shape[1]

    if sh_degree is None:
        assert color.shape == (B, N, 3)

    test_w2c = test_c2ws.float().inverse().unsqueeze(1)  # (B, 1, 4, 4)
    test_intr = test_intr.unsqueeze(1)  # (B, 1, 3, 3)

    if use_for_loop:
        renderings_list = []
        for b in range(B):
            mask = opacity[b] > 0.001
            pruned_xyz = xyz[b][mask]
            pruned_rotation = rotation[b][mask]
            pruned_scale = scale[b][mask]
            pruned_opacity = opacity[b][mask]
            pruned_color = color[b][mask]
            renderings, _, _ = render_func(
                pruned_xyz, pruned_rotation, pruned_scale, pruned_opacity, pruned_color,
                test_w2c[b], test_intr[b], W, H, sh_degree=sh_degree,
                near_plane=near_plane, far_plane=far_plane,
                backgrounds=backgrounds[b],
                render_mode="RGB",
                rasterize_mode='classic')
            renderings_list.append(renderings)
        renderings = torch.stack(renderings_list, dim=0)
    else:
        renderings, _, _ = render_func(xyz, rotation, scale, opacity, color,
                                        test_w2c, test_intr, W, H, sh_degree=sh_degree,
                                        near_plane=near_plane, far_plane=far_plane,
                                        # gsplat expects a shared background in the batched path.
                                        backgrounds=backgrounds[0],
                                        render_mode="RGB",
                                        rasterize_mode='classic')
    return renderings  # (B, 1, H, W, 3)

@dataclass
class DecoderSplattingGSplatCfg:
    name: Literal["splatting_gsplat"]
    scale_invariant: bool
    use_for_loop: bool


class DecoderSplattingGSplat(Decoder[DecoderSplattingGSplatCfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingGSplatCfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )
        from gsplat import rasterization
        self.render_func = rasterization
        self.scale_invariant = cfg.scale_invariant
        self.use_for_loop = cfg.use_for_loop

    def forward(
        self,
        encoder_output: EncoderOutput,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: Optional[Literal["depth", "disparity", "relative_disparity", "log"]] = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        gaussians: Gaussians = encoder_output.gaussians
        gaussians_mask = encoder_output.gaussians_mask
        gs_num = gaussians.means.shape[1]

        assert gaussians.scales is not None
        assert gaussians.rotations is not None

        extrinsics = rearrange(extrinsics, "b v i j -> (b v) i j")
        intrinsics = rearrange(intrinsics, "b v i j -> (b v) i j")
        near = rearrange(near, "b v -> (b v)")
        far = rearrange(far, "b v -> (b v)")
        background_color = repeat(self.background_color, "c -> (b v) c", b=b, v=v)
        gaussian_means = repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v)
        gaussian_scales = repeat(gaussians.scales, "b g xyz -> (b v) g xyz", v=v)
        gaussian_rotations = repeat(gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=v)
        harmonics = rearrange(gaussians.harmonics, "b g c d_sh -> b g d_sh c")
        gaussian_sh_coefficients = repeat(harmonics, "b g d_sh c -> (b v) g d_sh c", v=v)
        gaussian_opacities = repeat(gaussians.opacities, "b g -> (b v) g", v=v)
        gaussian_mask = repeat(gaussians_mask, "b g c-> (b v) g c", v=v) if gaussians_mask is not None else None

        if self.scale_invariant:
            scale = 1 / near # shape: [B*V]
            extrinsics = extrinsics.clone()
            extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[..., None]
            near = near * scale
            far = far * scale
            gaussian_scales = gaussian_scales * scale[..., None, None]
            gaussian_means = gaussian_means * scale[..., None, None]

        # denorm intrinsics for gsplat
        if intrinsics[..., 0, 2].min() < 1.0:
            # cx < 1.0 means normalized intrinsics, unlike splatting_cuda that accept fov, gsplat accept original intrinsics
            intrinsics = intrinsics.clone()
            intrinsics[..., 0, :] *= image_shape[1]
            intrinsics[..., 1, :] *= image_shape[0]

        sh_degree = int(round(math.sqrt(gaussians.harmonics.shape[-1])) - 1)
        if gaussian_mask is not None:
            gaussian_opacities = gaussian_opacities.clone()
            gaussian_opacities *= gaussian_mask.squeeze(-1)

        with torch.autocast(enabled=False, device_type="cuda"):
            renderings = render_gsplat(
                self.render_func,
                xyz=gaussian_means.float(),
                color=gaussian_sh_coefficients.float(),
                scale=gaussian_scales.float(),
                rotation=gaussian_rotations.float(),
                opacity=gaussian_opacities.float(),
                test_c2ws=extrinsics.float(),
                test_intr=intrinsics.float(),
                W=image_shape[1],
                H=image_shape[0],
                near_plane=float(near.min().float().item()),
                far_plane=float(far.max().float().item()),
                sh_degree=sh_degree,
                use_for_loop=self.use_for_loop,
                backgrounds=background_color.float()
            )

        renderings = rearrange(renderings, "(b v) 1 h w c -> b v c h w", b=b, v=v)

        return DecoderOutput(
            renderings,
            None,
        )
