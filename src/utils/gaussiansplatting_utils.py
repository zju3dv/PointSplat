from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor


"""
Adapted from the original 3D GS implementation, which opacity, rotation and scale is not activated
https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py
"""
def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    save_path: Path,
    opacity_threshold=None
):
    from plyfile import PlyData, PlyElement
    # These attibutes have already been activated, we need to deactivate them first before saving
    scales_deactivated = scales.log()
    opacities_deactivated = torch.logit(opacities).unsqueeze(-1)

    N = means.shape[0]
    xyz = means
    feature = harmonics.transpose(1, 2)
    normal = torch.zeros_like(xyz)  # (N, 3)
    f_dc = feature[:, 0].contiguous()
    f_rest_full = torch.zeros(N, 3*(3+1)**2-3).float()
    if feature.shape[1] > 1:
        f_rest = feature[:, 1:].transpose(1, 2).reshape(N, -1)  # (N, 3*(sh_degree+1)**2-3)
        # NOTE: for SH degree > 3, we only save the first 3 bands
        if f_rest.shape[1] > f_rest_full.shape[1]:
            f_rest = f_rest[:, :f_rest_full.shape[1]]
        f_rest_full[:, :f_rest.shape[1]] = f_rest
    f_rest_full = f_rest_full.contiguous()
    attributes = np.concatenate([xyz.numpy(),
                                 normal.numpy().astype(np.uint8),
                                 f_dc.numpy(),
                                 f_rest_full.numpy(),
                                 opacities_deactivated.numpy(),
                                 scales_deactivated.numpy(),
                                 rotations.numpy()
                                 ], axis=1)
    if opacity_threshold is not None:
        attributes = attributes[opacities_deactivated.squeeze(
            -1).sigmoid().numpy() > opacity_threshold]
    attribute_list = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    attribute_list += ['f_dc_{}'.format(i) for i in range(f_dc.shape[1])]
    attribute_list += ['f_rest_{}'.format(i)
                       for i in range(f_rest_full.shape[1])]
    attribute_list += ['opacity']
    attribute_list += ['scale_{}'.format(i) for i in range(scales_deactivated.shape[1])]
    attribute_list += ['rot_{}'.format(i) for i in range(rotations.shape[1])]
    dtype_full = [(attribute, 'f4') for attribute in attribute_list]
    dtype_full[3:6] = [(attribute, 'u1') for attribute in attribute_list[3:6]]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    save_path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([el]).write(save_path)

def save_gaussians(gaussians,
                      gaussians_mask,
                      example, save_path,
                      transform_to_cam0_space=False,
                      trim_border=True):

    v, _, h, w = example["context"]["image"].shape[1:]
    cam_0_c2w = example["context"]["extrinsics"][0, 0].detach().cpu()

    assert gaussians.means.shape[0] == 1, "Must be single batch"
    means = gaussians.means[0].detach().cpu()
    scales = gaussians.scales[0].detach().cpu()
    rotations = gaussians.rotations[0].detach().cpu()
    harmonics = gaussians.harmonics[0].detach().cpu()
    opacities = gaussians.opacities[0].detach().cpu()
    gaussians_num = means.shape[0]

    mask = torch.ones(gaussians_num, dtype=torch.bool)
    if gaussians_mask is not None:
        mask = mask & (gaussians_mask[0].detach().cpu()>0).bool().view(-1)

    is_gs_map = (gaussians_num // (v * h * w) * (v * h * w) == gaussians_num)
    if is_gs_map and trim_border:
        # Trick in Splat series
        spp = gaussians_num // (v * h * w)

        # Create a mask to filter the Gaussians. throw away Gaussians at the
        # borders, since they're generally of lower quality.
        center_mask = torch.zeros((h, w, spp, v), dtype=torch.bool)
        GAUSSIAN_TRIM = 8
        center_mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1
        mask = mask & center_mask.view(-1)

    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    harmonics = harmonics[mask]
    opacities = opacities[mask]

    if transform_to_cam0_space:
        from scipy.spatial.transform import Rotation as R
        view_rotation = cam_0_c2w[:3, :3].inverse()
        # Apply the rotation to the means (Gaussian positions).
        means = einsum(view_rotation, means, "i j, ... j -> ... i")

        # Apply the rotation to the Gaussian rotations.
        rotations = R.from_quat(rotations).as_matrix()
        rotations = view_rotation @ rotations
        rotations = R.from_matrix(rotations).as_quat()
        x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
        rotations = np.stack((w, x, y, z), axis=-1)
        rotations = torch.from_numpy(rotations)

        # Since our axes are swizzled for the spherical harmonics, we only export the DC band
        harmonics_view_invariant = harmonics[..., 0]
        harmonics = harmonics_view_invariant[..., None]

    export_ply(
        means,
        scales,
        rotations,
        harmonics,
        opacities,
        save_path,
    )


