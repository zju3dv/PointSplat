import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def matrix_to_quaternion(
    matrix: Float[Tensor, "*batch 3 3"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 4"]:
    """
    Convert rotation matrices to quaternions.
    Order matches scipy format (x, y, z, w).

    Based on https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    batch_shape = matrix.shape[:-2]

    # Extract diagonal and off-diagonal elements
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    trace = m00 + m11 + m22

    # We need to handle 4 cases based on which component is largest
    # to avoid numerical instabilities

    def case0():
        # w is largest
        s = torch.sqrt(1.0 + trace + eps) * 2  # s = 4 * w
        w = 0.25 * s
        x = (m21 - m12) / (s + eps)
        y = (m02 - m20) / (s + eps)
        z = (m10 - m01) / (s + eps)
        return torch.stack([x, y, z, w], dim=-1)

    def case1():
        # x is largest
        s = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2  # s = 4 * x
        w = (m21 - m12) / (s + eps)
        x = 0.25 * s
        y = (m01 + m10) / (s + eps)
        z = (m02 + m20) / (s + eps)
        return torch.stack([x, y, z, w], dim=-1)

    def case2():
        # y is largest
        s = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2  # s = 4 * y
        w = (m02 - m20) / (s + eps)
        x = (m01 + m10) / (s + eps)
        y = 0.25 * s
        z = (m12 + m21) / (s + eps)
        return torch.stack([x, y, z, w], dim=-1)

    def case3():
        # z is largest
        s = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2  # s = 4 * z
        w = (m10 - m01) / (s + eps)
        x = (m02 + m20) / (s + eps)
        y = (m12 + m21) / (s + eps)
        z = 0.25 * s
        return torch.stack([x, y, z, w], dim=-1)

    # Determine which case to use for each matrix
    mask0 = (trace > m00) & (trace > m11) & (trace > m22)
    mask1 = (m00 > m11) & (m00 > m22) & ~mask0
    mask2 = (m11 > m22) & ~mask0 & ~mask1
    mask3 = ~mask0 & ~mask1 & ~mask2

    quaternions = torch.zeros((*batch_shape, 4), dtype=matrix.dtype, device=matrix.device)

    if mask0.any():
        quaternions[mask0] = case0()[mask0]
    if mask1.any():
        quaternions[mask1] = case1()[mask1]
    if mask2.any():
        quaternions[mask2] = case2()[mask2]
    if mask3.any():
        quaternions[mask3] = case3()[mask3]

    return quaternions


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )
