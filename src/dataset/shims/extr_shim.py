import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Optional

from ..types import AnyExample, AnyViews


def preprocess_poses(
    in_c2ws: Float[Tensor, "batch 4 4"],
    scene_scale_factor: float = 1.35,
    ply_file_path: Optional[str] = None,
    align_to_dna_rendering: bool = False,
    return_transform: bool = False,
    return_scale: bool = False,
):
    """
    Preprocess the poses to:
    1. translate and rotate the scene to align the average camera direction and position
    2. rescale the whole scene to a fixed scale

    Args:
        in_c2ws: Input camera-to-world matrices
        scene_scale_factor: Factor to scale the scene
        ply_file_path: Optional path to ply file (unused, for compatibility)

    Returns:
        Normalized camera-to-world matrices
    """
    # Translation and Rotation
    # align coordinate system (OpenCV coordinate) to the mean camera
    # center is the average of all camera centers
    # average direction vectors are computed from all camera direction vectors (average down and forward)
    center = in_c2ws[:, :3, 3].mean(0)
    # average forward direction (z of opencv camera)
    avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)
    # average down direction (y of opencv camera)
    avg_down = in_c2ws[:, :3, 1].mean(0)
    avg_right = F.normalize(torch.cross(
        avg_down, avg_forward, dim=-1), dim=-1)  # (x of opencv camera)
    # (y of opencv camera)
    avg_down = F.normalize(torch.cross(
        avg_forward, avg_right, dim=-1), dim=-1)

    avg_pose = torch.eye(4, device=in_c2ws.device)  # average c2w matrix
    avg_pose[:3, :3] = torch.stack(
        [avg_right, avg_down, avg_forward], dim=-1)
    avg_pose[:3, 3] = center
    avg_pose = torch.linalg.inv(avg_pose)  # average w2c matrix
    in_c2ws = avg_pose @ in_c2ws

    if align_to_dna_rendering:
        dna_rendering_avg_pose_w2c = torch.tensor(
            [
                [0.8474, -0.0032, 0.5309, -0.0703],
                [0.5292, 0.0850, -0.8442, 0.1167],
                [-0.0424, 0.9964, 0.0737, -0.0271],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ],
            device=in_c2ws.device,
            dtype=in_c2ws.dtype,
        )
        alignment_transform = dna_rendering_avg_pose_w2c @ avg_pose.inverse()
        in_c2ws = torch.einsum("ij,vjk->vik", alignment_transform, in_c2ws)

    # Rescale the whole scene to a fixed scale
    scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
    scene_scale = scene_scale_factor * scene_scale

    in_c2ws[:, :3, 3] /= scene_scale

    return_list = [in_c2ws]
    if return_transform:
        return_list.append(alignment_transform)
    if return_scale:
        return_list.append(torch.max(torch.abs(in_c2ws[:, :3, 3])))
    if len(return_list) > 1:
        return tuple(return_list)
    return in_c2ws


def preprocess_sampled_views(
    context_c2ws: Float[Tensor, "num_context 4 4"],
    target_c2ws: Float[Tensor, "num_target 4 4"],
    scene_scale: Float[Tensor, ""],
) -> tuple[Float[Tensor, "num_context 4 4"], Float[Tensor, "num_target 4 4"]]:
    center = context_c2ws[:, :3, 3].mean(0)
    avg_forward = F.normalize(context_c2ws[:, :3, 2].mean(0), dim=-1)
    avg_down = context_c2ws[:, :3, 1].mean(0)
    avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
    avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

    avg_pose = torch.eye(4, device=context_c2ws.device)
    avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
    avg_pose[:3, 3] = center
    avg_pose = torch.linalg.inv(avg_pose)

    processed_context_c2ws = avg_pose @ context_c2ws
    processed_target_c2ws = avg_pose @ target_c2ws

    scene_scale_sampled = torch.max(torch.abs(processed_context_c2ws[:, :3, 3]))
    scale = scene_scale_sampled / scene_scale

    processed_context_c2ws[:, :3, 3] /= scale
    processed_target_c2ws[:, :3, 3] /= scale

    return processed_context_c2ws, processed_target_c2ws


def apply_extr_shim_to_views(
    views: AnyViews,
    scene_scale_factor: float = 1.35
) -> AnyViews:
    """Apply extrinsic normalization to views."""
    normalized_extrinsics = preprocess_poses(
        views["extrinsics"],
        scene_scale_factor=scene_scale_factor
    )
    return {
        **views,
        "extrinsics": normalized_extrinsics,
    }


def apply_extr_shim(
    example: AnyExample,
    scene_scale_factor: float = 1.35
) -> AnyExample:
    """
    Apply extrinsic normalization to both context and target views.

    Args:
        example: Dataset example with context and target views
        scene_scale_factor: Factor to scale the scene

    Returns:
        Example with normalized extrinsics
    """
    # Combine all extrinsics for normalization
    all_extrinsics = torch.cat([
        example["context"]["extrinsics"],
        example["target"]["extrinsics"]
    ], dim=0)

    # Normalize all extrinsics together
    all_normalized = preprocess_poses(all_extrinsics, scene_scale_factor=scene_scale_factor)

    # Split back to context and target
    num_context = len(example["context"]["extrinsics"])

    return {
        **example,
        "context": {
            **example["context"],
            "extrinsics": all_normalized[:num_context],
        },
        "target": {
            **example["target"],
            "extrinsics": all_normalized[num_context:],
        },
    }
