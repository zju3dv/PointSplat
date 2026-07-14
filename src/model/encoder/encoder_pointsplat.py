from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from easydict import EasyDict as edict


# Type Imports
from typing import Literal, Optional, List, Tuple
from jaxtyping import Float
from torch import Tensor
from ..types import Gaussians
from ...dataset.types import BatchedViews, DataShim, BatchedExample
from .encoder import Encoder, EncoderOutput

from ..blocks.layers.pointsplat_layers import PatchEmbed3D, GSLayer, Tokenizer
from ..blocks.pointimage_transformer import PointImageMMJointTransformerBlock
from .attention_processor.lrm_processor import _init_weights as init_weights
from .common.gaussians import build_covariance

@dataclass
class ImageTokenizerCfg:
    patch_size: int
    in_channels: int
    use_ln: bool

@dataclass
class TransformerCfg:
    d: int
    d_head: int
    n_layer: int
    input_ln: bool
    output_ln: bool


@dataclass
class GaussiansDecoderCfg:
    num_target: int
    hidden_dim: int
    use_rgb: bool
    sh_degree: int
    clip_scaling: List[float]
    scale_bias: float
    init_density: float
    xyz_offset_max_step: float
    fix_rotation: bool
    activation: str
    n_hidden_layers: int
    n_neurons: int
    prune_ratio: float

@dataclass
class EncoderPointsplatCfg:
    name: Literal["pointsplat"]
    image_tokenizer: ImageTokenizerCfg
    vhull_res: int
    bounds: List[float]
    min_cell_size: float
    cube_sample: bool
    threshold: float
    point_patch_size: int
    transformer: TransformerCfg
    gaussians_decoder: GaussiansDecoderCfg

def get_camera_ray_dir(
    image: Float[Tensor, "*#batch view 3 H W"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Tuple[Tensor, Tensor]:

    dtype = image.dtype
    device = image.device
    B, V, _, H, W = image.shape
    input_c2ws, input_intr_raw = extrinsics, intrinsics

    # Reshape the intrinsics
    fx = input_intr_raw[..., 0, 0].unsqueeze(-1) # (B, V, 1)
    fy = input_intr_raw[..., 1, 1].unsqueeze(-1) # (B, V, 1)
    cx = input_intr_raw[..., 0, 2].unsqueeze(-1) # (B, V, 1)
    cy = input_intr_raw[..., 1, 2].unsqueeze(-1) # (B, V, 1)

    input_intr = torch.cat([fx, fy, cx, cy], dim=-1)

    # Embed camera info
    ray_o = input_c2ws[:, :, :3, 3].unsqueeze(2).expand(-1, -1, H * W, -1).float() # (B, V, H*W, 3) # camera origin
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    x = (x.to(dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(device).contiguous()
    y = (y.to(dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(device).contiguous()

    # unproject to camera space
    x = (x - input_intr[:, :, 2:3]) / input_intr[:, :, 0:1] #
    y = (y - input_intr[:, :, 3:4]) / input_intr[:, :, 1:2] #
    ray_d = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # (B, V, H*W, 3)
    ray_d = F.normalize(ray_d, p=2, dim=-1)
    ray_d = ray_d @ input_c2ws[:, :, :3, :3].transpose(-1, -2).contiguous() # (B, V, H*W, 3)
    return ray_o, ray_d

def feat2gaussian(
    gaussian_params: dict[Literal['xyz', 'scale', 'rotation', 'feature', 'opacity'], Tensor],
) -> Gaussians:
    means, scales, rotations, sh_feature, opacities = gaussian_params['xyz'], gaussian_params['scale'], gaussian_params['rotation'], gaussian_params['feature'], gaussian_params['opacity']
    covariances = build_covariance(scales, rotations)

    return Gaussians(
        means=means.float(),
        covariances=covariances.float(),
        harmonics=sh_feature.permute(0, 1, 3, 2).contiguous().float(),
        opacities=opacities.squeeze(-1).float(),
        scales=scales,
        rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
    )

@torch.no_grad()
def compute_vhull(masks: Float[Tensor, "*#batch view 1 H W"],
                  extrinsics: Float[Tensor, "*#batch view 4 4"],
                  intrinsics: Float[Tensor, "*#batch view 3 3"],
                  bounds, resolution, min_cell_size=None, use_loop=False, disable_pbar=False, cube_sample=False):
    """
    Compute visual hull from multi-view masks, with an option to use for-loop for memory efficiency.
    Args:
        mask:      [B, V, 1, H, W]   torch.bool
        fxfycxcy:  [B, V, 4]      torch.float32 (fx, fy, cx, cy)
        c2w:       [B, V, 4, 4]   torch.float32 (camera to world)
        bounds: float or list of floats (with shape of [B ,6])
        resolution: int
        use_loop: bool, whether to use for-loop instead of batch processing
    Returns:
        logits: [B, N, N, N]         torch.float32
        coords_world: [B, N, N, N, 3] torch.float32
    """

    B, V, _, H, W = masks.shape
    device = masks.device

    # Generate the world coordinates grid
    if isinstance(bounds, float):
        # [x_min, y_min, z_min, x_max, y_max, z_max]
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
    if isinstance(bounds, list) and isinstance(bounds[0], float):
        bounds = [bounds for _ in range(B)]  # [B, 6]
    bounds = torch.as_tensor(bounds, device=device, dtype=torch.float32)
    if cube_sample:
        sizes = bounds[:, 3:6] - bounds[:, 0:3]  # [B, 3]
        if min_cell_size is None:
            min_cell_size = torch.min(sizes, dim=1)[0].min().item() / resolution
        steps = ((sizes / min_cell_size).floor().long()).clamp(min=1)  # [B, 3]

        # Calculate grid sizes and find minimum to prevent OOM
        grid_sizes = steps.prod(dim=1)  # [B]
        min_grid_len = grid_sizes.min().item()

        min_grid_len = min(8000000, min_grid_len)  # 1M points per batch

        coords_world = []
        for i in range(B):
            # All computations on GPU to avoid CPU-GPU transfers
            x_min, y_min, z_min = bounds[i, :3]
            x_max, y_max, z_max = bounds[i, 3:6]
            x_steps, y_steps, z_steps = steps[i]

            # Create linspace directly on GPU
            x = torch.linspace(x_min, x_max, x_steps, device=device, dtype=torch.float32)
            y = torch.linspace(y_min, y_max, y_steps, device=device, dtype=torch.float32)
            z = torch.linspace(z_min, z_max, z_steps, device=device, dtype=torch.float32)

            # Create meshgrid on GPU
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [N, 3]

            if grid.shape[0] > min_grid_len:
                # Use GPU-based random indices generation
                indices = torch.randperm(grid.shape[0], device=device)[:min_grid_len]
                grid = grid[indices]
            elif grid.shape[0] < min_grid_len:
                # Pad with repeated samples if needed (edge case)
                repeat_factor = (min_grid_len + grid.shape[0] - 1) // grid.shape[0]
                grid = grid.repeat(repeat_factor, 1)[:min_grid_len]

            coords_world.append(grid)

        coords_world = torch.stack(coords_world, dim=0)  # [B, min_grid_len, 3]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = bounds.split(1, dim=1)
        x_lin = torch.linspace(0, 1, resolution, device=device)  # [N]
        y_lin = torch.linspace(0, 1, resolution, device=device)  # [N]
        z_lin = torch.linspace(0, 1, resolution, device=device)  # [N]
        coords_world = []
        for b in range(B):
            x = x_lin * (x_max[b] - x_min[b]) + x_min[b]  # [N]
            y = y_lin * (y_max[b] - y_min[b]) + y_min[b]  # [N]
            z = z_lin * (z_max[b] - z_min[b]) + z_min[b]  # [N]
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')  # [N, N, N]
            grid = torch.stack([xx, yy, zz], dim=-1)  # [N, N, N, 3]
            coords_world.append(grid)
        coords_world = torch.stack(coords_world, dim=0)  # [B, N, N, N, 3]

    coords_world = coords_world.reshape(B, -1, 3)  # [B, N^3, 3]

    if use_loop:
        # Initialize accumulator for logits
        logits_accum = torch.zeros((B, coords_world.shape[1]), device=device)

        # Chunk-based processing to balance memory and speed
        chunk_size = 64
        total_chunks = (B * V + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, B * V)

            # Process current chunk
            chunk_logits = torch.zeros(
                (end_idx - start_idx, coords_world.shape[1]), device=device)

            for local_idx in range(end_idx - start_idx):
                global_idx = start_idx + local_idx
                b = global_idx // V
                v = global_idx % V

                if b >= B:  # Safety check
                    continue

                coords = coords_world[b]
                coords_homo = torch.cat(
                    [coords, torch.ones(coords.shape[0], 1, device=device)], dim=-1)

                # Compute world to camera transformation for current view
                w2c = torch.inverse(extrinsics[b, v])  # [4, 4]

                # Transform coordinates to camera space
                pts_cam = coords_homo @ w2c.T  # [N^3, 4]

                # Get camera intrinsics
                fx, fy, cx, cy = intrinsics[b, v, 0, 0], intrinsics[b, v, 1, 1], intrinsics[b, v, 0, 2], intrinsics[b, v, 1, 2]

                # Perspective projection
                z = pts_cam[:, 2] + 1e-8  # [N^3]
                u = (pts_cam[:, 0] / z) * fx + cx  # [N^3]
                v_proj = (pts_cam[:, 1] / z) * fy + cy  # [N^3]

                # Validity checks
                valid_z = z > 0  # [N^3]
                valid_uv = (u >= 0) & (
                    u <= W-1) & (v_proj >= 0) & (v_proj <= H-1)
                valid = valid_z & valid_uv  # [N^3]

                # Normalize to [-1, 1] for grid_sample
                u_norm = 2 * (u / (W - 1)) - 1  # [N^3]
                v_norm = 2 * (v_proj / (H - 1)) - 1  # [N^3]
                grid = torch.stack([u_norm, v_norm], dim=-
                                   1).view(1, -1, 1, 2)  # [1, N^3, 1, 2]

                # Sample mask for current view
                mask = masks[b, v].float().view(
                    1, 1, H, W)  # [1, 1, H, W]
                sampled = torch.nn.functional.grid_sample(
                    mask,
                    grid,
                    align_corners=True,
                    mode='nearest'
                ).squeeze()  # [N^3]

                # Compute visibility: inside mask and valid
                visibility = (sampled > 0.5) & valid
                visibility = visibility.view(-1)  # [N^3]

                # Store visibility for current chunk
                chunk_logits[local_idx] = visibility.float()

            # Accumulate chunk results back to main accumulator
            for local_idx in range(end_idx - start_idx):
                global_idx = start_idx + local_idx
                b = global_idx // V
                v = global_idx % V

                if b >= B:  # Safety check
                    continue

                logits_accum[b] += chunk_logits[local_idx]

        # Average over views
        logits = logits_accum / V
    else:
        # Original batch processing implementation
        coords_homo = torch.cat([coords_world, torch.ones_like(
            coords_world[..., :1])], -1)  # [B, N^3, 4]
        coords_homo = coords_homo.unsqueeze(1).expand(
            B, V, -1, 4).reshape(B*V, -1, 4)  # [B*V, N^3, 4]

        w2c = torch.inverse(extrinsics).view(B*V, 4, 4)
        pts_cam = torch.bmm(coords_homo, w2c.transpose(1, 2))

        # fx = fxfycxcys[..., 0].view(B*V)
        # fy = fxfycxcys[..., 1].view(B*V)
        # cx = fxfycxcys[..., 2].view(B*V)
        # cy = fxfycxcys[..., 3].view(B*V)
        fx, fy, cx, cy = intrinsics[..., 0, 0].view(B*V), intrinsics[..., 1, 1].view(B*V), intrinsics[..., 0, 2].view(B*V), intrinsics[..., 1, 2].view(B*V)

        z = pts_cam[..., 2] + 1e-8
        u = (pts_cam[..., 0] / z) * fx[:, None] + cx[:, None]
        v = (pts_cam[..., 1] / z) * fy[:, None] + cy[:, None]

        u_norm = 2 * (u / (W - 1)) - 1
        v_norm = 2 * (v / (H - 1)) - 1
        grid = torch.stack([u_norm, v_norm], -1).unsqueeze(2)

        mask_flattened = masks.view(B*V, 1, H, W)
        sampled = torch.nn.functional.grid_sample(
            mask_flattened.float(),
            grid,
            align_corners=True,
            mode='nearest'
        ).squeeze()

        valid_z = z > 0
        valid_uv = (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1)
        valid = valid_z & valid_uv

        visibility = (sampled > 0.5) & valid
        visibility = visibility.view(B, V, -1)
        logits = visibility.sum(dim=1) / V
        logits = logits.reshape(B, -1).to(torch.float32)
    return logits, coords_world

def get_vhull_points(masks: Float[Tensor, "*#batch view 1 H W"],
                     extrinsics: Float[Tensor, "*#batch view 4 4"],
                     intrinsics: Float[Tensor, "*#batch view 3 3"],
                     bounds, resolution, min_cell_size= None, cube_sample=False, threshold=0.8, use_loop=True):
        B, V = masks.shape[:2]
        vhull_batch_1, coords_world_1 = compute_vhull(masks,
                                                      extrinsics,
                                                      intrinsics,
                                                      bounds=bounds,
                                                      resolution=64,
                                                      use_loop=False,
                                                      disable_pbar=True,
                                                      cube_sample=False)
        vhull_mask_1 = (vhull_batch_1 >= 0.5).view(B, -1)
        vhull_pts_1 = coords_world_1.view(B, -1, 3)

        vhull_mask = []
        vhull_batch = []
        coords_world = []

        # Compute bounds for each batch using vectorized operations
        bound_list = []
        for i in range(B):
            batch_pts = vhull_pts_1[i][vhull_mask_1[i]]
            min_bounds = batch_pts.min(dim=0)[0]  # [3]
            max_bounds = batch_pts.max(dim=0)[0]  # [3]
            bound_range = (max_bounds - min_bounds) * 1.2
            mid_point = (min_bounds + max_bounds) / 2
            bound_list.append(torch.cat([mid_point - bound_range / 2, mid_point + bound_range / 2]).tolist())
        vhull_batch, coords_world = compute_vhull(masks,
                                                  extrinsics,
                                                  intrinsics,
                                                  bounds=bound_list,
                                                  resolution=resolution,
                                                  min_cell_size = min_cell_size,
                                                  use_loop=use_loop,
                                                  disable_pbar=True,
                                                  cube_sample=cube_sample)
        vhull_mask = (vhull_batch >= threshold)

        # use vhull as xyz
        vhull_mask = vhull_mask.view(B, -1)
        vhull_batch = vhull_batch.view(B, -1)
        coords_world = coords_world.view(B, -1, 3)
        return vhull_mask, vhull_batch, coords_world

def get_condition_patches(posed_input_images_tokens, mask_patches, token_num_per_frame, downsample_ratio=1.0):
    # Sum mask patches along the last dimension
    sum_mask_patches = mask_patches.sum(dim=-1)  # [b, (v hh ww)]

    # Create a boolean mask for patches with sum > 0
    valid_patches = sum_mask_patches > 0  # [b, (v hh ww)]

    # Get the maximum number of valid patches across all batches
    max_seqlen = valid_patches.sum(dim=-1).max()  # [b].max()

    # Sort patches by sum value to get top max_seqlen patches
    sorted_indices = sum_mask_patches.argsort(dim=-1, descending=True)  # [b, (v hh ww)]

    # Select top max_seqlen indices
    top_indices = sorted_indices[:, :max_seqlen]  # [b, max_seqlen]

    # Randomly sample max_seqlen patches
    sample_seqlen = int(max_seqlen * downsample_ratio)
    if sample_seqlen < max_seqlen:
        top_indices = top_indices[:, torch.randperm(max_seqlen)[:sample_seqlen]]

    # Use index_select which is more memory efficient than direct indexing
    batch_indices = torch.arange(posed_input_images_tokens.size(0), device=posed_input_images_tokens.device)[:, None]
    batch_indices = batch_indices.expand(-1, sample_seqlen).reshape(-1)

    # flat_indices = top_indices.reshape(-1)
    # posed_and_masked_input_images_tokens = posed_input_images_tokens[batch_indices, flat_indices].reshape(
    #     posed_input_images_tokens.size(0), sample_seqlen, posed_input_images_tokens.size(2)
    # )

    image_indices = top_indices // token_num_per_frame  # [b, max_seqlen]

    # Sort by image indices to group same image tokens together
    batch_size = image_indices.size(0)
    sorted_order = torch.argsort(image_indices, dim=1) # [b, max_seqlen]

    # Apply sorting to both image_indices and top_indices
    batch_idx = torch.arange(batch_size, device=image_indices.device).unsqueeze(1)
    sorted_image_indices = image_indices[batch_idx, sorted_order] # [b, max_seqlen]
    sorted_top_indices = top_indices[batch_idx, sorted_order] # [b, max_seqlen]

    # Extract posed_input_images_tokens according to sorted_top_indices
    batch_indices_expanded = torch.arange(batch_size, device=posed_input_images_tokens.device)[:, None]
    batch_indices_expanded = batch_indices_expanded.expand(-1, sample_seqlen).reshape(-1)
    flat_sorted_indices = sorted_top_indices.reshape(-1)

    posed_and_masked_input_images_tokens = posed_input_images_tokens[batch_indices_expanded, flat_sorted_indices].reshape(
        batch_size, sample_seqlen, posed_input_images_tokens.size(2)
    )

    index_len_list = []
    max_unique_images = 0

    for b in range(batch_size):
        unique_indices, counts = torch.unique(sorted_image_indices[b], return_counts=True)
        index_len_list.append(counts)
        max_unique_images = max(max_unique_images, len(unique_indices))

    # Pad to ensure same number of unique image indices across batches
    index_len = torch.zeros(batch_size, max_unique_images, device=posed_input_images_tokens.device, dtype=torch.long)
    for b in range(batch_size):
        counts = index_len_list[b]
        index_len[b, :len(counts)] = counts

    return posed_and_masked_input_images_tokens, index_len

def voxel_ray_query(points: torch.Tensor,
                    ray_o: torch.Tensor,
                    ray_d: torch.Tensor,
                    eps: float = 0.002,
                    voxel_size: float = 0.005,
                    max_steps: int = 256,
                    max_grid_cells: int = 50_000_000,
                    prune_reentry: bool = False):
    """
    Voxel-grid ray query with a batched DDA traversal.
    - Builds the voxel grid inside the point-cloud bounding box without origin alignment
      or normalization.
    - Steps each ray with 3D DDA, using one Python loop over steps while processing all
      active rays in parallel.
    - Uses dense cell_count and cell_offset tables to gather candidate points for all
      active rays and compute the closest valid hit.

    Args:
        points: [B, N, 3] point coordinates in any world coordinate frame.
        ray_o: [B, M, 3] ray origins.
        ray_d: [B, M, 3] ray directions, normalized internally.
        eps: hit radius; point-to-ray distances below eps count as hits.
        voxel_size: voxel edge length.
        max_steps: maximum DDA steps.
        max_grid_cells: maximum voxel count before raising to avoid excessive memory.

    Returns:
        hits: [B, M, 3] closest hit coordinates, or NaN for missing hits.
    """
    assert points.ndim == 3 and points.size(-1) == 3
    assert ray_o.shape[:-1] == ray_d.shape[:-1]
    assert points.shape[0] == ray_o.shape[0]

    device = points.device
    dtype = points.dtype

    B, N, _ = points.shape
    _, M, _ = ray_o.shape

    hits = torch.full((B, M, 3), float('nan'), device=device, dtype=dtype)
    INF = torch.tensor(float('inf'), device=device, dtype=dtype)

    for b in range(B):
        pts = points[b]         # [N,3]
        ro = ray_o[b]           # [M,3]
        rd = ray_d[b]
        rd = rd / (rd.norm(dim=-1, keepdim=True) + 1e-12)

        # Build a voxel grid from the point-cloud bounds.
        p_min = pts.min(dim=0).values - eps
        p_max = pts.max(dim=0).values + eps

        dims_f = torch.clamp(torch.ceil((p_max - p_min) / voxel_size), min=1)
        nx, ny, nz = int(dims_f[0].item()), int(dims_f[1].item()), int(dims_f[2].item())
        num_cells = nx * ny * nz
        if num_cells > max_grid_cells:
            raise RuntimeError(
                f"Grid too large: {nx}x{ny}x{nz}={num_cells} cells. "
                f"Increase voxel_size or set a larger max_grid_cells.")

        # Linear indexing strides, with x as the fastest axis.
        sx, sy, sz = 1, nx, nx * ny

        g = torch.floor((pts - p_min) / voxel_size).to(torch.long)  # [N,3]
        g = torch.stack([
            g[:, 0].clamp(0, nx - 1),
            g[:, 1].clamp(0, ny - 1),
            g[:, 2].clamp(0, nz - 1)
        ], dim=-1)
        cell_id = g[:, 0] * sx + g[:, 1] * sy + g[:, 2] * sz  # [N]

        # Sort by cell id and build a CSR-like cell lookup.
        order = torch.argsort(cell_id)
        cell_sorted = cell_id[order]
        pid_sorted = order

        uniq, counts = torch.unique_consecutive(cell_sorted, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts
        ends = starts + counts

        # Dense lookup: each cell stores its sorted-point offset and count.
        cell_count = torch.zeros(num_cells, device=device, dtype=torch.int32)
        cell_offset = torch.full((num_cells,), -1, device=device, dtype=torch.int32)
        cell_count[uniq] = counts.to(torch.int32)
        cell_offset[uniq] = starts.to(torch.int32)

        # Initialize DDA state from the ray/AABB intersections.
        invd = 1.0 / (rd + 1e-12)
        t0s = (p_min - ro) * invd  # [M,3]
        t1s = (p_max - ro) * invd
        tmin = torch.minimum(t0s, t1s).amax(dim=-1)
        tmax = torch.maximum(t0s, t1s).amin(dim=-1)
        valid = tmax >= torch.maximum(tmin, torch.zeros_like(tmin))

        t_enter = torch.clamp(tmin, min=0.0)
        start_pos = ro + t_enter.unsqueeze(-1) * rd  # [M,3]
        gv = torch.floor((start_pos - p_min) / voxel_size).to(torch.long)
        gv = torch.stack([
            gv[:, 0].clamp(0, nx - 1),
            gv[:, 1].clamp(0, ny - 1),
            gv[:, 2].clamp(0, nz - 1)
        ], dim=-1)

        step = torch.sign(rd).to(torch.long)  # [-1,0,1]
        tDelta = torch.empty_like(rd)
        abs_rd = rd.abs() + 1e-12
        tDelta[:, 0] = voxel_size / abs_rd[:, 0]
        tDelta[:, 1] = voxel_size / abs_rd[:, 1]
        tDelta[:, 2] = voxel_size / abs_rd[:, 2]

        gv_f = gv.to(dtype)
        h = torch.tensor(voxel_size, device=device, dtype=dtype)
        # For positive directions, use the upper voxel face; otherwise use the lower.
        nb_x = torch.where(rd[:, 0] > 0, p_min[0] + (gv_f[:, 0] + 1.0) * h, p_min[0] + gv_f[:, 0] * h)
        nb_y = torch.where(rd[:, 1] > 0, p_min[1] + (gv_f[:, 1] + 1.0) * h, p_min[1] + gv_f[:, 1] * h)
        nb_z = torch.where(rd[:, 2] > 0, p_min[2] + (gv_f[:, 2] + 1.0) * h, p_min[2] + gv_f[:, 2] * h)
        tMax = torch.stack([
            (nb_x - ro[:, 0]) / (rd[:, 0] + 1e-12),
            (nb_y - ro[:, 1]) / (rd[:, 1] + 1e-12),
            (nb_z - ro[:, 2]) / (rd[:, 2] + 1e-12)
        ], dim=-1)
        tMax = torch.where(rd == 0, INF, tMax)

        best_t = torch.full((M,), float('inf'), device=device, dtype=dtype)
        best_p = torch.full((M, 3), float('nan'), device=device, dtype=dtype)
        best_cell_id = torch.full((M,), -1, device=device, dtype=torch.long)

        # Main DDA loop. Each step processes all active rays and candidates in parallel.
        for _ in range(max_steps):
            alive = valid & (t_enter <= tmax) & (t_enter <= best_t)
            if not torch.any(alive):
                break

            gv_alive = gv[alive]
            cell_alive = gv_alive[:, 0] * sx + gv_alive[:, 1] * sy + gv_alive[:, 2] * sz  # [A]
            cnt = cell_count[cell_alive].to(torch.long)   # [A]
            off = cell_offset[cell_alive].to(torch.long)  # [A]
            has_pts = cnt > 0
            if torch.any(has_pts):
                alive_idx = torch.nonzero(alive, as_tuple=False).squeeze(1)
                alive_idx = alive_idx[has_pts]
                cnt_sel = cnt[has_pts]
                off_sel = off[has_pts]

                A = cnt_sel.numel()
                maxc = int(cnt_sel.max().item())

                rel = torch.arange(maxc, device=device).unsqueeze(0).expand(A, maxc)
                mask = rel < cnt_sel.unsqueeze(1)  # [A,maxc]
                pos2d = off_sel.unsqueeze(1) + rel  # [A,maxc]

                flat_pos = pos2d[mask]
                pts_idx = pid_sorted[flat_pos]  # [K]

                pts_full = torch.zeros((A, maxc, 3), device=device, dtype=dtype)
                pts_full[mask] = pts[pts_idx]

                ro_sel = ro[alive_idx]               # [A,3]
                rd_sel = rd[alive_idx]               # [A,3]

                v = pts_full - ro_sel[:, None, :]     # [A,maxc,3]
                t_val = (v * rd_sel[:, None, :]).sum(dim=-1)              # [A,maxc]
                dist = (v - t_val[..., None] * rd_sel[:, None, :]).norm(dim=-1)
                hit_mask = mask & (t_val >= 0) & (dist < eps)

                t_step = torch.where(hit_mask, t_val, INF)
                t_step_min, t_step_arg = torch.min(t_step, dim=1)  # [A]

                better = t_step_min < best_t[alive_idx]
                if torch.any(better):
                    idx_upd = alive_idx[better]
                    best_t[idx_upd] = t_step_min[better]

                    gather_row = torch.nonzero(better, as_tuple=False).squeeze(1)
                    col = t_step_arg[better]
                    best_pts_sel = pts_full[gather_row, col]  # [num_better,3]
                    best_p[idx_upd] = best_pts_sel
                    best_cell_id[idx_upd] = cell_alive[has_pts][better]

            alive_now = valid & (t_enter <= tmax) & (t_enter <= best_t)
            if not torch.any(alive_now):
                break
            idx = torch.nonzero(alive_now, as_tuple=False).squeeze(1)

            tMax_alive = tMax[idx]          # [A,3]
            step_alive = step[idx]          # [A,3]
            gv_alive2 = gv[idx]             # [A,3]
            tDelta_alive = tDelta[idx]      # [A,3]

            axis = torch.argmin(tMax_alive, dim=-1)  # [A]
            t_next = tMax_alive.gather(1, axis.unsqueeze(1)).squeeze(1)

            tMax_alive.scatter_add_(1, axis.unsqueeze(1), tDelta_alive.gather(1, axis.unsqueeze(1)))
            gv_alive2[torch.arange(gv_alive2.size(0), device=device), axis] += step_alive.gather(1, axis.unsqueeze(1)).squeeze(1)

            tMax[idx] = tMax_alive
            gv[idx] = gv_alive2
            t_enter[idx] = t_next

        # Optionally suppress repeated hits from the same cell.
        if prune_reentry:
            valid_mask = ~torch.isnan(best_p[:, 0])
            if torch.any(valid_mask):
                valid_cell_ids = best_cell_id[valid_mask]
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

                unique_cells, first_indices = torch.unique(valid_cell_ids, return_inverse=True)

                keep_mask = torch.zeros_like(valid_mask)
                for i, cell_id in enumerate(unique_cells):
                    cell_positions = valid_indices[valid_cell_ids == cell_id]
                    if len(cell_positions) > 0:
                        keep_mask[cell_positions[0]] = True

                best_p[valid_mask & ~keep_mask] = float('nan')

        hits[b] = best_p

    return hits

def patchify_3d(points, grid_hash, patch_size):
    """
    Args:
        points: [B, N, c] tensor, where c >= 3 (first 3 are xyz, rest are features)
        grid_hash: [B, N] tensor, int
        patch_size: int
    Returns:
        out: [B, max_patches, patch_size, c]
    """
    B, N, c = points.shape
    device = points.device

    # 1. Flatten batch
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(-1)  # [B*N]
    points_flat = points.reshape(-1, c)  # [B*N, c]
    grid_hash_flat = grid_hash.reshape(-1)  # [B*N]

    # 2. Combine batch and grid_hash to avoid hash collision between batches
    max_hash = grid_hash_flat.max().item() + 1
    global_hash = batch_idx * max_hash + grid_hash_flat  # [B*N]

    # 3. Sort by global_hash
    sorted_hash, sort_idx = global_hash.sort()
    sorted_points = points_flat[sort_idx]

    # 4. Find patch start indices
    unique_hashes, counts = torch.unique_consecutive(sorted_hash, return_counts=True)
    patch_starts = torch.cat([torch.tensor([0], device=device), counts.cumsum(0)[:-1]])

    # 5. Batch sample/fill for each patch (fully vectorized GPU version)
    total_patches = unique_hashes.size(0)
    patches = torch.zeros((total_patches, patch_size, c), device=device, dtype=points.dtype)

    # Create patch-wise indices for sampling
    patch_idx = torch.arange(total_patches, device=device).unsqueeze(1)  # [total_patches, 1]
    sample_idx = torch.arange(patch_size, device=device).unsqueeze(0)    # [1, patch_size]

    # Expand to get all combinations
    patch_idx_expanded = patch_idx.expand(total_patches, patch_size)      # [total_patches, patch_size]
    sample_idx_expanded = sample_idx.expand(total_patches, patch_size)    # [total_patches, patch_size]

    # Calculate the actual indices in sorted_points for each patch and sample
    patch_starts_expanded = patch_starts.unsqueeze(1).expand(-1, patch_size)  # [total_patches, patch_size]
    counts_expanded = counts.unsqueeze(1).expand(-1, patch_size)              # [total_patches, patch_size]

    # For patches with count >= patch_size, we need random sampling
    # For patches with count < patch_size, we use modulo to repeat
    needs_sampling = counts >= patch_size

    # Generate random indices for patches that need sampling
    max_count = counts.max().item()
    if max_count > 0:
        # Create random permutations for all patches (we'll only use what we need)
        random_base = torch.rand(total_patches, max(max_count, patch_size), device=device)
        random_indices = random_base.argsort(dim=1)  # [total_patches, max_count]

        # For sampling patches: use random indices
        sampling_indices = random_indices[:, :patch_size]  # [total_patches, patch_size]

        # For repeating patches: use modulo indices
        repeating_indices = sample_idx_expanded % counts_expanded.clamp(min=1)

        # Combine based on needs_sampling mask
        needs_sampling_expanded = needs_sampling.unsqueeze(1).expand(-1, patch_size)
        final_indices = torch.where(needs_sampling_expanded, sampling_indices, repeating_indices)

        # Add patch starts to get absolute indices in sorted_points
        absolute_indices = patch_starts_expanded + final_indices

        # Clamp to ensure we don't go out of bounds for any patch
        max_indices = patch_starts_expanded + counts_expanded - 1
        absolute_indices = torch.min(absolute_indices, max_indices)

        # Gather the points
        patches = sorted_points[absolute_indices]  # [total_patches, patch_size, c]

    # 6. Restore batch structure
    batch_ids = (unique_hashes // max_hash).long()  # [total_patches]
    num_patches_per_batch = torch.bincount(batch_ids, minlength=B)
    max_patches = num_patches_per_batch.max().item()
    out = torch.zeros((B, max_patches, patch_size, c), device=device, dtype=points.dtype)
    out[..., 2] = 1e3 # set z to avoid numerical issue
    mask = torch.zeros((B, max_patches), device=device, dtype=torch.bool)
    patch_ptr = 0
    for b in range(B):
        n_patch = num_patches_per_batch[b].item()
        if n_patch > 0:
            out[b, :n_patch] = patches[patch_ptr:patch_ptr+n_patch]
            mask[b, :n_patch] = True
            patch_ptr += n_patch
    return out, mask  # [B, max_patches, patch_size, c]

class EncoderPointsplat(Encoder[EncoderPointsplatCfg]):
    def __init__(self, cfg: EncoderPointsplatCfg) -> None:
        super().__init__(cfg)

        self.patch_size = cfg.image_tokenizer.patch_size


        self.input_ln=cfg.transformer.input_ln
        self.output_ln=cfg.transformer.output_ln


        self.image_tokenizer = self.build_tokenizer(
            input_dim=cfg.image_tokenizer.in_channels * self.patch_size ** 2,
            output_dim=cfg.transformer.d,
            use_ln = cfg.image_tokenizer.use_ln,
        )

        # build transformer
        if self.input_ln:
            self.transformer_input_layernorm = nn.LayerNorm(cfg.transformer.d, bias=False, eps=1e-6)
        if self.output_ln:
            self.transformer_output_layernorm = nn.LayerNorm(cfg.transformer.d, bias=False, eps=1e-6)
        self.transformer_blocks = self.build_transformer(cfg.transformer)

        self.point_patch_size = cfg.point_patch_size
        self.gaussian_hidden_dim = cfg.gaussians_decoder.hidden_dim
        self.out_linear = nn.Linear(cfg.transformer.d, self.point_patch_size * self.gaussian_hidden_dim)
        self.input_point_tokenizer = PatchEmbed3D(feature_dim=6, dim = cfg.transformer.d, point_per_patch=self.point_patch_size)
        self.output_tokenizer = GSLayer(in_channels=self.gaussian_hidden_dim,
                                    use_rgb=cfg.gaussians_decoder.use_rgb,
                                    sh_degree=cfg.gaussians_decoder.sh_degree,
                                    clip_scaling=cfg.gaussians_decoder.clip_scaling,
                                    init_scaling=cfg.gaussians_decoder.scale_bias,
                                    init_density=cfg.gaussians_decoder.init_density,
                                    xyz_offset=True,
                                    restrict_offset=True,
                                    xyz_offset_max_step=cfg.gaussians_decoder.xyz_offset_max_step,
                                    fix_opacity=False,
                                    fix_rotation=cfg.gaussians_decoder.fix_rotation,
                                    use_fine_feat=False,
                                    mlp_net_config=edict(
                                        activation=cfg.gaussians_decoder.activation,
                                        n_hidden_layers=cfg.gaussians_decoder.n_hidden_layers,
                                        n_neurons=cfg.gaussians_decoder.n_neurons
                                    )
                            )

        self.vhull_res = cfg.vhull_res
        self.bounds = [i for i in cfg.bounds]

        self.register_buffer('dna_bg', torch.tensor([0.31, 0.39, 0.29]).reshape(1, 1, 3, 1, 1))

    def build_tokenizer(self, input_dim, output_dim, use_sinusoidal=False, use_ln=False):
        """Helper function to create a tokenizer with given config"""
        tokenizer = Tokenizer(input_dim, output_dim, use_sinusoidal = use_sinusoidal, use_ln = use_ln)
        tokenizer.apply(init_weights)
        return tokenizer

    def build_transformer(self, config: TransformerCfg):
        model = [
            PointImageMMJointTransformerBlock(
                dim=config.d,
                num_heads=config.d_head,
                point_only=i == config.n_layer - 1,
            )
            for i in range(config.n_layer)
        ]
        model = nn.ModuleList(model)
        model.apply(init_weights)
        return model

    @torch.no_grad()
    def get_posed_input(self,
                        images: Float[Tensor, "*#batch view 3 H W"] | None = None,
                        ray_o: Float[Tensor, "*#batch view 3 H W"] | None=None,
                        ray_d: Float[Tensor, "*#batch view 3 H W"] | None=None,
                        method: Literal["custom_plucker", "aug_plucker", "default_plucker"] = "default_plucker"):
        '''
            Args:
                images: [b, v, 3, h, w]
                ray_o: [b, v, 3, h, w]
                ray_d: [b, v, 3, h, w]
                method: Method for creating pose conditioning, one of "custom_plucker", "aug_plucker", "default_plucker"
            Returns:
                posed_images: [b, v, 3+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        def normalize_rgb(images):
            return images * 2.0 - 1.0


        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)

        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)

        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([normalize_rgb(images), pose_cond], dim=2)

    @torch.no_grad()
    def sample_points(self,
                    xyz: Float[Tensor, "*#batch N 3"],
                    mask: Float[Tensor, "*#batch v 1 H W"],
                    ray_o: Float[Tensor, "*#batch v 3 H W"],
                    ray_d: Float[Tensor, "*#batch v 3 H W"],
                    posed_input_images: Float[Tensor, "*#batch v 9 H W"],
                    num_target: int = 240000):
        B = xyz.shape[0]
        device = xyz.device

        # Rearrange mask to [B, N, 1], squeeze to [B, N]
        ray_mask = (rearrange(mask, "b v c h w -> b (v h w) c") > 0).reshape(B, -1)  # [B, N]
        ray_o_all = rearrange(ray_o, "b v c h w -> b (v h w) c")  # [B, N, 3]
        ray_d_all = rearrange(ray_d, "b v c h w -> b (v h w) c")  # [B, N, 3]
        plucker_rays = rearrange(posed_input_images[:, :, -6:, ...], "b v c h w -> b (v h w) c")

        min_cell_size = self.cfg.min_cell_size
        voxel_size = min_cell_size * 2.5 if min_cell_size is not None else 0.005
        min_hit = voxel_ray_query(xyz,
                                ray_o_all,
                                ray_d_all,
                                voxel_size=voxel_size)
        nan_mask = torch.isnan(min_hit).any(dim=-1)  # [B, N]
        valid_mask = ~nan_mask & ray_mask
        min_hit = torch.cat([min_hit, plucker_rays], dim=-1)

        valid_num_list = []
        valid_points_list = []
        for b in range(B):
            valid_points = min_hit[b][valid_mask[b]]
            valid_num_list.append(valid_points.shape[0])
            valid_points_list.append(valid_points)
        num_select = min(num_target, max(valid_num_list))
        filtered_min_hit = []
        for b in range(B):
            valid_points = valid_points_list[b]
            if valid_points.shape[0] >= num_select:
                target_indices = torch.randperm(valid_points.shape[0], device=device)[:num_select]
                valid_points = valid_points[target_indices]
            else:
                repeat_times = (num_select + valid_points.shape[0] - 1) // valid_points.shape[0]
                valid_points = valid_points.repeat(repeat_times, 1)[:num_select]
            filtered_min_hit.append(valid_points)
        filtered_min_hit = torch.stack(filtered_min_hit, dim=0)
        points_xyz = filtered_min_hit[..., :3]
        points = filtered_min_hit

        return points_xyz, points

    def pass_layers(self, tokens, gradient_checkpoint=False, checkpoint_every=1, **kwargs):
        # cond_tokens, attention_mask=None,
        num_layers = len(self.transformer_blocks)

        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for layer in self.transformer_blocks:
                tokens, kwargs = layer(tokens, **kwargs)
        else:
            # Gradient checkpointing enabled - process layers in groups
            def _process_layer_group(tokens, start_idx, end_idx, **kwargs_inner):
                """Helper to process a group of consecutive layers."""
                for idx in range(start_idx, end_idx):
                    tokens, kwargs_inner = self.transformer_blocks[idx](tokens, **kwargs_inner)
                return tokens, kwargs_inner

            # Process layer groups with gradient checkpointing
            for start_idx in range(0, num_layers, checkpoint_every):
                end_idx = min(start_idx + checkpoint_every, num_layers)
                tokens, kwargs = torch.utils.checkpoint.checkpoint(
                    _process_layer_group,
                    tokens,
                    start_idx,
                    end_idx,
                    **kwargs,
                    use_reentrant=False
                )

        return tokens

    def forward(
        self,
        context: BatchedViews,
        global_step: int,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> EncoderOutput:
        assert self.training == False, "Only supports inference currently."

        context_mask = context['mask']
        B, v_input, _, H, W = context_mask.shape

        masked_context_images = context['image'] * context_mask + (1.0 - context_mask) * self.dna_bg.expand_as(context['image'])

        context_ray_o, context_ray_d = get_camera_ray_dir(context['image'], context['extrinsics'], context['intrinsics'])
        context_ray_o = rearrange(context_ray_o, "b v (h w) c -> b v c h w", h=H, w=W)
        context_ray_d = rearrange(context_ray_d, "b v (h w) c -> b v c h w", h=H, w=W)

        posed_input_images = self.get_posed_input(
            images=masked_context_images, ray_o=context_ray_o, ray_d=context_ray_d
        )

        posed_input_images_tokens = rearrange(
            posed_input_images,
            "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

        mask_patches = rearrange(
            context_mask,
            "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
            c=1,
            ph=self.patch_size,
            pw=self.patch_size,
        )

        cond, cond_length_list = get_condition_patches(posed_input_images_tokens, mask_patches, (H*W) // (self.patch_size**2))

        cond = self.image_tokenizer(cond)

        vhull_mask, _, coords_world = get_vhull_points(context_mask, context['extrinsics'], context['intrinsics'],
                                                       self.bounds, self.vhull_res,
                                                        min_cell_size=self.cfg.min_cell_size,
                                                        cube_sample=self.cfg.cube_sample,
                                                        threshold=self.cfg.threshold,
                                                        use_loop=False)

        # structurize the vhull points
        valid_indices_list = [torch.where(vhull_mask[i])[0] for i in range(B)]
        l_min = min([len(indices) for indices in valid_indices_list]+[400000,])
        padded_indices = torch.full((B, l_min), -1, dtype=torch.long, device=coords_world.device)
        for i, indices in enumerate(valid_indices_list):
            if len(indices) > l_min:
                perm = torch.randperm(len(indices), device=coords_world.device)[:l_min]
                padded_indices[i] = indices[perm]
            else:
                padded_indices[i, :len(indices)] = indices
        vhull_xyz = torch.gather(coords_world, 1, padded_indices.unsqueeze(-1).expand(-1, -1, 3))

        query_xyz, query = self.sample_points(vhull_xyz,
                                        mask=context_mask,
                                        ray_o=context_ray_o,
                                        ray_d=context_ray_d,
                                        posed_input_images=posed_input_images,
                                        num_target=self.cfg.gaussians_decoder.num_target)

        min_bound = query_xyz.min(dim=1)[0]  # [B, 3]
        max_bound = query_xyz.max(dim=1)[0]  # [B, 3]
        bound_size = (max_bound - min_bound)  # [B, 3]
        min_cell_size = self.cfg.min_cell_size
        cell_size = min_cell_size * 2.5 if min_cell_size is not None else 0.005
        grid_res = (bound_size / cell_size).ceil().clamp(min=1).max()
        cell_size = torch.ones_like(bound_size) * cell_size

        # query_xyz: [B, N, 3], min_bound: [B, 3], cell_size: [B, 3]
        grid_coords = ((query_xyz - min_bound.unsqueeze(1)) / cell_size.unsqueeze(1)).long()  # [B, N, 3]
        grid_hash = grid_coords[..., 0] + grid_coords[..., 1] * grid_res + grid_coords[..., 2] * (grid_res * grid_res)  # [B, N]
        query_patchfied, query_mask = patchify_3d(query, grid_hash, self.point_patch_size)  # [B, max_patches, patch_size, C]


        query_xyz = query_patchfied[..., :3] # [B, seq_len, patch_size, 3]
        query = query_patchfied[..., :9] # [B, max_patches, patch_size, C]

        encoded_query = self.input_point_tokenizer(query)
        if self.input_ln:
            encoded_query = self.transformer_input_layernorm(encoded_query)

        latent = self.pass_layers(encoded_query,
                                  condition=cond,
                                  condition_length_list=cond_length_list,
                                  gradient_checkpoint=True,
                                  checkpoint_every=1)
        if self.output_ln:
            latent = self.transformer_output_layernorm(latent)

        self.output_tokenizer.hyper_step(global_step)

        latent = self.out_linear(latent)
        latent = rearrange(latent, 'b n (p c) -> b n p c', p=self.point_patch_size) # [B, N, patch_size, gs_dim]
        gs_param = self.output_tokenizer(latent, query_xyz)

        # convert to float32
        for k, v in gs_param.items():
            gs_param[k] = v.to(torch.float32)
        gaussians = gs_param

        # GS Pruning
        B, num_gaussians = gaussians['xyz'].shape[:2]

        # GS Pruning Step 1: Prune by opacity ratio
        prune_ratio = self.cfg.gaussians_decoder.prune_ratio
        if prune_ratio > 0:
            num_keep = int(num_gaussians * (1 - prune_ratio))
            keep_idx = gaussians["opacity"].argsort(dim=1, descending=True)[:, :num_keep]
            for k, v in gaussians.items():
                v_flat = v.reshape(B, num_gaussians, -1)
                gaussians[k] = v_flat.gather(1, keep_idx.expand(-1, -1, v_flat.shape[-1])).reshape(B, -1, *v.shape[2:])
            num_gaussians = num_keep


        # format shim
        gaussians_color = gaussians.pop('color')
        if self.cfg.gaussians_decoder.use_rgb:
            assert gaussians_color.shape == (B, num_gaussians, 3)
            gaussians['feature'] = ((gaussians_color- 0.5) / 0.282).view(B, num_gaussians, 1, 3)
        else:
            gaussians['feature'] = gaussians_color


        gaussians_format: Gaussians = feat2gaussian(gaussians)

        return EncoderOutput(
            gaussians=gaussians_format,
            gaussians_mask=None,
        )


    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:

            return batch
        return data_shim
