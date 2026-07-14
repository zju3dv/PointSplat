import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import torch.nn.functional as F

from ..types import AnyExample, AnyViews


def rescale_image(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    """Rescale a single image using LANCZOS resampling."""
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.Resampling.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def rescale_and_crop_tensors(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    target_shape: tuple[int, int],
    masks: Tensor | None = None,
    depths: Tensor | None = None,
    center_crop: bool = True,
) -> dict[str, Tensor]:
    """
    Rescale and center-crop images/masks/depths to target shape, adjusting intrinsics.

    Returns a dict with keys: 'image', 'intrinsics', and optionally 'mask', 'depth'.
    """
    *_, h_in, w_in = images.shape
    h_out, w_out = target_shape
    assert h_out <= h_in and w_out <= w_in, f"Target shape {target_shape} must be <= input shape ({h_in}, {w_in})"

    # Step 1: Rescale to intermediate size (matching either height or width)
    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Rescale images using high-quality LANCZOS
    *batch, c, h, w = images.shape
    images_flat = images.reshape(-1, c, h, w)
    images_scaled = torch.stack([rescale_image(img, (h_scaled, w_scaled)) for img in images_flat])
    images_scaled = images_scaled.reshape(*batch, c, h_scaled, w_scaled)

    # Rescale masks and depths using bilinear interpolation
    masks_scaled = None
    if masks is not None:
        masks_scaled = F.interpolate(
            masks, size=(h_scaled, w_scaled), mode="bilinear", align_corners=True
        )

    depths_scaled = None
    if depths is not None:
        depths_scaled = F.interpolate(
            depths.unsqueeze(1), size=(h_scaled, w_scaled), mode="bilinear", align_corners=True
        ).squeeze(1)

    # Step 2: Center crop to target size
    if center_crop:
        row = (h_scaled - h_out) // 2
        col = (w_scaled - w_out) // 2
        images_cropped = images_scaled[..., :, row : row + h_out, col : col + w_out]
    else:
        row = 0
        col = 0
        images_cropped = images_scaled

    # Step 3: Adjust intrinsics based on actual scaling ratios
    # Following the logic from dataset_renbody.py
    intrinsics_adjusted = intrinsics.clone()
    resize_ratio_x = w_scaled / w_in
    resize_ratio_y = h_scaled / h_in

    # Scale fx, fy, cx, cy by their respective ratios
    intrinsics_adjusted[..., 0, 0] *= resize_ratio_x  # fx
    intrinsics_adjusted[..., 1, 1] *= resize_ratio_y  # fy
    intrinsics_adjusted[..., 0, 2] *= resize_ratio_x  # cx
    intrinsics_adjusted[..., 1, 2] *= resize_ratio_y  # cy

    # Adjust principal point for crop offset
    if center_crop:
        intrinsics_adjusted[..., 0, 2] -= col  # cx -= start_w
        intrinsics_adjusted[..., 1, 2] -= row  # cy -= start_h

    # Build result dict
    result = {
        "image": images_cropped,
        "intrinsics": intrinsics_adjusted,
    }

    if masks_scaled is not None:
        result["mask"] = masks_scaled[..., :, row : row + h_out, col : col + w_out]

    if depths_scaled is not None:
        result["depth"] = depths_scaled[..., row : row + h_out, col : col + w_out]

    return result


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    """Apply rescale and crop to a single view (context or target)."""
    processed = rescale_and_crop_tensors(
        images=views["image"],
        intrinsics=views["intrinsics"],
        target_shape=shape,
        masks=views.get("mask", None),
        depths=views.get("depth", None),
    )

    # Update views with processed data
    return {**views, **processed}


def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }
