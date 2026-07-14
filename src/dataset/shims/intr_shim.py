import torch

from ..types import BatchedExample, BatchedViews

def apply_intr_shim_to_views(views: BatchedViews,) -> BatchedViews:
    _, _, _, h, w = views["image"].shape

    # Adjust the intrinsics to account for the cropping.
    intrinsics = views["intrinsics"].clone()
    if torch.all(intrinsics[..., 0, 2] > 1.0):
        intrinsics[..., 0, :] /= w
        intrinsics[..., 1, :] /= h

    return {
        **views,
        "intrinsics": intrinsics,
    }


def apply_intr_shim(batch: BatchedExample) -> BatchedExample:
    return {
        **batch,
        "context": apply_intr_shim_to_views(batch["context"]),
        "target": apply_intr_shim_to_views(batch["target"]),
    }
