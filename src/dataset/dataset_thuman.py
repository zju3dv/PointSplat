import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_crop_shim
from .shims.extr_shim import preprocess_poses, preprocess_sampled_views
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetTHumanCfg(DatasetCfgCommon):

    name: Literal["thuman"]
    roots: "list[Path]"
    near: float = -1.0
    far: float = -1.0
    shuffle_val: bool = True
    max_val_batches: int = -1
    train_times_per_scene: int = 1
    num_training_scenes: int = -1
    num_val_scenes: int = -1
    time_range: int = 1  # THuman is single-frame data.
    start_scene_training: int = 0
    start_scene_val: int = 0
    random_view: bool = True
    num_views: int = 8
    norm_extrinsic: bool = True
    norm_sampled_views: bool = False
    scene_scale_factor: float = 1.35


class DatasetTHuman(Dataset):
    """Dataset for THuman2.0 render format.

    Expected layout, where root is datasets/THuman2_0_render:
        root/
          train/
            img/<frame_name>/<view_id>.jpg
            mask/<frame_name>/<view_id>.png
            depth/<frame_name>/<view_id>.png  (currently unused)
            parm/<frame_name>/{view_id}_intrinsic.npy
            parm/<frame_name>/{view_id}_extrinsic.npy
          val/
            ...

    frame_name follows xxxx_yyy:
        - xxxx: scene name
        - yyy: camera id
        - view_id: view id (0, 1, 2, ...)

    Each scene has one frame and may include multiple cameras.

    The returned example matches DatasetRENBODY and contains:
        "context" / "target" / "scene" / "timestep"
    context and target contain matching fields:
        extrinsics / intrinsics / image / mask / near / far / index / timesteps
    """

    cfg: DatasetTHumanCfg
    stage: Stage
    view_sampler: ViewSampler

    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetTHumanCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler

        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        base_root = cfg.roots[0] if cfg.roots else Path("data")
        if self.stage == "train":
            self.data_root = os.path.join(base_root, "train")
        elif self.stage in ("val", "test", "demo"):
            self.data_root = os.path.join(base_root, "val")
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 'train' or 'val'")

        self.img_dir = os.path.join(self.data_root, "img")
        self.mask_dir = os.path.join(self.data_root, "mask")
        self.depth_dir = os.path.join(self.data_root, "depth")
        self.parm_dir = os.path.join(self.data_root, "parm")

        self._init_scene_names()
        self._init_frame_settings()

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def _init_scene_names(self) -> None:
        """Collect frame names from img, extract scene names, and group by scene."""
        all_frame_names = [
            d
            for d in os.listdir(self.img_dir)
            if os.path.isdir(os.path.join(self.img_dir, d))
        ]
        all_frame_names = sorted(all_frame_names)

        scene_to_frames = {}
        for frame_name in all_frame_names:
            if "_" in frame_name:
                scene_name = frame_name.rsplit("_", 1)[0]
            else:
                scene_name = frame_name
            if scene_name not in scene_to_frames:
                scene_to_frames[scene_name] = []
            scene_to_frames[scene_name].append(frame_name)

        all_scene_names = sorted(scene_to_frames.keys())
        total_scene_num = len(all_scene_names)

        if self.stage == "train":
            if self.cfg.shuffle_val:
                all_scene_names = self.shuffle(all_scene_names)
            num_train_scenes = getattr(self.cfg, "num_training_scenes", -1)
            if num_train_scenes == -1:
                num_train_scenes = len(all_scene_names)
            start_train_scene_idx = getattr(self.cfg, "start_scene_training", 0)
            end_train_scene_idx = min(
                start_train_scene_idx + num_train_scenes, total_scene_num
            )
            all_scene_names = all_scene_names[start_train_scene_idx:end_train_scene_idx]
        elif self.stage in ("val", "test", "demo"):
            num_val_scenes = getattr(
                self.cfg, "num_val_scenes", len(all_scene_names)
            )
            if num_val_scenes == -1:
                num_val_scenes = len(all_scene_names)
            num_val_scenes = min(num_val_scenes, len(all_scene_names))
            start_val_scene_idx = getattr(self.cfg, "start_scene_val", 0)
            assert start_val_scene_idx >= 0
            end_val_scene_idx = min(
                start_val_scene_idx + num_val_scenes, len(all_scene_names)
            )
            all_scene_names = all_scene_names[start_val_scene_idx:end_val_scene_idx]
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 'train' or 'val'")

        self.all_scene_names = all_scene_names
        self.scene_to_frames = {
            scene: sorted(scene_to_frames[scene]) for scene in all_scene_names
        }

    def _init_frame_settings(self) -> None:
        """Set the single-frame THuman frame metadata."""
        self.time_range = 1
        self.max_frame_per_scene = 1
        self.frame_per_scene = 1
        self.frame_id_list = [0]
        self.total_frame_num = len(self.all_scene_names) * len(self.frame_id_list)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Tensor:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return value.repeat(num_views)

    def _list_cameras_for_scene(self, scene_name: str) -> "list[str]":
        """List available camera frame names for a scene."""
        frame_names = self.scene_to_frames.get(scene_name, [])
        valid_cameras = []
        for frame_name in frame_names:
            img_path = os.path.join(self.img_dir, frame_name, "2.jpg")
            if os.path.exists(img_path):
                valid_cameras.append(frame_name)
        return valid_cameras

    def _load_image_and_mask(self, frame_name: str):
        """Load one camera image and mask in DatasetRENBODY-compatible format."""
        img_path = os.path.join(self.img_dir, frame_name, "2.jpg")
        mask_path = os.path.join(self.mask_dir, frame_name, "2.png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        mask = np.array(mask) / 255.0
        mask = torch.from_numpy(mask).permute(2, 0, 1).mean(dim=0).unsqueeze(0).float()
        return image, mask

    def _load_intr_extr(self, frame_name: str):
        """Load intrinsics and extrinsics from the fixed THuman file names."""
        intr_path = os.path.join(self.parm_dir, frame_name, "2_intrinsic.npy")
        extr_path = os.path.join(self.parm_dir, frame_name, "2_extrinsic.npy")
        if not os.path.exists(intr_path):
            raise FileNotFoundError(intr_path)
        if not os.path.exists(extr_path):
            raise FileNotFoundError(extr_path)

        intr = np.load(intr_path)
        extr = np.load(extr_path)  # THuman extrinsic is W2C

        if intr.ndim == 2 and intr.shape[0] == 3 and intr.shape[1] == 3:
            K = torch.from_numpy(intr).float()
        else:
            intr = np.array(intr).reshape(-1)
            assert intr.size >= 4, "Intrinsic npy must contain at least [fx, fy, cx, cy]"
            fx, fy, cx, cy = intr[:4]
            K = torch.eye(3, dtype=torch.float32)
            K[0, 0] = float(fx)
            K[1, 1] = float(fy)
            K[0, 2] = float(cx)
            K[1, 2] = float(cy)

        extr = np.array(extr)
        if extr.ndim == 2 and extr.shape == (4, 4):
            extr_4x4 = torch.from_numpy(extr).float()
        elif extr.ndim == 2 and extr.shape == (3, 4):
            extr_4x4 = torch.eye(4, dtype=torch.float32)
            extr_4x4[:3, :] = torch.from_numpy(extr).float()
        else:
            raise ValueError(f"Unexpected extrinsic shape: {extr.shape}")

        c2w = extr_4x4.inverse()
        return K, c2w

    def __getitem__(self, idx: int):
        if (
            self.stage == "val"
            and self.cfg.max_val_batches > 0
            and self.cfg.shuffle_val
        ):
            idx = random.randint(0, self.total_frame_num - 1)

        idx = idx % self.total_frame_num
        scene_idx = idx // self.frame_per_scene
        scene_name = self.all_scene_names[scene_idx]

        frame_idx = 0

        cameras = self._list_cameras_for_scene(scene_name)
        if len(cameras) == 0:
            raise RuntimeError(f"No valid cameras found for scene {scene_name}")

        max_views = min(self.cfg.num_views, len(cameras))
        cameras = cameras[:max_views]

        intrinsics_list = []
        c2ws_list = []
        images_meta = []  # frame_name list
        timesteps_list = [] 

        for frame_name in cameras:
            try:
                K, c2w = self._load_intr_extr(frame_name)
                intrinsics_list.append(K)
                c2ws_list.append(c2w)
                images_meta.append(frame_name)
                timesteps_list.append(torch.tensor(frame_idx, dtype=torch.int64))
            except Exception as e:
                print(f"Error processing camera {frame_name}: {e}")
                raise e

        if len(intrinsics_list) == 0:
            raise RuntimeError(f"All views failed to load for scene {scene_name}")

        intrinsics_tensor = torch.stack(intrinsics_list, dim=0)  # (V, 3, 3)
        c2ws_tensor = torch.stack(c2ws_list, dim=0)  # (V, 4, 4)
        timesteps_tensor = torch.stack(timesteps_list, dim=0)  # (V,)

        scene_scale = 1.0 / self.cfg.scene_scale_factor
        if self.cfg.norm_extrinsic:
            c2ws_tensor, scene_scale = preprocess_poses(
                c2ws_tensor,
                scene_scale_factor=self.cfg.scene_scale_factor,
                align_to_dna_rendering=True,
                return_scale=True,
            )

        context_indices, target_indices = self.view_sampler.sample(
            scene_name,
            c2ws_tensor,
            intrinsics_tensor,
        )

        context_images = []
        context_masks = []
        target_images = []
        target_masks = []

        for index in context_indices:
            frame_name = images_meta[index]
            image, mask = self._load_image_and_mask(frame_name)
            context_images.append(image)
            context_masks.append(mask)

        for index in target_indices:
            frame_name = images_meta[index]
            image, mask = self._load_image_and_mask(frame_name)
            target_images.append(image)
            target_masks.append(mask)

        context_c2ws = c2ws_tensor[context_indices]
        context_intrinsics = intrinsics_tensor[context_indices]
        context_timesteps = timesteps_tensor[context_indices]

        target_c2ws = c2ws_tensor[target_indices]
        target_intrinsics = intrinsics_tensor[target_indices]
        target_timesteps = timesteps_tensor[target_indices]

        context_images = torch.stack(context_images, dim=0)
        context_masks = torch.stack(context_masks, dim=0)
        target_images = torch.stack(target_images, dim=0)
        target_masks = torch.stack(target_masks, dim=0)

        nf_scale = 1.0
        if self.cfg.norm_sampled_views:
            context_c2ws, target_c2ws = preprocess_sampled_views(
                context_c2ws, target_c2ws, scene_scale=scene_scale
            )

        example = {
            "context": {
                "extrinsics": context_c2ws,
                "intrinsics": context_intrinsics,
                "image": context_images,
                "mask": context_masks,
                "near": self.get_bound("near", len(context_indices)) / nf_scale,
                "far": self.get_bound("far", len(context_indices)) / nf_scale,
                "index": context_indices,
                "timesteps": context_timesteps,
            },
            "target": {
                "extrinsics": target_c2ws,
                "intrinsics": target_intrinsics,
                "image": target_images,
                "mask": target_masks,
                "near": self.get_bound("near", len(target_indices)) / nf_scale,
                "far": self.get_bound("far", len(target_indices)) / nf_scale,
                "index": target_indices,
                "timesteps": target_timesteps,
            },
            "scene": scene_name,
            "timestep": f"{frame_idx:06d}",
        }

        example = apply_crop_shim(example, tuple(self.cfg.image_shape))
        return example

    def __len__(self) -> int:
        if self.stage == "train":
            return self.total_frame_num * self.cfg.train_times_per_scene
        elif self.stage == "val":
            return (
                self.cfg.max_val_batches
                if self.cfg.max_val_batches > 0
                else self.total_frame_num
            )
        else:
            return self.total_frame_num
