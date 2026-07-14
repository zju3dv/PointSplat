import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_crop_shim
from .shims.extr_shim import preprocess_poses
from .types import Stage
from .view_sampler import ViewSampler

@dataclass
class DatasetRENBODYCfg(DatasetCfgCommon):
    name: Literal["renbody"]
    roots: "list[Path]"
    near: float = -1.0
    far: float = -1.0
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    num_training_scenes: int = -1
    num_val_scenes: int = -1
    start_scene_training: int = 0
    start_scene_val: int = 0
    frame_per_training_scene: int = 150
    frame_per_val_scene: int = 1
    start_frame_training: int = 0
    start_frame_val: int = 0
    frame_gap_training: int = 1
    frame_gap_val: int = 1
    random_view: bool = True
    num_views: int = 8
    norm_extrinsic: bool = True
    scene_scale_factor: float = 1.35

class DatasetRENBODY(Dataset):
    cfg: DatasetRENBODYCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.1
    far: float = 1000.0

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __init__(
        self,
        cfg: DatasetRENBODYCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        self.data_root = cfg.roots[0] if cfg.roots else Path("data")  # Use first root as data_root
        self.scene_dir = os.path.join(self.data_root, "scenes")
        self.mask_dir = os.path.join(self.data_root, "masks")

        # Load validation scene names from data_root/validation_index.json if available
        # Fallback to the default hardcoded list otherwise
        self.validation_scene_names = None
        val_index_path = os.path.join(self.data_root, "validation_index.json")
        if os.path.isfile(val_index_path):
            try:
                with open(val_index_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    self.validation_scene_names = data
            except Exception:
                self.validation_scene_names = None
        assert self.validation_scene_names is not None

        all_scene_names = [d for d in os.listdir(
            self.scene_dir) if os.path.isdir(os.path.join(self.scene_dir, d))]
        all_scene_names = sorted(all_scene_names)

        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            all_scene_names = self.shuffle(all_scene_names)

        all_scene_names = [p for p in all_scene_names if os.path.exists(os.path.join(self.mask_dir, p))]
        total_scene_num = len(all_scene_names)


        if self.stage == 'train':
            all_scene_names = [p for p in all_scene_names if p not in self.validation_scene_names]
            num_train_scenes = getattr(cfg, 'num_training_scenes', -1)
            if num_train_scenes == -1:
                num_train_scenes = len(all_scene_names)
            start_train_scene_idx = getattr(cfg, 'start_scene_training', 0)
            end_train_scene_idx = min(start_train_scene_idx + num_train_scenes, total_scene_num)
            all_scene_names = all_scene_names[start_train_scene_idx:end_train_scene_idx]
        elif self.stage == 'val' or self.stage == 'test':
            all_scene_names = self.validation_scene_names # Align with Diffuman4D validation set
            num_val_scenes = getattr(cfg, 'num_val_scenes', len(all_scene_names))
            if num_val_scenes == -1:
                num_val_scenes = len(all_scene_names)
            num_val_scenes = min(num_val_scenes, len(all_scene_names))
            start_val_scene_idx = getattr(cfg, 'start_scene_val', 0)
            assert start_val_scene_idx >= 0
            end_val_scene_idx = min(start_val_scene_idx + num_val_scenes, len(all_scene_names))
            all_scene_names = all_scene_names[start_val_scene_idx:end_val_scene_idx]
        elif self.stage == 'demo':
            num_val_scenes = getattr(cfg, 'num_val_scenes', len(all_scene_names))
            if num_val_scenes == -1:
                num_val_scenes = len(all_scene_names)
            num_val_scenes = min(num_val_scenes, len(all_scene_names))
            start_val_scene_idx = getattr(cfg, 'start_scene_val', 0)
            assert start_val_scene_idx >= 0
            end_val_scene_idx = min(start_val_scene_idx + num_val_scenes, len(all_scene_names))
            all_scene_names = all_scene_names[start_val_scene_idx:end_val_scene_idx]
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 'train' or 'val'")

        self.all_scene_paths = [
            os.path.join(self.scene_dir, d) for d in all_scene_names]
        self.all_mask_paths = [
            os.path.join(self.mask_dir, d) for d in all_scene_names]

        self.all_scene_paths = sorted(self.all_scene_paths)
        self.all_mask_paths = sorted(self.all_mask_paths)

        self.camera_count = 48

        def load_scene_info(scene_path):
            try:
                scene_info = json.load(
                    open(os.path.join(scene_path, "transforms.json")))
                return scene_info
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=min(32, len(self.all_scene_paths))) as executor:
            scene_infos = list(executor.map(
                load_scene_info, self.all_scene_paths))
        self.scene_infos = [info for info in scene_infos if info is not None]

        self.max_frame_per_scene = 150
        if self.stage == 'train':
            self.frame_per_scene = getattr(cfg, 'frame_per_training_scene', 150)
            start_frame = getattr(cfg, 'start_frame_training', 0)
            frame_gap = getattr(cfg, 'frame_gap_training', 1)
        else:
            self.frame_per_scene = getattr(cfg, 'frame_per_val_scene', 1)
            start_frame = getattr(cfg, 'start_frame_val', 0)
            frame_gap = getattr(cfg, 'frame_gap_val', 1)
        assert start_frame + frame_gap * self.frame_per_scene <= self.max_frame_per_scene
        end_frame = start_frame + frame_gap * self.frame_per_scene
        self.frame_id_list = list(range(start_frame, end_frame, frame_gap))
        self.total_frame_num = len(self.scene_infos) * len(self.frame_id_list)

    def load_images(self, scene_dir, mask_dir, frame):
        """Load image and mask from disk without any resizing or cropping."""
        candidate_ext = ['png', 'webp', 'jpg', 'jpeg']
        image_path = None
        mask_path = None
        for ext in candidate_ext:
            image_candidate = os.path.join(scene_dir, f'images/{frame["camera_label"]}/{frame["timestep"]:06d}.{ext}')
            if os.path.exists(image_candidate):
                image_path = image_candidate
            mask_candidate = os.path.join(mask_dir, f'fmasks/{frame["camera_label"]}/{frame["timestep"]:06d}.{ext}')
            if os.path.exists(mask_candidate):
                mask_path = mask_candidate
        if mask_path is None or image_path is None:
            if image_path is None:
                print(f"Image not found for: {scene_dir} {frame['camera_label']} {frame['timestep']:06d}")
            if mask_path is None:
                print(f"Mask not found for: {scene_dir} {frame['camera_label']} {frame['timestep']:06d}")
            raise FileNotFoundError(f"File not found for: {scene_dir} {frame['camera_label']} {frame['timestep']:06d}")

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = np.array(mask) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        fxfycxcy = np.array(
            [frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]])
        fxfycxcy = torch.from_numpy(fxfycxcy).float()

        return image, fxfycxcy, mask

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Tensor:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __getitem__(self, idx):
        idx = idx % self.total_frame_num
        scene_idx = idx // self.frame_per_scene
        frame_idx = self.frame_id_list[idx % self.frame_per_scene]
        scene_path = self.all_scene_paths[scene_idx]
        mask_path = self.all_mask_paths[scene_idx]
        scene_info = self.scene_infos[scene_idx]

        ply_file_path = scene_info["ply_file_path"]
        camera_list = scene_info["frames"]

        camera_ids = list(range(self.cfg.num_views))
        camera_list = [camera_list[i] for i in camera_ids]

        intrinsics = []
        c2ws = []
        images = []
        masks = []

        scene_name = scene_path.split("/")[-1]
        for cam in camera_list:
            base_frame_path = cam["file_path"]
            try:
                frame_path = base_frame_path.replace("000000", f"{frame_idx:06d}")
                frame = cam.copy()
                frame["file_path"] = frame_path
                frame["timestep"] = frame_idx
                image, intri, mask = self.load_images(scene_path, mask_path, frame)
                c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1
                intrinsics.append(intri)
                c2ws.append(c2w)
                images.append(image)
                masks.append(mask)

            except Exception as e:
                raise RuntimeError(f"Error processing image {frame_path}: {e}") from e

        intrinsics_tensor = torch.stack(intrinsics, dim=0)
        c2ws_tensor = torch.stack(c2ws, dim=0)
        images_tensor = torch.stack(images, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        if self.cfg.norm_extrinsic:
            c2ws_tensor = preprocess_poses(
                c2ws_tensor,
                scene_scale_factor=self.cfg.scene_scale_factor
            )

        intrinsics_3x3 = []
        for intri in intrinsics_tensor:
            fx, fy, cx, cy = intri
            K = torch.eye(3, dtype=torch.float32)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            intrinsics_3x3.append(K)
        intrinsics_3x3 = torch.stack(intrinsics_3x3, dim=0)

        try:
            context_indices, target_indices = self.view_sampler.sample(
                scene_name,
                c2ws_tensor,  # extrinsics
                intrinsics_3x3,  # intrinsics
            )
        except ValueError:
            # Fallback if view sampling fails - split manually
            num_context = len(camera_ids) // 2
            all_indices = torch.arange(len(camera_ids))
            context_indices = all_indices[:num_context]
            target_indices = all_indices[num_context:]

        nf_scale = 1.0
        example = {
            "context": {
                "extrinsics": c2ws_tensor[context_indices],
                "intrinsics": intrinsics_3x3[context_indices],
                "image": images_tensor[context_indices],
                "mask": masks_tensor[context_indices],
                "near": self.get_bound("near", len(context_indices)) / nf_scale,
                "far": self.get_bound("far", len(context_indices)) / nf_scale,
                "index": context_indices,
            },
            "target": {
                "extrinsics": c2ws_tensor[target_indices],
                "intrinsics": intrinsics_3x3[target_indices],
                "image": images_tensor[target_indices],
                "mask": masks_tensor[target_indices],
                "near": self.get_bound("near", len(target_indices)) / nf_scale,
                "far": self.get_bound("far", len(target_indices)) / nf_scale,
                "index": target_indices,
            },
            "scene": scene_name,
            "timestep": f"{frame_idx:06d}"
        }

        example = apply_crop_shim(example, tuple(self.cfg.image_shape))

        return example

    def __len__(self):
        if self.stage == "train":
            return self.total_frame_num * self.cfg.train_times_per_scene
        else:
            return self.total_frame_num
