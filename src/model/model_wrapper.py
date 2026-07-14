import torch
from einops import rearrange
from pytorch_lightning import LightningModule
import numpy as np
import json
import os
import torch.nn.functional as F
import math
from PIL import Image

from ..dataset.data_module import get_data_shim
from ..dataset import DatasetCfg
from ..global_cfg import get_cfg

from .encoder import Encoder, EncoderOutput
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .decoder.decoder import Decoder, DecoderOutput

from ..utils.benchmarker import Benchmarker
from ..utils.image_io import save_image, save_video
from ..utils.step_tracker import StepTracker
from ..utils.metrics_utils import compute_lpips, compute_psnr, compute_ssim
from ..utils.typing_utils import *
from ..utils.visualization.vis_depth import viz_depth_tensor
from ..utils.stablize_camera import render_stabilization_path
from ..utils.gaussiansplatting_utils import save_gaussians


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    save_gt_image: bool
    save_input_images: bool
    save_depth: bool
    save_depth_concat_img: bool
    save_depth_npy: bool
    save_gaussian: bool
    render_chunk_size: int | None
    stablize_camera: bool
    stab_camera_kernel: int
    downsample_context_ratio: float
    mask_output: bool
    move_to_cpu: bool


class ModelWrapper(LightningModule):
    dataset_cfg: DatasetCfg
    encoder: Encoder
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    test_cfg: TestCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        test_cfg: TestCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        step_tracker: StepTracker | None
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.test_cfg = test_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)

        # This is used for testing.
        self.benchmarker = Benchmarker()

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def test_step(self, batch, batch_idx):
        batch = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1


        # Downsample context images and mask if needed
        if self.test_cfg.downsample_context_ratio != 1.0:
            h_target = int(h * self.test_cfg.downsample_context_ratio)
            w_target = int(w * self.test_cfg.downsample_context_ratio)

            context_img = batch["context"]["image"]
            b, v, c, h_c, w_c = context_img.shape
            context_img = context_img.reshape(b * v, c, h_c, w_c)
            context_img = F.interpolate(
                context_img,
                size=(h_target, w_target),
                mode="bilinear",
                align_corners=True,
            ).reshape(b, v, c, h_target, w_target)
            batch["context"]["image"] = context_img

            mask = batch["context"].get("mask", None)
            if mask is not None:
                # mask may be (b, v, 1, h, w) or (b, v, h, w)
                orig_shape = mask.shape
                if mask.ndim == 5:
                    mask = mask.reshape(b * v, orig_shape[2], orig_shape[3], orig_shape[4])
                    mask = F.interpolate(
                        mask,
                        size=(h_target, w_target),
                        mode="bilinear",
                        align_corners=True,
                    )
                    mask = mask.reshape(b, v, orig_shape[2], h_target, w_target)
                elif mask.ndim == 4:
                    mask = mask.reshape(b * v, 1, orig_shape[2], orig_shape[3])
                    mask = F.interpolate(
                        mask,
                        size=(h_target, w_target),
                        mode="bilinear",
                        align_corners=True,
                    )
                    mask = mask.reshape(b, v, h_target, w_target)
                batch["context"]["mask"] = mask

        scene_name = f'{batch["scene"][0]}_{batch["timestep"][0]}' if 'timestep' in batch else batch["scene"][0]

        # save input views for visualization
        if self.test_cfg.save_input_images:
            self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
            path = Path(get_cfg()["output_dir"])

            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            for idx, color in zip(index, input_images):
                save_image(color, path / "images" / scene_name / f"color/input_{idx:0>6}.png")

        # save depth vis
        if self.test_cfg.save_depth or self.test_cfg.save_gaussian:
            visualization_dump = {}
        else:
            visualization_dump = None

        #######################
        # Encoder
        #######################

        with self.benchmarker.time("encoder"):
            encoder_output: EncoderOutput = self.encoder(
                batch["context"],
                self.global_step,
                # deterministic=False,
                visualization_dump=visualization_dump,
            )

        pred_depths = encoder_output.depth
        gaussians = encoder_output.gaussians
        encoder_output.context = batch["context"]

        #######################
        # Decoder
        #######################
        with self.benchmarker.time("decoder", num_calls=v):

            camera_poses = batch["target"]["extrinsics"]

            if self.test_cfg.stablize_camera:
                stable_poses = render_stabilization_path(
                    camera_poses[0].detach().cpu().numpy(),
                    k_size=self.test_cfg.stab_camera_kernel,
                )

                stable_poses = list(
                    map(
                        lambda x: np.concatenate(
                            (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                        ),
                        stable_poses,
                    )
                )
                stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                    camera_poses
                )
                camera_poses = stable_poses.unsqueeze(0)


            if self.test_cfg.render_chunk_size is not None:
                chunk_size = self.test_cfg.render_chunk_size
                num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                output = None
                for i in range(num_chunks):
                    start = chunk_size * i
                    end = chunk_size * (i + 1)

                    render_intrinsics = batch["target"]["intrinsics"]
                    render_near = batch["target"]["near"]
                    render_far = batch["target"]["far"]

                    curr_output: DecoderOutput = self.decoder.forward(
                        encoder_output,
                        camera_poses[:, start:end],
                        render_intrinsics[:, start:end],
                        render_near[:, start:end],
                        render_far[:, start:end],
                        (h, w),
                        depth_mode=None
                    )

                    if i == 0:
                        output = curr_output
                    else:
                        # ignore depth
                        output.color = torch.cat(
                            (output.color, curr_output.color), dim=1
                        )

                    # Free intermediate chunk output
                    if i > 0:
                        del curr_output
            else:
                output: DecoderOutput = self.decoder.forward(
                    encoder_output,
                    camera_poses,
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=None,
                )

        # ============================================
        # Step 1: Optionally move inference results to CPU to prevent GPU memory accumulation
        # ============================================

        # Determine target device based on config
        target_device = 'cpu' if self.test_cfg.move_to_cpu else self.device

        # Move rendering results to target device
        images_prob = output.color[0].detach().to(target_device)  # [v, c, h, w]
        rgb_gt = batch["target"]["image"][0].detach().to(target_device)  # [v, c, h, w]

        if self.test_cfg.mask_output and batch["target"].get("mask", None) is not None:
            mask = batch["target"]["mask"][0].detach().to(target_device).expand_as(rgb_gt)
            background_color = torch.tensor(self.dataset_cfg.background_color, dtype=torch.float32, device=target_device)[None, :, None, None]
            rgb_gt = rgb_gt * mask + (1.0 -mask) * background_color
            images_prob = images_prob * mask + (1.0 - mask) * background_color
        else:
            mask = None

        # Move depth results to target device
        if self.test_cfg.save_depth:
            if pred_depths is not None:
                depth = pred_depths[0].detach().to(target_device)  # [V, H, W]
            else:
                depth = visualization_dump["depth"][0, :, :, :, 0, 0].detach().to(target_device)  # [V, H, W]

            context_index = batch["context"]["index"][0].to(target_device)

        # Store other needed info
        target_index = batch["target"]["index"][0].to(target_device)
        context_index_for_video = batch["context"]["index"][0].to(target_device)

        if encoder_output.gaussians_mask is not None:
            gaussians_num = (encoder_output.gaussians_mask > 0).sum().item()
        else:
            gaussians_num = gaussians.means.size(1)
        # save gaussians BEFORE clearing GPU (needs GPU tensors)
        if self.test_cfg.save_gaussian:
            path_for_gaussian = Path(get_cfg()["output_dir"])
            save_path = path_for_gaussian / 'gaussians' / (scene_name + '.ply')
            save_gaussians(gaussians, encoder_output.gaussians_mask, batch, save_path)

        # Clear GPU memory if data was moved to CPU
        if self.test_cfg.move_to_cpu:
            del encoder_output, gaussians, output
            if pred_depths is not None:
                del pred_depths
            if visualization_dump is not None:
                del visualization_dump
            torch.cuda.empty_cache()

        # ============================================
        # Step 2: Process data (on CPU or GPU depending on config)
        # ============================================

        self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
        path = Path(get_cfg()["output_dir"])

        # save depth
        if self.test_cfg.save_depth:
            if self.test_cfg.save_depth_concat_img:
                # concat (img0, img1, depth0, depth1)
                image = batch['context']['image'][0].detach().cpu()  # [V, 3, H, W] in [0,1]
                image = rearrange(image, "b c h w -> h (b w) c")  # [H, VW, 3]
                image_concat = (image.numpy() * 255).astype(np.uint8)  # [H, VW, 3]
                depth_concat = []

            for idx, depth_i in zip(context_index, depth):
                depth_viz = viz_depth_tensor(
                    1.0 / depth_i, return_numpy=True
                )  # [H, W, 3]

                if self.test_cfg.save_depth_concat_img:
                    depth_concat.append(depth_viz)

                save_path = path / "images" / scene_name / "depth" / f"{idx:0>6}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(depth_viz).save(save_path)

                # save depth as npy
                if self.test_cfg.save_depth_npy:
                    depth_npy = depth_i.numpy()
                    save_path = path / "images" / scene_name / "depth" / f"{idx:0>6}.npy"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(save_path, depth_npy)

            if self.test_cfg.save_depth_concat_img:
                depth_concat = np.concatenate(depth_concat, axis=1)  # [H, VW, 3]
                concat = np.concatenate((image_concat, depth_concat), axis=0)  # [2H, VW, 3]

                save_path = path / "images" / scene_name / "depth" /  f"img_depth_{scene_name}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(concat).save(save_path)

            # Clear depth data
            del depth
            if self.test_cfg.save_depth_concat_img:
                del image_concat, depth_concat, concat

        # Save renderings (all from CPU)
        if self.test_cfg.save_image:
            if self.test_cfg.save_gt_image:
                for index, color, gt in zip(target_index, images_prob, rgb_gt):
                    save_image(color, path / "images" / scene_name / f"color/{index:0>6}.png")
                    save_image(gt, path / "images" / scene_name / f"color/{index:0>6}_gt.png")
            else:
                for index, color in zip(target_index, images_prob):
                    save_image(color, path / "images" / scene_name / f"color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in context_index_for_video])
            save_video(
                [a for a in images_prob],
                path / "videos" / f"{scene_name}_frame_{frame_str}.mp4",
            )

        # compute scores (move data to GPU only when computing)
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v

            if 'gs_num' not in self.test_step_outputs:
                self.test_step_outputs['gs_num'] = []
            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs['gs_num'].append(gaussians_num)
            if self.test_cfg.mask_output and batch["target"].get("mask", None) is not None:
                psnr_list = []
                ssim_list = []
                lpips_list = []
                for i in range(v):
                    mask_indices = mask[i].nonzero(as_tuple=False)
                    min_h = mask_indices[:, 1].min()
                    max_h = mask_indices[:, 1].max()
                    min_w = mask_indices[:, 2].min()
                    max_w = mask_indices[:, 2].max()
                    # Ensure data is on GPU for metric computation
                    rgb_i = images_prob[i, :, min_h:max_h, min_w:max_w].unsqueeze(0)
                    rgb_gt_i = rgb_gt[i, :, min_h:max_h, min_w:max_w].unsqueeze(0)
                    if target_device == 'cpu':
                        rgb_i = rgb_i.to(self.device)
                        rgb_gt_i = rgb_gt_i.to(self.device)

                    psnr_list.append(compute_psnr(rgb_gt_i, rgb_i).mean().item())
                    ssim_list.append(compute_ssim(rgb_gt_i, rgb_i).mean().item())
                    lpips_list.append(compute_lpips(rgb_gt_i, rgb_i).mean().item())

                self.test_step_outputs[f"psnr"].append(np.mean(psnr_list))
                self.test_step_outputs[f"ssim"].append(np.mean(ssim_list))
                self.test_step_outputs[f"lpips"].append(np.mean(lpips_list))
            else:
                rgb_gpu = images_prob.to(self.device)
                rgb_gt_gpu = rgb_gt.to(self.device)

                self.test_step_outputs[f"psnr"].append(
                    compute_psnr(rgb_gt_gpu, rgb_gpu).mean().item()
                )
                self.test_step_outputs[f"ssim"].append(
                    compute_ssim(rgb_gt_gpu, rgb_gpu).mean().item()
                )
                self.test_step_outputs[f"lpips"].append(
                    compute_lpips(rgb_gt_gpu, rgb_gpu).mean().item()
                )

        # Clean up
        if target_device == 'cpu':
            del images_prob, rgb_gt
            torch.cuda.empty_cache()

    def on_test_end(self) -> None:
        out_dir = Path(self.test_cfg.output_path)
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()
