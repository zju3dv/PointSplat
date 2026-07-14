<div align="center">


## PointSplat: Compact Gaussian Splatting via Human-Centric Prediction

**ECCV 2026**

[Yujie Guo](https://guoyujie.cn/)<sup>1</sup>, [Yudong Jin](https://github.com/krahets)<sup>1</sup>, [Lingteng Qiu](https://lingtengqiu.github.io/)<sup>3</sup>, [Zehong Shen](https://zehongs.github.io/)<sup>1</sup>, [Zhen Xu](https://zhenx.me/)<sup>1</sup>, Jing Zhang<sup>2</sup>, Xianchao Shen<sup>2</sup>, [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/)<sup>1</sup>, [Sida Peng](https://pengsida.net/)<sup>1</sup>, [Xiaowei Zhou](https://xzhou.me/)<sup>1†</sup>

<sup>1</sup>Zhejiang University &nbsp;&nbsp; <sup>2</sup>ByteDance &nbsp;&nbsp; <sup>3</sup>CUHK, Shenzhen

<sup>†</sup>Corresponding author

[![Project Page](https://img.shields.io/badge/Project-Page-5f8fbd)](https://zju3dv.github.io/pointsplat)
[![arXiv](https://img.shields.io/badge/arXiv-2606.32036-c68b5b)](https://arxiv.org/abs/2606.32036)

<img src="assets/overview.gif" width="100%">

</div>

## ⚙️ Installation

Create a conda environment and install the dependencies:

```bash
conda create -n pointsplat python=3.10 -y
conda activate pointsplat
pip install -r requirements.txt
```

## 📦 Checkpoint

Download the pretrained checkpoint with:

```bash
hf download Yujie0012/PointSplat_pretrained_weights pointsplat_mixed.pt \
  --local-dir pretrained
```

You can also download the [checkpoint](https://huggingface.co/Yujie0012/PointSplat_pretrained_weights) manually. Either way, the final layout should be:

```text
pretrained/
└── pointsplat_mixed.pt
```

## 🚀 Inference

### Inference on DNA-Rendering

**Step 1: Download example data.**

Download the DNA-Rendering example package from [PointSplat_example_data](https://huggingface.co/datasets/Yujie0012/PointSplat_example_data) with:

```bash
hf download Yujie0012/PointSplat_example_data DNA_Rendering_example.zip \
  --repo-type dataset \
  --local-dir datasets
unzip datasets/DNA_Rendering_example.zip -d datasets
```

Expected data layout:

```text
datasets/DNA_Rendering_example/
├── validation_index.json
├── scenes/
│   └── <scene_id>/
│       ├── transforms.json
│       └── images/
│           └── <view_id>/
│               └── 000030.webp
└── masks/
    └── <scene_id>/
        └── fmasks/
            └── <view_id>/
                └── 000030.png
```

> [!TIP]
> To test the model on more DNA-Rendering scenes, follow the [processed DNA-Rendering dataset](https://github.com/zju3dv/Diffuman4D/blob/main/README.md#processed-dna-rendering-dataset) instructions in Diffuman4D.

**Step 2: Run the inference script.**

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main \
  +experiment=pointsplat_renbody \
  output_dir=experiments/eval_pointsplat_dna_rendering \
  dataset.roots='[datasets/DNA_Rendering_example]' \
  checkpointing.pretrained_encoder=pretrained/pointsplat_mixed.pt
```

### Inference on THuman2.0

**Step 1: Download example data.**

Download the THuman2.0 example package from [PointSplat_example_data](https://huggingface.co/datasets/Yujie0012/PointSplat_example_data) with:

```bash
hf download Yujie0012/PointSplat_example_data THuman2_0_example.zip \
  --repo-type dataset \
  --local-dir datasets
unzip datasets/THuman2_0_example.zip -d datasets
```

Expected data layout:

```text
datasets/THuman2_0_example/
└── val/
    ├── img/
    │   └── <scene_id>_<view_id>/
    │       └── 2.jpg
    ├── mask/
    │   └── <scene_id>_<view_id>/
    │       └── 2.png
    └── parm/
        └── <scene_id>_<view_id>/
            ├── 2_intrinsic.npy
            └── 2_extrinsic.npy
```


> [!TIP]
> For the full THuman2.0 training/test data, refer to the [dataset preparation](https://github.com/aipixel/GPS-Gaussian/tree/main#dataset-preparation) instructions in [GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian/tree/main).

**Step 2: Run the inference script.**

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main \
  +experiment=pointsplat_thuman \
  output_dir=experiments/eval_pointsplat_thuman \
  dataset.roots='[datasets/THuman2_0_example]' \
  checkpointing.pretrained_encoder=pretrained/pointsplat_mixed.pt
```

## 📚 Citation

If you find this code useful for your research, please consider citing:

```bibtex
@article{guo2026pointsplat,
  title={PointSplat: Compact Gaussian Splatting via Human-Centric Prediction},
  author={Yujie Guo and Yudong Jin and Lingteng Qiu and Zehong Shen and Zhen Xu and Jing Zhang and Xianchao Shen and Hujun Bao and Sida Peng and Xiaowei Zhou},
  journal={arXiv preprint arXiv:2606.32036},
  year={2026}
}
```
