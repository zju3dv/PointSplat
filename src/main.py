import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import hydra
import torch
from jaxtyping import install_import_hook

from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.utilities import rank_zero_only

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.utils.typing_utils import *
    from src.utils.step_tracker import StepTracker
    from src.utils.ckpt_utils import load_pretrained_model
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

@rank_zero_only
def print_rank0(*args, **kwargs):
    print(*args, **kwargs)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def main(cfg_dict: DictConfig):

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print_rank0(f"Saving outputs to {output_dir}.")

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        precision=cfg.trainer.precision,
        accelerator="gpu",
        logger=False,
        devices=torch.cuda.device_count(),
        strategy='ddp' if torch.cuda.device_count() > 1 else "auto",
        enable_progress_bar=True,
        num_nodes=cfg.trainer.num_nodes,
        plugins=LightningEnvironment() if cfg.trainer.use_plugins else None,
    )
    torch.manual_seed(cfg_dict.trainer.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_wrapper = ModelWrapper(
        cfg.dataset,
        cfg.test,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        step_tracker
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    print_rank0("test:", len(data_module.test_dataloader()))
    checkpoint_path = Path(cfg.checkpointing.load) if cfg.checkpointing.load else None

    # load pretrained model
    status = load_pretrained_model(model_wrapper, cfg.checkpointing)
    print_rank0(status)

    trainer.test(
        model_wrapper,
        datamodule=data_module,
        ckpt_path=checkpoint_path,
    )


if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    main()
