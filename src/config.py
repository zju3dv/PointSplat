
from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import TestCfg
from .utils.ckpt_utils import CheckpointingCfg
from .utils.typing_utils import *


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    seed: int
    num_nodes: int
    precision: Literal["64", "64-true", "32", "32-true", "16", "16-mixed", "bf16", "bf16-mixed"]
    use_plugins: bool



@dataclass
class RootCfg:
    mode: Literal["test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    test: TestCfg


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
    )
