import torch

from .typing_utils import *


@dataclass
class CheckpointingCfg:
    load: Optional[str]
    pretrained_model: Optional[str]
    pretrained_depth: Optional[str]
    pretrained_encoder: Optional[str]
    pretrained_decoder: Optional[str]
    no_strict_load: bool


def _unwrap_state_dict(checkpoint):
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "network" in checkpoint:
        return checkpoint["network"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _format_status(name, status) -> str:
    return (
        f"=== {name} loaded result ===\n"
        f"Missing keys: {status.missing_keys if status.missing_keys else 'No missing keys'}\n"
        f"Unexpected keys: {status.unexpected_keys if status.unexpected_keys else 'No unexpected keys'}\n"
    )


def load_pretrained_model(
    model_wrapper,
    checkpointing_cfg: CheckpointingCfg,
) -> str:
    strict_load = not checkpointing_cfg.no_strict_load
    results = []

    if checkpointing_cfg.pretrained_model is not None:
        pretrained_model = _unwrap_state_dict(
            torch.load(checkpointing_cfg.pretrained_model, map_location="cpu")
        )
        status = model_wrapper.load_state_dict(pretrained_model, strict=True)
        results.append(_format_status("pretrained_model", status))

    if checkpointing_cfg.pretrained_encoder is not None:
        pretrained_encoder = _unwrap_state_dict(
            torch.load(checkpointing_cfg.pretrained_encoder, map_location="cpu")
        )
        status = model_wrapper.encoder.load_state_dict(pretrained_encoder, strict=False)
        results.append(_format_status("pretrained_encoder", status))

    if checkpointing_cfg.pretrained_depth is not None:
        pretrained_depth = _unwrap_state_dict(
            torch.load(checkpointing_cfg.pretrained_depth, map_location="cpu")
        )
        status = model_wrapper.encoder.depth_predictor.load_state_dict(
            pretrained_depth,
            strict=strict_load,
        )
        results.append(_format_status("pretrained_depth", status))

    if checkpointing_cfg.pretrained_decoder is not None:
        pretrained_decoder = _unwrap_state_dict(
            torch.load(checkpointing_cfg.pretrained_decoder, map_location="cpu")
        )
        status = model_wrapper.decoder.load_state_dict(
            pretrained_decoder,
            strict=False,
        )
        results.append(_format_status("pretrained_decoder", status))

    return "\n".join(results)
