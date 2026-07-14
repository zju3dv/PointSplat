from typing import Optional, Union

from .encoder import Encoder, EncoderOutput
from .encoder_pointsplat import EncoderPointsplat, EncoderPointsplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "pointsplat": (EncoderPointsplat, None)
}

EncoderCfg = EncoderPointsplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
