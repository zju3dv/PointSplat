from ...dataset import DatasetCfg
from .decoder import Decoder
from .decoder_splatting_gsplat import DecoderSplattingGSplat, DecoderSplattingGSplatCfg

DECODERS = {
    "splatting_gsplat": DecoderSplattingGSplat
}

DecoderCfg = DecoderSplattingGSplatCfg


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg)
