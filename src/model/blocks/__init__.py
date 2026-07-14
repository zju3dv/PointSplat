import torch
import torch.nn as nn

from .qknorm_transformer import QK_Norm_TransformerBlock, MultiHeadCrossAttention
from .pointimage_transformer import PointImageMMJointTransformerBlock

def init_weights(module, std=0.02):
    """Initialize weights for linear and embedding layers.

    Args:
        module: Module to initialize
        std: Standard deviation for normal initialization
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
