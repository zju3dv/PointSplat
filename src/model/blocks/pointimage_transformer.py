import torch
import torch.nn as nn
from .qknorm_transformer import QK_Norm_TransformerBlock

class ImageAABackbone(nn.Module):
    def __init__(self, dim: int, head_dim: int, use_qk_norm: bool):
        super().__init__()
        self.frame_attention = QK_Norm_TransformerBlock(
            dim=dim,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
        )
        self.global_attention = QK_Norm_TransformerBlock(
            dim=dim,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
        )
    def forward(self, x, condition_length_list):
        # condition_length_list: tensor [B, V]
        batch_size, v_num = condition_length_list.shape

        # 1. global attention
        x = self.global_attention(x)

        # 2. frame attention
        frame_intermediates = []
        for b in range(batch_size):
            frame_intermediates_b = []
            start_idx = 0
            for v in range(v_num):
                token_len = condition_length_list[b, v]
                frame_tokens_v = self.frame_attention(x[b, start_idx:start_idx+token_len].unsqueeze(0))
                frame_intermediates_b.append(frame_tokens_v)
                start_idx += token_len
            frame_intermediates_b = torch.cat(frame_intermediates_b, dim=1) # [1, N, C]
            frame_intermediates.append(frame_intermediates_b)
        frame_tokens = torch.cat(frame_intermediates, dim=0) # [B, N, C]

        return frame_tokens


class PointImageMMJointTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        point_only: bool,
    ):
        super().__init__()
        num_attention_heads = num_heads
        attention_head_dim = dim // num_attention_heads
        assert attention_head_dim * num_attention_heads == dim

        self.point_only = point_only

        self.point_backbone = QK_Norm_TransformerBlock(
            dim=dim,
            head_dim=attention_head_dim,
            use_qk_norm=True,
        )

        if not self.point_only:
            self.image_backbone = ImageAABackbone(
                dim=dim,
                head_dim=attention_head_dim,
                use_qk_norm=True,
            )

        self.global_backbone = QK_Norm_TransformerBlock(
            dim=dim,
            head_dim=attention_head_dim,
            use_qk_norm=True,
        )

    def forward(self, x, **kwargs):
        assert kwargs.get('condition', None) is not None
        assert kwargs.get('condition_length_list', None) is not None
        q_len = x.shape[1]
        x = torch.cat([x, kwargs['condition']], dim=1)
        x = self.global_backbone(x)
        point_tokens = x[:, :q_len]
        cond_tokens = x[:, q_len:]
        point_tokens = self.point_backbone(point_tokens)
        if not self.point_only:
            cond_tokens = self.image_backbone(cond_tokens, kwargs['condition_length_list'])
        kwargs['condition'] = cond_tokens
        return point_tokens, kwargs