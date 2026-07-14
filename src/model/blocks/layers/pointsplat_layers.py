import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from typing import Optional
import torch.nn.functional as F
from torch.autograd import Function
from torch.amp import custom_bwd, custom_fwd

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))
trunc_exp = _TruncExp.apply

class Tokenizer(nn.Module):
    def __init__(self, input_dim, output_dim, use_sinusoidal, use_ln, L=16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_sinusoidal = use_sinusoidal
        self.use_ln = use_ln

        if self.use_sinusoidal:
            self.L = L
            self.mlp_input_dim = 3 * self.L
            self.mlp = nn.Sequential(
                nn.Linear(self.mlp_input_dim, 1024),
                nn.SiLU(),
                nn.Linear(1024, 1024),
                nn.SiLU(),
                nn.Linear(1024, output_dim),
            )
        else:
            self.mlp_input_dim = input_dim
            self.mlp = nn.Linear(self.mlp_input_dim, output_dim)

        if self.use_ln:
            self.ln = nn.LayerNorm(self.mlp_input_dim, bias=False)

    def forward(self, x, cond=None):
        if self.use_sinusoidal:
            freqs = 2.0 ** torch.arange(self.L, dtype=torch.float32, device=x.device)  # [L]

            x_comp = x[..., 0].unsqueeze(-1)  # [N, 1]
            y_comp = x[..., 1].unsqueeze(-1)
            z_comp = x[..., 2].unsqueeze(-1)

            x_encoded = torch.sin(x_comp * freqs)
            y_encoded = torch.sin(y_comp * freqs)
            z_encoded = torch.sin(z_comp * freqs)

            encoded = torch.cat([x_encoded, y_encoded, z_encoded], dim=-1)  # [N, 3L]
        else:
            encoded = x
        if self.use_ln:
            encoded = self.ln(encoded)
        encoded = self.mlp(encoded)
        return encoded



class PatchEmbed3D(nn.Module):
    # input: tensor of shape (B, N, 64, 3+feat_dim), the last dim is xyz+feat, 64 is the number of point per patch

    def __init__(self, feature_dim=0, hidden_dim=48, point_per_patch=64, dim=128):
        super().__init__()
        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )

        self.register_buffer("basis", e)  # 3 x 16

        self.mlp = nn.Linear(hidden_dim + (3 + feature_dim) * point_per_patch, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # x.shape: (B, N, point_per_patch, 3+feat_dim)
        B, N, point_per_patch, c_input = x.shape
        x = x.view(B*N, point_per_patch, c_input)
        # 1. get the center c of the patch
        c = x[..., :3].mean(dim=1) # shape: (B*N, 3)
        # 2. encode the center position like pointembed
        projections = torch.einsum("bd,de->be", c, self.basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=-1) # shape: (B*N, hidden_dim)
        # 3. encode the patch
        patch3d = rearrange(x, 'b p c -> b (p c)') # shape: (B*N, point_per_patch*(3+feat_dim))
        encoded_x = torch.cat([embeddings, patch3d], dim=-1) # shape: (B*N, hidden_dim + point_per_patch*(3+feat_dim))
        # 4. pass through a mlp and then normalize
        if mask is not None:
            ...
        encoded_x = self.mlp(encoded_x)
        encoded_x = self.norm(encoded_x)
        encoded_x = encoded_x.view(B, N, -1)
        return encoded_x


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")
def inverse_sigmoid(x):

    if isinstance(x, float):
        x = torch.tensor(x).float()

    return torch.log(x / (1 - x))


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        layers = [
            self.make_linear(
                dim_in, n_neurons, is_first=True, is_last=False, bias=bias
            ),
            self.make_activation(activation),
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    n_neurons, n_neurons, is_first=False, is_last=False, bias=bias
                ),
                self.make_activation(activation),
            ]
        layers += [
            self.make_linear(
                n_neurons, dim_out, is_first=False, is_last=True, bias=bias
            )
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(output_activation)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

class LinerParameterTuner:
    def __init__(self, start, start_value, end_value, end):
        self.start = start
        self.start_value = start_value
        self.end_value = end_value
        self.end = end
        self.total_steps = self.end - self.start

    def get_value(self, step):
        if step < self.start:
            return self.start_value
        elif step > self.end:
            return self.end_value

        current_step = step - self.start

        ratio = current_step / self.total_steps

        current_value = self.start_value + ratio * (self.end_value - self.start_value)
        return current_value

class StaticParameterTuner:
    def __init__(self, v):
        self.v = v

    def get_value(self, step):
        return self.v

class GSLayer(nn.Module):
    """W/O Activation Function"""

    def setup_functions(self):

        self.scaling_activation = trunc_exp  # proposed by torch-ngp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.rgb_activation = torch.sigmoid

    def __init__(
        self,
        in_channels,
        use_rgb,
        clip_scaling=0.2,
        init_scaling=-5.0,
        init_density=0.1,
        sh_degree=None,
        xyz_offset=True,
        restrict_offset=True,
        xyz_offset_max_step=None,
        fix_opacity=False,
        fix_rotation=False,
        use_fine_feat=False,
        mlp_net_config = None
    ):
        super().__init__()
        self.setup_functions()

        self.mlp_net = MLP(in_channels, in_channels,
                           activation=mlp_net_config.activation,
                           n_hidden_layers=mlp_net_config.n_hidden_layers,
                           n_neurons=mlp_net_config.n_neurons)

        if isinstance(clip_scaling, (list,)):
            self.clip_scaling_pruner = LinerParameterTuner(*clip_scaling)
        else:
            self.clip_scaling_pruner = StaticParameterTuner(clip_scaling)
        self.clip_scaling = self.clip_scaling_pruner.get_value(0)

        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset = xyz_offset
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.use_fine_feat = use_fine_feat

        self.attr_dict = {
            "shs": (sh_degree + 1) ** 2 * 3,
            "scaling": 3,
            "xyz": 3,
            "opacity": None,
            "rotation": None,
        }
        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1

        self.attr_dict["rotation"] = 4

        self.out_layers = nn.ModuleDict()
        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs" and use_rgb:
                    out_ch = 3
                if key == "shs":
                    shs_out_ch = out_ch
                layer = nn.Linear(in_channels, out_ch)
            # initialize
            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            elif key == "xyz":
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            self.out_layers[key] = layer

        if self.use_fine_feat:
            fine_shs_layer = nn.Linear(in_channels, shs_out_ch)
            nn.init.constant_(fine_shs_layer.weight, 0)
            nn.init.constant_(fine_shs_layer.bias, 0)
            self.out_layers["fine_shs"] = fine_shs_layer

    def hyper_step(self, step):
        self.clip_scaling = self.clip_scaling_pruner.get_value(step)


    def _forward(self, x, pts, x_fine=None):
        assert len(x.shape) == 2
        x = self.mlp_net(x)

        ret = {}
        for k in self.attr_dict:
            layer = self.out_layers[k]

            v = layer(x)
            if k == "rotation":
                v = self.rotation_activation(v)
            elif k == "scaling":
                v = self.scaling_activation(v)
                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    v = self.opacity_activation(v)
            elif k == "shs":
                if self.use_rgb:
                    v = self.rgb_activation(v)

                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v_fine = torch.tanh(v_fine)
                        v = v + v_fine
                    v = torch.reshape(v, (v.shape[0], 3))
                else:
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v = v + v_fine
                    v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                if self.restrict_offset:
                    max_step = self.xyz_offset_max_step
                    v = (torch.sigmoid(v) - 0.5) * max_step
                if self.xyz_offset:
                    pass
                else:
                    assert NotImplementedError
                    v = v + pts
                k = "offset_xyz"
            ret[k] = v

        ret["use_rgb"] = self.use_rgb

        return ret

    def forward(self, x, pts, pts_rgb=None, mask=None):
        # x.shape [B, seq_len, patch_size, dim]
        # pts.shape [B, seq_len, patch_size, 3]
        # pts_rgb.shape [B, seq_len, patch_size, 3]
        # mask.shape [B, seq_len]
        B, seq_len, patch_size, _ = x.shape
        if mask is not None:
            mask = mask.unsqueeze(-1)
            mask = mask.expand(B, seq_len, patch_size).reshape(B, -1)
        else:
            mask = torch.ones((B, seq_len*patch_size), device=x.device, dtype=torch.bool)

        if self.use_rgb:
            if pts_rgb is None:
                color = torch.zeros(B, seq_len*patch_size, 3, device=x.device, dtype=torch.float32)
            else:
                color = pts_rgb.to(torch.float32).view(B, seq_len*patch_size, 3)
        else:
            color = torch.zeros(B, seq_len*patch_size, self.attr_dict['shs'] // 3, 3, device=x.device, dtype=torch.float32)
            if pts_rgb is not None:
                pts_sh_0 = ((pts_rgb - 0.5) / 0.282).view(B, seq_len*patch_size, 3)  # [B, seq_len, patch_size, 3]
                color[..., 0, :] += pts_sh_0

        gs_params = {
            "xyz": pts.view(B, seq_len*patch_size, 3).to(torch.float32),
            "color": color,
            "scale": torch.zeros(B, seq_len*patch_size, self.attr_dict['scaling'], device=x.device, dtype=torch.float32),
            "rotation": torch.zeros(B, seq_len*patch_size, self.attr_dict['rotation'], device=x.device, dtype=torch.float32),
            "opacity": 1e-6 * torch.ones(B, seq_len*patch_size, self.attr_dict['opacity'], device=x.device, dtype=torch.float32),
        }
        x = x.view(B, seq_len*patch_size, -1)
        pts = pts.view(B, seq_len*patch_size, 3)
        for b in range(B):
            indices = torch.arange(seq_len*patch_size, device=x.device)[mask[b]]
            if len(indices) == 0:
                continue

            ret = self._forward(x[b][indices], pts[b][indices])

            for key in ret:
                if isinstance(ret[key], torch.Tensor):
                    ret[key] = ret[key].to(torch.float32)

            gs_params["xyz"][b][indices]+=ret["offset_xyz"]
            gs_params["scale"][b][indices]=ret["scaling"]
            gs_params["rotation"][b][indices]=ret["rotation"]
            gs_params["opacity"][b][indices]=ret["opacity"]

            if pts_rgb is None:
                gs_params["color"][b][indices]=ret["shs"]
            else:
                gs_params["color"][b][indices]+=ret["shs"]
        return gs_params
