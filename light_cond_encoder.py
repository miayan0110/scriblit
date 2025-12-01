import math
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

class LightCondEncoder(nn.Module):
    def __init__(self, cross_dim:int, n_tokens:int=4, fourier_B_intensity:int=0, fourier_B_color:int=4):
        super().__init__()
        self.K = n_tokens
        self.Bi = fourier_B_intensity
        self.Bc = fourier_B_color
        self.cross_dim = cross_dim

        # 計算輸入維度：i(1) + c(3) + Fi + Fc + 交互項(i*c)
        in_dim = 1 + 3 + (2*self.Bi*1) + (2*self.Bc*3) + 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(),
            nn.Linear(256, 256),    nn.SiLU(),
            nn.Linear(256, n_tokens * cross_dim)
        )
        # CFG 用的「無條件」學習 token（K×D）
        self.null_ctx = nn.Parameter(torch.zeros(n_tokens, cross_dim))

    def _fourier(self, x: torch.Tensor, B: int) -> torch.Tensor:
        # x: (N,D) in [0,1]; B=0 -> 空張量
        if B <= 0:
            return torch.empty(x.size(0), 0, device=x.device, dtype=x.dtype)
        k = torch.arange(B, device=x.device, dtype=x.dtype)
        f = (2.0**k) * math.pi
        x = x.unsqueeze(-1) * f               # (N,D,B)
        sin, cos = torch.sin(x).flatten(1), torch.cos(x).flatten(1)
        return torch.cat([sin, cos], dim=-1)  # (N, 2*D*B)

    def forward(self, intensity: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
        """
        intensity: (N,1) in [0,1]
        color:     (N,3) in [0,1]
        return:    (N, K, D)  --> 給 UNet/ControlNet 的 encoder_hidden_states
        """
        # 顏色只取方向（幅度交給 intensity）
        c = color / (color.norm(dim=-1, keepdim=True) + 1e-8)
        feats = [intensity, c,
                 self._fourier(intensity, self.Bi),
                 self._fourier(c,         self.Bc),
                 intensity * c]                      # 交互項
        x = torch.cat([t for t in feats if t.numel() > 0], dim=-1)  # (N, in_dim)
        y = self.mlp(x)                                           # (N, K*D)
        return y.view(x.size(0), self.K, self.cross_dim)          # (N, K, D)

class CustomEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, cross_dim: int, K: int = 4, Bi: int = 6, Bc: int = 6):
        super().__init__()
        self.enc = LightCondEncoder(cross_dim, K, Bi, Bc)
        self.D = cross_dim

    def forward(self, intensity, color, pad_to_77: bool = False):
        cond = self.enc(intensity, color)  # (B, K, D)
        if pad_to_77:
            B, K, D = cond.shape
            if K < 77:
                # null_ctx: (D,) → (1,1,D) → (B, 77-K, D)
                null = self.enc.null_ctx.view(1, 1, D).to(cond)
                pad_blk = null.expand(B, 77 - K, D)
                cond = torch.cat([cond, pad_blk], dim=1)      # → (B, 77, D)
            elif K > 77:
                cond = cond[:, :77, :]                        # 視需求也可 raise
        return cond
