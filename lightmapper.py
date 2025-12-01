# lightmapper.py
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import torch
import torch.nn as nn

# --- ① 核心網路：單純 nn.Module，負責 forward ---
class LightmapperCore(nn.Module):
    """
    Input : (B, 9, H, W) = [mask(3), normal(3), depth(3)]
    Output: (B, 3, H, W) = lightmap in [0,1] (sigmoid)
    """
    def __init__(self, in_ch=9, out_ch=3, base_ch=64):
        super().__init__()
        c = base_ch
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),     nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c, c*2, 3, padding=1),   nn.SiLU(),
            nn.Conv2d(c*2, c*2, 3, padding=1), nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c*2, c, 2, stride=2), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),           nn.SiLU(),
            nn.ConvTranspose2d(c, c, 2, stride=2),   nn.SiLU(),
            nn.Conv2d(c, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 9, f"LightmapperCore expects input (B,9,H,W), got {tuple(x.shape)}"
        z = self.enc(x)
        y = self.dec(z)
        return torch.sigmoid(y)  # 和 Indoor_dataset 的 conditioning 範圍對齊 [0,1]


# --- ② Diffusers 包裝器：負責 config 與 save/load ---
class Lightmapper(ModelMixin, ConfigMixin):
    """
    Diffusers 友好的外層：持有一個 core(nn.Module)，並支援
    - save_pretrained / from_pretrained
    - config 自動保存
    """
    @register_to_config
    def __init__(self, in_ch=9, out_ch=3, base_ch=64):
        super().__init__()
        self.core = LightmapperCore(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)
