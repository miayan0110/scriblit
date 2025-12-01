#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask-driven relighting (NORMAL + DEPTH) — PyTorch CUDA
- Area lights from masks (white=emitter)
- Lambert N·L + distance falloff + optional depth-based soft shadows (ray marching)
- Multi-light, colored lights, percentile normalization, optional downscale for speed
- Uses GPU if available; falls back to CPU.

新增（做法 B）：
- 光源 z 取自「燈具 mask 的中心區塊（離邊界最遠）」的深度中位數；
  每個子光點各自取局部中心深度（更精準的面光源）。
- 若系統有 SciPy，使用距離變換 EDT；否則使用純 PyTorch 近似（反覆侵蝕計數，L∞ 距離）

pip install torch pillow numpy
（可選）pip install scipy

"""

import os
from os import PathLike
import io
import math
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_from_disk, Image as HfImage, Dataset as DS
from typing import Optional

# try SciPy EDT（有的話更精準）
try:
    from scipy.ndimage import distance_transform_edt as _edt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------------------- utils ----------------------

def as_pil(x, mode: Optional[str] = None) -> Image.Image:
    """
    將多種輸入統一轉為 PIL.Image：
      - 路徑字串 / PathLike
      - HuggingFace datasets.Image（多半有 .to_pil()）
      - dict：可能包含 'path' / 'bytes' / 'array' / 'image'
      - PIL.Image
      - numpy.ndarray（H×W×C 或 H×W）
      - torch.Tensor（H×W、1×H×W、3×H×W、或 H×W×3）
    參數:
      mode: 轉成 'RGB' 或 'L' 等（若為 None 則不轉）
    """
    img = None

    # 1) HF datasets 的 Image 物件（或其他帶 .to_pil() 的物件）
    if hasattr(x, "to_pil") and callable(getattr(x, "to_pil")):
        img = x.to_pil()

    # 2) dict（HF datasets 在 decode=False 或重新編碼時常見）
    elif isinstance(x, dict):
        if "image" in x and hasattr(x["image"], "to_pil"):
            img = x["image"].to_pil()
        elif "path" in x and isinstance(x["path"], (str, PathLike)):
            img = Image.open(x["path"])
        elif "bytes" in x:
            img = Image.open(io.BytesIO(x["bytes"]))
        elif "array" in x:
            arr = x["array"]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
            img = Image.fromarray(arr)
        else:
            raise TypeError(f"Unsupported HF dict keys: {list(x.keys())}")

    # 3) 已是 PIL
    elif isinstance(x, Image.Image):
        img = x

    # 4) 檔名/路徑
    elif isinstance(x, (str, bytes, PathLike)):
        img = Image.open(x)

    # 5) numpy array
    elif isinstance(x, np.ndarray):
        arr = x
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # 6) torch tensor
    elif isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.ndim == 2:  # [H,W]
            arr = t.numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
            img = Image.fromarray(arr)
        elif t.ndim == 3:  # [C,H,W] 或 [H,W,C]
            if t.shape[0] in (1, 3):   # [C,H,W] -> [H,W,C]
                t = t.permute(1, 2, 0)
            arr = t.numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
            img = Image.fromarray(arr)
        else:
            raise TypeError(f"Unsupported tensor shape: {tuple(t.shape)}")

    else:
        raise TypeError(f"Unsupported image type: {type(x)}")

    if mode is not None:
        img = img.convert(mode)
    return img

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """把 [C,H,W] (0–1) tensor 轉成 PIL.Image"""
    arr = (t.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def tensor_gray_to_pil(t: torch.Tensor) -> Image.Image:
    """把 [1,H,W] (0–1) tensor 轉成灰階 PIL.Image"""
    arr = (t[0].clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def to_tensor01(img: Image.Image, mode='rgb'):
    if mode == 'rgb':
        arr = np.asarray(img.convert('RGB')).astype(np.float32)/255.0
        t = torch.from_numpy(arr).permute(2,0,1)  # C,H,W
    elif mode == 'gray':
        arr = np.asarray(img.convert('L')).astype(np.float32)/255.0
        t = torch.from_numpy(arr)[None]  # 1,H,W
    else:
        raise ValueError('mode must be rgb or gray')
    return t

def load_normals(path_or_img, size=None, device='cpu'):
    img = as_pil(path_or_img, 'RGB')
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    t = to_tensor01(img, 'rgb').to(device)
    n = t*2.0 - 1.0
    n = nn.functional.normalize(n, dim=0, eps=1e-8)
    return n  # 3,H,W (unit)

def load_depth(path_or_img, size=None, device='cpu', robust=(1,99)):
    img = as_pil(path_or_img)
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    # to float
    if img.mode in ['RGB','RGBA']:
        img = img.convert('L')
    arr = np.asarray(img).astype(np.float32)
    p1, p99 = np.percentile(arr, robust)
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-8), 0, 1)
    t = torch.from_numpy(arr)[None].to(device)  # 1,H,W
    return t

def load_mask_bool(path_or_img, size=None, device='cpu'):
    img = as_pil(path_or_img, 'L')
    if size is not None:
        img = img.resize((size[1], size[0]), Image.NEAREST)
    arr = (np.asarray(img) > 127).astype(np.float32)
    t = torch.from_numpy(arr)[None].to(device)  # 1,H,W (0/1)
    return t

def percentile_normalize(I, mask_emitter, p=99.0):
    """I: 1xHxW, mask_emitter: 1xHxW (0/1). Exclude emitter for percentile."""
    env = I[mask_emitter<=0.0]
    if env.numel() == 0:
        scale = I.max().clamp(min=1e-8)
    else:
        # dtype 對齊，避免 quantile dtype mismatch
        q = torch.tensor(p/100.0, device=I.device, dtype=I.dtype)
        k = torch.quantile(env, q)
        scale = torch.maximum(k, torch.tensor(1e-8, device=I.device, dtype=I.dtype))
    In = torch.clamp(I/scale, 0.0, 1.0)
    return In

def composite_with_color(original, light_gray_or_rgb,
                         ambient=0.1, color=(1,1,1),
                         light_weight=1.0,
                         max_gain=None,
                         tone='none',
                         white_point=1.5):
    """
    original: 3xHxW
    light_gray_or_rgb: 1xHxW 或 3xHxW
    合成採用「獨立權重」：gain = ambient + light_weight * lightRGB
    - max_gain: 若設數值，會把 gain clamp 到 [0, max_gain]
    - tone='exposure': 先線性相加，再做曝光式 tone-mapping
    """
    # 建立彩色 light
    if light_gray_or_rgb.shape[0] == 1:
        c = torch.tensor(color, device=original.device, dtype=original.dtype).view(3,1,1)
        light_rgb = light_gray_or_rgb.expand(3,-1,-1) * c
    else:
        light_rgb = light_gray_or_rgb

    # 獨立權重混合（不互斥）
    gain_lin = ambient + light_weight * light_rgb  # 3xHxW

    if tone == 'exposure':
        gain = gain_lin / (1.0 + gain_lin / white_point)
    else:
        gain = gain_lin

    if max_gain is not None:
        gain = torch.clamp(gain, 0.0, max_gain)
    else:
        gain = torch.clamp(gain, 0.0, 10.0)  # 寬鬆上限避免炸掉

    return torch.clamp(original * gain, 0.0, 1.0)


# ---------------------- 發光方向（沿用原本） ----------------------

@torch.no_grad()
def infer_emission_from_depth(mask_bool, depth, normals=None,
                              n_dirs=96, radius_px=90, step_px=3,
                              down_bias=0.25, mix_wall_normal=0.35):
    """
    不依賴燈具形狀：在 mask 質心四周對 n_dirs 個影像平面方向做短距離 ray，
    比較是否「很快撞到近深度」→ 開闊度分數；把分數最高的方向當發光主方向。
    回傳: (emit_dir: 3,), emit_sharpness(float), is_isotropic(bool)
    """
    dev = depth.device
    H, W = depth.shape[-2], depth.shape[-1]
    m = mask_bool[0] > 0
    assert m.any(), "mask is empty"
    ys, xs = torch.nonzero(m, as_tuple=True)
    cy, cx = ys.float().mean(), xs.float().mean()
    # 起點深度用 mask 內中位數，較穩
    z0 = depth[0, ys, xs].median()

    # 影像平面方向（圓周，不分類形狀）
    thetas = torch.linspace(0, 2*math.pi, steps=n_dirs+1, device=dev)[:-1]
    dirs2 = torch.stack([thetas.sin(), thetas.cos()], dim=1)       # (n,2): y,x
    dirs2 = nn.functional.normalize(dirs2, dim=1, eps=1e-8)

    # 每方向做短距離 ray，累計「通行分數」
    scores = torch.zeros(n_dirs, device=dev)
    for i, d in enumerate(dirs2):
        free = 0.0
        for t in range(step_px, radius_px+1, step_px):
            y = int((cy + d[0]*t).round().clamp(0, H-1))
            x = int((cx + d[1]*t).round().clamp(0, W-1))
            z = depth[0, y, x]
            # 前方若「顯著更靠近相機」→ 視為遇到遮擋，停止
            if z < z0 - 0.03:  # 可依深度尺度調 0.02~0.06
                break
            free += 1.0
        scores[i] = free

    # 分數正規化
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # 各向同性：各方向差不多（方差很小）
    if scores.var() < 1e-4:
        return None, 1.0, True

    # 取最開闊的方向（影像平面）
    i_max = int(scores.argmax())
    d2 = dirs2[i_max]  # (2,)
    # 3D 方向：(y,x,z)；z 先 0，再混入牆外法線與「往下」偏置
    emit = torch.tensor([d2[0], d2[1], 0.0], device=dev)

    if normals is not None:
        n_wall = nn.functional.normalize(normals[:, m].mean(dim=1), dim=0, eps=1e-8)
    else:
        n_wall = torch.tensor([0.0, 0.0, 0.15], device=dev)  # 沒有 normals 時的微小外推

    down = torch.tensor([1.0, 0.0, 0.0], device=dev)  # y+ 視為往下（依你現有座標）
    emit_dir = nn.functional.normalize(
        0.65*emit + mix_wall_normal*n_wall + down_bias*down, dim=0, eps=1e-8
    )

    # sharpness 由集中度估計：差距越大越聚光
    peak = scores[i_max].item()
    mean = scores.mean().item()
    spread = float(max(1e-6, peak - mean))
    emit_sharpness = min(16.0, 2.0 + 20.0*spread)  # 約 2~16

    return emit_dir, emit_sharpness, False


def soft_clamp_inside(I, mask, radius=2):
    """把 emitter 區域稍微拉亮但不硬設 1"""
    kernel = torch.ones(1,1,2*radius+1,2*radius+1, device=I.device)
    m = (mask > 0).float()
    md = (F.conv2d(m.unsqueeze(0), kernel, padding=radius) > 0).float()[0]  # 1xHxW
    I = torch.maximum(I, 0.85*md)
    return I


# ---------------------- 中心區塊（紅區）深度：做法 B ----------------------

def _edt_inside_mask_torch(mask01: torch.Tensor, max_iter=512):
    """
    若無 SciPy，使用純 PyTorch 近似的 inside distance：
    反覆三乘三侵蝕，記下每個像素能撐過的輪數（相當於 L∞ 距離）。
    回傳：1xHxW 的 float dt（數值越大離邊界越遠）
    """
    m = (mask01 > 0).float()
    cur = m.clone()
    dt = torch.zeros_like(m)
    level = 0
    # erosion: min filter ~ max_pool on inverted
    while cur.sum() > 0 and level < max_iter:
        dt[cur > 0] = level
        inv = 1.0 - cur
        pooled = F.max_pool2d(inv.unsqueeze(0), 3, stride=1, padding=1)  # 3x3
        cur = (1.0 - pooled).squeeze(0)
        level += 1
    return dt

def build_center_core(mask01: torch.Tensor, keep_ratio=0.2) -> torch.Tensor:
    """
    取得「離邊界最遠的 top keep_ratio」作為中心區塊（紅區概念）
    回傳：1xHxW 的 0/1 tensor
    """
    dev = mask01.device
    m = (mask01[0] > 0).detach().cpu().numpy()
    if _HAS_SCIPY:
        dt = _edt(m.astype(np.uint8))   # SciPy 的內部距離
        dt = torch.from_numpy(dt).to(dev, dtype=mask01.dtype)[None]
    else:
        # PyTorch 近似
        dt = _edt_inside_mask_torch(mask01)

    # 只看 mask 內的 dt 分佈，取 top-p% 做 core
    dt_vals = dt[mask01 > 0]
    if dt_vals.numel() == 0:
        return (mask01 > 0).float()
    thr = torch.quantile(dt_vals, 1.0 - keep_ratio)
    core = ((dt >= thr) & (mask01 > 0)).float()
    return core

def local_center_depth(depth01: torch.Tensor,
                       core_mask01: torch.Tensor,
                       ly: float, lx: float, r: int = 6) -> torch.Tensor:
    """
    在 (ly,lx) 周圍 r 視窗內，取 core 區塊的深度中位數（沒有就回退全域 core）
    """
    H, W = depth01.shape[-2], depth01.shape[-1]
    y0, y1 = max(0, int(ly)-r), min(H, int(ly)+r+1)
    x0, x1 = max(0, int(lx)-r), min(W, int(lx)+r+1)
    core_local = core_mask01[0, y0:y1, x0:x1] > 0
    if core_local.any():
        return depth01[0, y0:y1, x0:x1][core_local].median()
    else:
        core_all = core_mask01[0] > 0
        if core_all.any():
            return depth01[0, core_all].median()
        return depth01[0].median()  # 最後保底


# ---------------------- core lighting ----------------------

def sample_points_from_mask(mask_small, ns):
    """mask_small: 1xHsxWs in {0,1}"""
    ys, xs = torch.nonzero(mask_small[0], as_tuple=True)
    if ys.numel() == 0:
        raise RuntimeError("Mask has no white pixels.")
    idx = torch.randint(0, ys.numel(), (ns,), device=mask_small.device)
    pts = torch.stack([ys[idx].float(), xs[idx].float()], dim=1)  # ns x 2 (y,x)
    return pts

def compute_light_irradiance(normals, depth, mask, nsamples=64, nsteps=64,
                             light_height_z=0.22, intensity=12.0, falloff='linear',
                             with_shadows=True, emit_dir=None, emit_sharpness=1.0,
                             # --- 新增：中心區塊深度（做法 B） ---
                             use_center_z=True,      # 開啟做法B
                             core_keep=0.2,          # 取離邊界最遠 top-p
                             local_core_r=6,         # 找局部 core 的半徑
                             center_bias=0.0):       # 小偏移（依深度定義）
    """
    normals: 3xHsxWs (unit)
    depth:   1xHsxWs in [0,1]
    mask:    1xHsxWs (0/1)
    Returns: 1xHsxWs irradiance (not normalized)
    """
    use_amp = (depth.device.type == 'cuda')
    with torch.amp.autocast('cuda', enabled=use_amp):
        _, Hs, Ws = depth.shape
        device = depth.device
        yy = torch.arange(Hs, device=device).view(Hs,1).expand(Hs,Ws).float()
        xx = torch.arange(Ws, device=device).view(1,Ws).expand(Hs,Ws).float()

        pts = sample_points_from_mask(mask, nsamples)  # nsx2
        acc = torch.zeros((Hs,Ws), device=device, dtype=depth.dtype)

        # 準備中心 core（離邊界最遠）
        if use_center_z:
            core = build_center_core(mask, keep_ratio=core_keep)
        else:
            core = None

        for i in range(nsamples):
            ly, lx = pts[i]

            # —— 每個子光點的 z_L：取就近中心區塊中位數（做法 B）
            if use_center_z and core is not None:
                z_src = local_center_depth(depth, core, float(ly), float(lx), r=local_core_r)
                z_src = z_src + depth.new_tensor(center_bias)
                Lz = z_src - depth[0]
            else:
                # 舊方法：全場固定高度（較粗）
                Lz = light_height_z - depth[0]

            Ly = ly - yy
            Lx = lx - xx
            dist = torch.sqrt(Ly*Ly + Lx*Lx + Lz*Lz + 1e-12)
            L = torch.stack([Ly/dist, Lx/dist, Lz/dist], dim=0)  # 3,Hs,Ws
            NdotL = torch.clamp((normals * L).sum(dim=0), 0.0, 1.0)

            if falloff == 'none':
                atten = 1.0
            elif falloff == 'linear':
                atten = 1.0 / (1e-3 + dist)
            else:
                atten = 1.0 / (1e-3 + dist*dist)

            if with_shadows:
                vis = torch.zeros_like(NdotL)
                mask_lit = NdotL > 1e-4
                if mask_lit.any():
                    ys = torch.where(mask_lit)[0].float()
                    xs = torch.where(mask_lit)[1].float()
                    zP = depth[0, mask_lit]  # M
                    ts = torch.linspace(0, 1, nsteps, device=device)  # S
                    ys_line = ly + (ys[:,None] - ly) * ts[None,:]     # MxS
                    xs_line = lx + (xs[:,None] - lx) * ts[None,:]
                    # 起點 z 用 z_src（或 light_height_z）
                    z0 = (z_src if (use_center_z and core is not None) else depth.new_tensor(light_height_z))
                    zs_line = z0 + (zP[:,None] - z0) * ts[None,:]

                    yi = torch.clamp(ys_line.round().long(), 0, Hs-1)
                    xi = torch.clamp(xs_line.round().long(), 0, Ws-1)
                    depth_along = depth[0, yi, xi]                   # MxS
                    blocked = (depth_along[:,3:] < (zs_line[:,3:] - 1e-3)).any(dim=1)
                    vis_vals = (~blocked).float()                    # M
                    vis[mask_lit] = vis_vals
            else:
                vis = 1.0

            # 發光方向加權（shape-agnostic）：抑制被遮擋半空間
            d_emit = -L  # 光→像素方向
            if emit_dir is not None:
                n_emit = nn.functional.normalize(
                    emit_dir.to(normals.device).to(normals.dtype), dim=0, eps=1e-8
                )  # (3,)
                emit_cos = torch.clamp((n_emit.view(3,1,1) * d_emit).sum(dim=0), 0.0, 1.0)
                emit_weight = emit_cos ** emit_sharpness
            else:
                emit_weight = 1.0

            acc += intensity * NdotL * atten * vis * emit_weight

    return acc[None] / max(1, nsamples)

def compute_lightmap_from_mask_torch(normal_path, depth_path, mask_path,
                                     scale=0.5, nsamples=64, nsteps=64,
                                     light_height_z=0.22, intensity=12.0,
                                     falloff='linear', with_shadows=True,
                                     clamp_inside=True, percentile=99.0,
                                     device=None,
                                     auto_emission=True,
                                     n_dirs=96, radius_px=90, step_px=3,
                                     down_bias=0.25, mix_wall_normal=0.35,
                                     # —— 新增：中心區塊深度（做法 B）的參數
                                     use_center_z=True,
                                     core_keep=0.2,
                                     local_core_r=6,
                                     center_bias=0.0):
    # Decide device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # full size from mask
    full_mask_img = as_pil(mask_path, 'L')
    Hf, Wf = full_mask_img.size[1], full_mask_img.size[0]
    Hs, Ws = max(1,int(Hf*scale)), max(1,int(Wf*scale))

    normals = load_normals(normal_path, size=(Hs,Ws), device=device)
    depth   = load_depth(depth_path,  size=(Hs,Ws), device=device)
    mask    = load_mask_bool(mask_path, size=(Hs,Ws), device=device)

    # 自動推發光方向／各向同性
    if auto_emission:
        emit_dir, emit_sharp, is_iso = infer_emission_from_depth(
            mask, depth, normals=normals,
            n_dirs=n_dirs, radius_px=radius_px, step_px=step_px,
            down_bias=down_bias, mix_wall_normal=mix_wall_normal
        )
        if is_iso:
            emit_dir, emit_sharp = None, 1.0
    else:
        emit_dir, emit_sharp = None, 1.0

    with torch.no_grad():
        I_small = compute_light_irradiance(
            normals, depth, mask,
            nsamples=nsamples, nsteps=nsteps,
            light_height_z=light_height_z, intensity=intensity,
            falloff=falloff, with_shadows=with_shadows,
            emit_dir=emit_dir, emit_sharpness=emit_sharp,
            # —— 做法 B 參數傳入
            use_center_z=use_center_z,
            core_keep=core_keep,
            local_core_r=local_core_r,
            center_bias=center_bias
        )

        # percentile normalization on environment
        I_small = percentile_normalize(I_small, mask, p=percentile)
        if clamp_inside:
            I_small = soft_clamp_inside(I_small, mask, radius=2)

        # upsample to full res
        I_full = nn.functional.interpolate(I_small.unsqueeze(0), size=(Hf,Wf), mode='bilinear', align_corners=False).squeeze(0)

    return I_full  # 1xHf xWf


# ---------------------- 高階 API（維持你的介面） ----------------------

class PhysicalRelightingConfig:
    def __init__(self, ori, normal, depth):
        # 基本輸入
        self.normal = normal
        self.depth = depth
        self.original = ori
        self.mask = []   # [(path, (r,g,b), gain)]

        # 光照計算
        self.scale = 1.0
        self.nsamples = 64
        self.nsteps = 64
        self.height = 0.22
        self.intensity = 12.0
        self.falloff = 'linear'   # ['none','linear','inverse_square']
        self.with_shadows = False
        self.no_clamp = False
        self.percentile = 99.0

        # 環境光 / 合成
        self.ambient = 0.75
        self.light_weight = 1.0
        self.max_gain = None
        self.tone = 'none'        # ['none','exposure']
        self.white_point = 1.5

        # 光源自動方向
        self.auto_emission = False
        self.emit_dirs = 96
        self.emit_radius = 90
        self.emit_step = 3
        self.emit_down_bias = 0.25
        self.emit_mix_normal = 0.35

        # —— 中心區塊（做法 B）參數
        self.use_center_z = True
        self.core_keep = 0.2
        self.local_core_r = 6
        self.center_bias = 0.0

        # # 輸出檔名
        # self.out_light = 'callable_lightmap_v2.png'
        # self.out_relit = 'callable_relit_v2.png'

    def add_mask(self, path_or_img, color=(1.0,1.0,1.0), gain=1.0):
        """模仿 argparse 的 --mask path_or_img:r,g,b@gain"""
        self.mask.append((path_or_img, color, gain))


def compute_relighting(phys_cfg: PhysicalRelightingConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('Device:', device)

    # load original
    orig_img = as_pil(phys_cfg.original, 'RGB')
    orig = to_tensor01(orig_img,'rgb').to(device)

    # accumulate colored lights
    Hf, Wf = orig.shape[1], orig.shape[2]
    light_rgb = torch.zeros((3,Hf,Wf), device=device)

    for (mask_path, color, gain) in phys_cfg.mask:
        I = compute_lightmap_from_mask_torch(
            phys_cfg.normal, phys_cfg.depth, mask_path,
            scale=phys_cfg.scale, nsamples=phys_cfg.nsamples, nsteps=phys_cfg.nsteps,
            light_height_z=phys_cfg.height, intensity=phys_cfg.intensity,
            falloff=phys_cfg.falloff, with_shadows=phys_cfg.with_shadows,
            clamp_inside=(not phys_cfg.no_clamp), percentile=phys_cfg.percentile,
            device=device, auto_emission=phys_cfg.auto_emission,
            n_dirs=phys_cfg.emit_dirs, radius_px=phys_cfg.emit_radius, step_px=phys_cfg.emit_step,
            down_bias=phys_cfg.emit_down_bias, mix_wall_normal=phys_cfg.emit_mix_normal,
            # —— 做法 B 參數
            use_center_z=phys_cfg.use_center_z,
            core_keep=phys_cfg.core_keep,
            local_core_r=phys_cfg.local_core_r,
            center_bias=phys_cfg.center_bias
        )  # 1xHxW

        c = torch.tensor(color, device=device).view(3,1,1)
        light_rgb += (I * gain).expand(3,-1,-1) * c

    light_rgb = light_rgb.clamp(0,1)
    light_gray = light_rgb.max(dim=0, keepdim=True).values  # 方便存灰階 lightmap

    # composite
    relit = composite_with_color(
        orig, light_rgb,
        ambient=phys_cfg.ambient,
        light_weight=phys_cfg.light_weight,
        max_gain=phys_cfg.max_gain,
        tone=phys_cfg.tone,
        white_point=phys_cfg.white_point
    )

    # return images instead of saving
    lightmap = tensor_gray_to_pil(light_gray)  # 灰階光圖
    i_relit = tensor_to_pil(relit)            # 彩色 relit 結果
    return i_relit, lightmap

# if __name__ == '__main__':
#     ds: DS = load_from_disk('/mnt/HDD7/miayan/paper/relighting_datasets/lsun_train_cap')
#     for c in ds.column_names:
#         ds = ds.cast_column(c, HfImage(decode=True))

#     ori = ds[0]['image']
#     normal = ds[0]['normal']
#     depth = ds[0]['depth']
#     phys_cfg = PhysicalRelightingConfig(ori=ori, normal=normal, depth=depth)
#     phys_cfg.add_mask(ds[0]['mask'], (1,0,0), 1)
#     phys, lightmap = compute_relighting(phys_cfg)
#     phys.save(phys_cfg.out_relit)
#     lightmap.save(phys_cfg.out_light)