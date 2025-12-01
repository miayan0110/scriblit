#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask-driven relighting (NORMAL + DEPTH) — PyTorch CUDA
- Area lights from masks (white=emitter)
- Lambert N·L + distance falloff + optional depth-based soft shadows (ray marching)
- Multi-light, colored lights, percentile normalization, optional downscale for speed
- Uses GPU if available; falls back to CPU.

pip install torch pillow numpy

Example:
python relight_mask_torch.py \
  --normal _frame.0000.normal_cam.png \
  --depth  _frame.0000.depth_meters.png \
  --original _frame.jpg \
  --mask _frame_mask.png:1.0,0.95,0.8 \
  --scale 1.0 --nsamples 128 --nsteps 96 \
  --height 0.22 --intensity 12 --falloff linear \
  --ambient 0.08 --percentile 99 \
  --with-shadows
"""

import argparse, math
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

# ---------------------- utils ----------------------

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

def load_normals(path, size=None, device='cpu'):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    t = to_tensor01(img, 'rgb').to(device)
    n = t*2.0 - 1.0
    n = nn.functional.normalize(n, dim=0, eps=1e-8)
    return n  # 3,H,W (unit)

def load_depth(path, size=None, device='cpu', robust=(1,99)):
    img = Image.open(path)
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

def load_mask_bool(path, size=None, device='cpu'):
    img = Image.open(path).convert('L')
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
        q = torch.tensor(p/100.0, device=I.device, dtype=I.dtype)  # ★ dtype 對齊
        k = torch.quantile(env, q)
        scale = torch.maximum(k, torch.tensor(1e-8, device=I.device))
    In = torch.clamp(I/scale, 0.0, 1.0)
    return In

def composite_with_color(original, light_gray_or_rgb,
                         ambient=0.1, color=(1,1,1),
                         light_weight=1.0,           # ★ 新：燈光獨立權重
                         max_gain=None,              # ★ 新：上限(避免過曝)，None=不限制
                         tone='none',                # ★ 新：'none' 或 'exposure'
                         white_point=1.5):           # ★ 新：曝光式白點
    """
    original: 3xHxW
    light_gray_or_rgb: 1xHxW 或 3xHxW
    合成採用「獨立權重」：gain = ambient + light_weight * lightRGB
    - max_gain: 若設數值，會把 gain clamp 到 [0, max_gain]
    - tone='exposure': 先線性相加，再做簡單曝光式 tone-mapping
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
        # 簡單曝光曲線：避免無限增亮造成過曝
        gain = gain_lin / (1.0 + gain_lin / white_point)
    else:
        gain = gain_lin

    if max_gain is not None:
        gain = torch.clamp(gain, 0.0, max_gain)  # 可選上限
    else:
        gain = torch.clamp(gain, 0.0, 10.0)       # 給個寬鬆的數值防炸

    return torch.clamp(original * gain, 0.0, 1.0)

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

def soft_clamp_inside(I, mask, radius=3):
    # mask 先做小膨脹，再把 I 限制到至少某個明亮值（但不是硬設 1）
    import torch.nn.functional as F
    kernel = torch.ones(1,1,2*radius+1,2*radius+1, device=I.device)
    m = (mask > 0).float()
    md = (F.conv2d(m.unsqueeze(0), kernel, padding=radius) > 0).float()[0]  # 1xHxW
    I = torch.maximum(I, 0.85*md)   # 或者  clamp 到 ≥0.85
    return I

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
                             with_shadows=True, emit_dir=None, emit_sharpness=1.0):
    """
    normals: 3xHsxWs (unit)
    depth:   1xHsxWs in [0,1]
    mask:    1xHsxWs (0/1)
    Returns: 1xHsxWs irradiance (not normalized)
    """
    use_amp = (depth.device.type == 'cuda')
    with torch.amp.autocast('cuda', enabled=use_amp):  # ★ 新寫法
        _, Hs, Ws = depth.shape
        device = depth.device
        yy = torch.arange(Hs, device=device).view(Hs,1).expand(Hs,Ws).float()
        xx = torch.arange(Ws, device=device).view(1,Ws).expand(Hs,Ws).float()

        pts = sample_points_from_mask(mask, nsamples)  # nsx2
        acc = torch.zeros((Hs,Ws), device=device, dtype=depth.dtype)

        for i in range(nsamples):
            ly, lx = pts[i]
            Lz = light_height_z - depth[0]       # Hs,Ws
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
                # Ray marching; vectorize by marching same steps for all pixels above threshold
                vis = torch.zeros_like(NdotL)
                mask_lit = NdotL > 1e-4
                if mask_lit.any():
                    ys = torch.where(mask_lit)[0].float()
                    xs = torch.where(mask_lit)[1].float()
                    zP = depth[0, mask_lit]  # M
                    # param t [0,1]
                    ts = torch.linspace(0, 1, nsteps, device=device)  # S
                    ys_line = ly + (ys[:,None] - ly) * ts[None,:]     # MxS
                    xs_line = lx + (xs[:,None] - lx) * ts[None,:]
                    zs_line = light_height_z + (zP[:,None] - light_height_z) * ts[None,:]

                    yi = torch.clamp(ys_line.round().long(), 0, Hs-1)
                    xi = torch.clamp(xs_line.round().long(), 0, Ws-1)
                    depth_along = depth[0, yi, xi]                   # MxS
                    blocked = (depth_along[:,3:] < (zs_line[:,3:] - 1e-3)).any(dim=1)
                    vis_vals = (~blocked).float()                    # M
                    vis[mask_lit] = vis_vals
            else:
                vis = 1.0

            # ★ 發光方向加權（shape-agnostic）：抑制被遮擋半空間
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
                                     down_bias=0.25, mix_wall_normal=0.35):
    # Decide device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # full size from mask
    full_mask_img = Image.open(mask_path).convert('L')
    Hf, Wf = full_mask_img.size[1], full_mask_img.size[0]
    Hs, Ws = max(1,int(Hf*scale)), max(1,int(Wf*scale))

    normals = load_normals(normal_path, size=(Hs,Ws), device=device)
    depth   = load_depth(depth_path,  size=(Hs,Ws), device=device)
    mask    = load_mask_bool(mask_path, size=(Hs,Ws), device=device)

    # ★ 自動推發光方向／各向同性
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
            emit_dir=emit_dir, emit_sharpness=emit_sharp
        )

        # percentile normalization on environment
        I_small = percentile_normalize(I_small, mask, p=percentile)
        if clamp_inside:
            I_small = soft_clamp_inside(I_small, mask, radius=2)

        # upsample to full res
        I_full = nn.functional.interpolate(I_small.unsqueeze(0), size=(Hf,Wf), mode='bilinear', align_corners=False).squeeze(0)

    return I_full  # 1xHf xWf

# ---------------------- CLI ----------------------

def parse_mask_arg(s):
    # 支援 path[:r,g,b][@gain]
    col = (1.0,1.0,1.0); gain = 1.0
    if '@' in s: s, gain_str = s.split('@',1); gain = float(gain_str)
    if ':' in s:
        s, c = s.split(':',1)
        col = tuple(float(x) for x in c.split(','))
    return s, col, gain

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", type=str, default='imnormal_1.png')
    ap.add_argument("--depth", type=str, default='im_1_depth.png')
    ap.add_argument("--original", type=str, default='im_1.png')
    ap.add_argument('--mask', action='append', type=parse_mask_arg, required=True)
    ap.add_argument('--scale', type=float, default=1)   # 計算光線時所小的程度，縮小一點可以算得比較快
    ap.add_argument('--nsamples', type=int, default=64)
    ap.add_argument('--nsteps', type=int, default=64)
    ap.add_argument('--height', type=float, default=0.22)
    ap.add_argument('--intensity', type=float, default=12.0)
    ap.add_argument('--falloff', type=str, default='linear', choices=['none','linear','inverse_square'])
    ap.add_argument('--with-shadows', action='store_true')
    ap.add_argument('--no-clamp', action='store_true')
    ap.add_argument('--percentile', type=float, default=99.0)
    ap.add_argument('--ambient', type=float, default=0.75)   # 環境光
    ap.add_argument('--out-light', default='callable_lightmap.png')
    ap.add_argument('--out-relit', default='callable_relit.png')

    # 自動判斷光線應該朝哪個方向擴散
    ap.add_argument('--auto-emission', action='store_true', help='從 depth/normal 自動推發光方向')
    ap.add_argument('--emit-dirs', type=int, default=96)
    ap.add_argument('--emit-radius', type=int, default=90)
    ap.add_argument('--emit-step', type=int, default=3)
    ap.add_argument('--emit-down-bias', type=float, default=0.25)
    ap.add_argument('--emit-mix-normal', type=float, default=0.35)

    # 環境光與燈光獨立
    ap.add_argument('--light-weight', type=float, default=1, help='獨立的燈光權重(增益)，與 ambient 不互斥')
    ap.add_argument('--max-gain', type=float, default=None, help='可選：總增益上限(避免過曝)。例如 2.0；留空表示不限制')
    ap.add_argument('--tone', type=str, default='none', choices=['none', 'exposure'], help='可選：none=線性；exposure=曝光式 tone-mapping')
    ap.add_argument('--white-point', type=float, default=1.5, help='tone=exposure 時的白點，控制何時開始壓亮區')

    args = ap.parse_args()

    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    # load original
    orig_img = Image.open(args.original).convert('RGB')
    orig = to_tensor01(orig_img,'rgb').to(device)

    # accumulate colored lights
    Hf, Wf = orig.shape[1], orig.shape[2]
    light_rgb = torch.zeros((3,Hf,Wf), device=device)

    for (mask_path, color, gain) in args.mask:
        I = compute_lightmap_from_mask_torch(
            args.normal, args.depth, mask_path,
            scale=args.scale, nsamples=args.nsamples, nsteps=args.nsteps,
            light_height_z=args.height, intensity=args.intensity,
            falloff=args.falloff, with_shadows=args.with_shadows,
            clamp_inside=(not args.no_clamp), percentile=args.percentile,
            device=device, auto_emission=args.auto_emission,
            n_dirs=args.emit_dirs, radius_px=args.emit_radius, step_px=args.emit_step,
            down_bias=args.emit_down_bias, mix_wall_normal=args.emit_mix_normal
        )  # 1xHxW
        c = torch.tensor(color, device=device).view(3,1,1)
        light_rgb += (I * gain).expand(3,-1,-1) * c

    light_rgb = light_rgb.clamp(0,1)
    light_gray = light_rgb.max(dim=0, keepdim=True).values  # for convenience

    # composite
    relit = composite_with_color(
        orig, light_rgb,
        ambient=args.ambient,
        light_weight=args.light_weight,
        max_gain=args.max_gain,
        tone=args.tone,
        white_point=args.white_point
    )

    # save
    to_img = lambda t: Image.fromarray((t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
    Image.fromarray((light_gray[0].cpu().numpy()*255).astype(np.uint8)).save(args.out_light)
    to_img(relit).save(args.out_relit)
    print('Saved:', args.out_light, args.out_relit)

if __name__ == '__main__':
    main()