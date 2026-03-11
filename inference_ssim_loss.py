import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import heapq
import csv
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from huggingface_hub import snapshot_download
from natsort import natsorted
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim
from skimage import color as skcolor
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont

# Custom Imports
from network_controlnet import ControlNetModel
from datasets import load_dataset, Image as HfImage, Dataset
import pyiqa
niqe_metric = pyiqa.create_metric('niqe', device=torch.device('cuda'))
brisque_metric = pyiqa.create_metric('brisque', device=torch.device('cuda'))
lpips_metric = pyiqa.create_metric('lpips', device=torch.device('cuda'))

# Relighting API Imports
import physical_relighting_api_v2 as relighting_api_v2
import physical_relighting_api_v2_gemini as relighting_api_gemini
import physical_relighting_api_v2_gemini_amb as relighting_api_gemini_amb

# Encoders
from light_cond_encoder import CustomEncoder as CustomEncoderV1
from light_cond_encoder_amb import CustomEncoder as CustomEncoderAmb
from albedo_estimator import AlbedoWrapper


# ==========================================
# [New Class] Control Metrics (IM & CA)
# 說明：用於量化強度與顏色控制的準確性
# ==========================================
class RelightingControlMetrics:
    def __init__(self, eps=1e-8, resolution=512):
        self.eps = eps
        self.luma_weights = torch.tensor([0.2126, 0.7152, 0.0722])
        self.resizer = transforms.Resize((resolution, resolution))

    def get_weighted_average(self, img_tensor, weight_mask=None):
        """
        If weight_mask is None: return global average.
        Else: return weighted average (original behavior).
        """
        if weight_mask is None:
            # img_tensor can be (H,W) or (C,H,W)
            return img_tensor.mean(dim=(-2, -1)) if img_tensor.ndim == 3 else img_tensor.mean()

        device = img_tensor.device
        mask = weight_mask.to(device)

        # 解析度對齊：確保 mask 與 img_tensor 尺寸一致
        if img_tensor.shape[-2:] != mask.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=img_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze()

        if img_tensor.ndim == 3:
            mask = mask.unsqueeze(0) if mask.ndim == 2 else mask

        weighted_sum = torch.sum(img_tensor * mask, dim=(-2, -1))
        weight_total = torch.sum(mask) + self.eps
        return weighted_sum / weight_total

    def get_chromaticity_vector(self, img_pil, weight_mask=None):
         # 確保輸入影像解析度一致（沿用你原本 resizer）
        img_resized = self.resizer(img_pil)

        # 轉 numpy RGB [0,1]
        rgb = np.asarray(img_resized).astype(np.float32) / 255.0  # (H,W,3), RGB
        # 轉 Lab
        lab = skcolor.rgb2lab(rgb)  # (H,W,3), L in [0,100], a,b ~ [-128,127]
        ab = lab[:, :, 1:3]         # (H,W,2)

        # 轉 torch tensor
        device = weight_mask.device if weight_mask is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ab_t = torch.from_numpy(ab).to(device)  # (H,W,2)

        # 若不用 mask：整張圖平均
        if weight_mask is None:
            mean_ab = ab_t.mean(dim=(0, 1))  # (2,)
            return mean_ab

        # 若有 mask：做加權平均（保留你原本能力）
        mask = weight_mask.to(device)
        # 對齊尺寸
        if mask.shape[-2:] != ab_t.shape[:2]:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=ab_t.shape[:2],
                mode='bilinear',
                align_corners=False
            ).squeeze()

        w = mask.unsqueeze(-1)  # (H,W,1)
        weighted_sum = (ab_t * w).sum(dim=(0, 1))         # (2,)
        weight_total = w.sum() + self.eps
        return weighted_sum / weight_total
    
    def build_deltaY_weight_mask(self, ori_pil, relit_pil, q=92):
        """
        ROI from luminance change |ΔY|
        - Gaussian blur to suppress edge noise
        - Hard percentile threshold (binary mask)
        """

        ori = self.resizer(ori_pil).convert("RGB")
        rel = self.resizer(relit_pil).convert("RGB")

        o = np.asarray(ori).astype(np.float32) / 255.0
        r = np.asarray(rel).astype(np.float32) / 255.0

        Yo = 0.2126 * o[..., 0] + 0.7152 * o[..., 1] + 0.0722 * o[..., 2]
        Yr = 0.2126 * r[..., 0] + 0.7152 * r[..., 1] + 0.0722 * r[..., 2]

        d = np.abs(Yr - Yo)

        # 🔥 去邊緣雜訊
        import cv2
        d = cv2.GaussianBlur(d, (7, 7), 0)

        # 🔥 只取顯著變化區域
        thr_val = np.percentile(d, q)
        mask = (d > thr_val).astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.from_numpy(mask).to(device)
    
    def ca_directional_delta_ab(self, ori_pil, relit_pil, target_rgb, weight_mask=None):
        """
        Directional CA:
        cosine( mean_ROI(Δab), target_ab )
        """

        ori = self.resizer(ori_pil).convert("RGB")
        rel = self.resizer(relit_pil).convert("RGB")

        o = np.asarray(ori).astype(np.float32) / 255.0
        r = np.asarray(rel).astype(np.float32) / 255.0

        o_lab = skcolor.rgb2lab(o)
        r_lab = skcolor.rgb2lab(r)

        delta_ab = (r_lab[:, :, 1:3] - o_lab[:, :, 1:3]).astype(np.float32)

        device = weight_mask.device if weight_mask is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d_t = torch.from_numpy(delta_ab).to(device)

        if weight_mask is None:
            dp = d_t.mean(dim=(0, 1))
        else:
            m = weight_mask.to(device)
            if m.shape[-2:] != d_t.shape[:2]:
                m = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0),
                    size=d_t.shape[:2],
                    mode="bilinear",
                    align_corners=False
                ).squeeze()
            w = m.unsqueeze(-1)
            dp = (d_t * w).sum(dim=(0, 1)) / (w.sum() + self.eps)

        # target ab direction
        trgb = (np.array(target_rgb, dtype=np.float32) / 255.0).reshape(1, 1, 3)
        t_ab = skcolor.rgb2lab(trgb)[0, 0, 1:3].astype(np.float32)
        t = torch.from_numpy(t_ab).to(device)

        if torch.norm(dp) < 1e-6:
            return 0.0

        return F.cosine_similarity(
            dp.unsqueeze(0),
            t.unsqueeze(0),
            dim=1,
            eps=self.eps
        ).item()
        
    def ca_directional_deltaY_roi(self, ori_pil, relit_pil, target_rgb, q=92):
        """
        ROI = |ΔY|
        CA  = cosine(mean_ROI(Δab), target_ab)
        """
        w = self.build_deltaY_weight_mask(ori_pil, relit_pil, q=q)

        if float(w.sum().item()) < 1e-6:
            return 0.0

        return self.ca_directional_delta_ab(
            ori_pil,
            relit_pil,
            target_rgb,
            weight_mask=w
        )

    def save_ca_roi_mask(self, ori_pil, relit_pil, save_prefix, q=92):
        """
        Save:
            save_prefix + "_mask.png"
            save_prefix + "_overlay.png"
        """

        w = self.build_deltaY_weight_mask(ori_pil, relit_pil, q=q)
        w_np = w.detach().cpu().numpy()

        # -------- save mask --------
        mask_u8 = (w_np * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_u8, mode="L")
        mask_pil.save(save_prefix + "_mask.png")

        # -------- save overlay --------
        base = self.resizer(relit_pil).convert("RGB")
        base_np = np.asarray(base).astype(np.float32)

        red_overlay = np.zeros_like(base_np)
        red_overlay[..., 0] = 255  # red channel

        alpha = 0.5
        overlay = base_np.copy()
        overlay[w_np > 0] = (
            base_np[w_np > 0] * (1 - alpha)
            + red_overlay[w_np > 0] * alpha
        )

        overlay = overlay.astype(np.uint8)
        Image.fromarray(overlay).save(save_prefix + "_overlay.png")

    def calculate_ca(self, delta_p_list, target_p_list):
        scores = []
        for dp, pt in zip(delta_p_list, target_p_list):
            if torch.norm(dp) < 1e-6:
                scores.append(0.0)
                continue
            cos_sim = F.cosine_similarity(dp.unsqueeze(0), pt.unsqueeze(0), eps=self.eps)
            scores.append(cos_sim.item())
        return np.mean(scores) if scores else 0.0

    def calculate_im(self, alpha_list, luma_list):
        if len(alpha_list) < 2:
            return 0.0, 0.0
        spearman_corr, _ = spearmanr(alpha_list, luma_list)
        pearson_corr, _ = pearsonr(alpha_list, luma_list)
        return float(spearman_corr), float(pearson_corr)

# ==========================================
# [Class] Differentiable SSIM for Guidance
# 說明：用於 Guided Generation 的可微分 SSIM Loss
# ==========================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        # 輸入必須是 (B, C, H, W)，範圍 0~1
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1.0 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

# ==========================================
# 1. 通用設定與模型載入 (Setup Pipeline)
# ==========================================
def setup_pipeline(args):
    print(f"--- [Setup] Initializing Pipeline for Version: {args.version} ---")
    
    # 下載或讀取本地模型
    if os.path.exists(args.version) and os.path.isdir(args.version):
        print(f"  > Found local experiment folder: {args.version}")
        experiment_dir = args.version
    else:
        print(f"  > Local folder '{args.version}' not found. Downloading from Hugging Face...")
        local_dir = snapshot_download(repo_id="Miayan/project", repo_type="model", allow_patterns=[f"{args.version}/*"])
        experiment_dir = os.path.join(local_dir, args.version)
    
    # 讀取最新的 Checkpoint
    checkpoints = [f for f in os.listdir(experiment_dir) if f.startswith('checkpoint')]
    best_ckpt = natsorted(checkpoints)[-1]
    ckpt_path = os.path.join(experiment_dir, best_ckpt)
    print(f"  > Using checkpoint: {ckpt_path}")

    # 讀取 Config (判斷是否使用 Ambient Condition, Color ControlNet 等)
    config_path = os.path.join(experiment_dir, "config.yaml")
    train_cfg = OmegaConf.load(config_path) if os.path.exists(config_path) else None
    
    use_ambient_cond = train_cfg.get('model', {}).get('enable_ambient_cond', False) if train_cfg else False
    relighting_impl = train_cfg.get('data', {}).get('relighting_impl', 'v2') if train_cfg else 'v2'
    use_pred_albedo = train_cfg.get('albedo_estimator', {}).get('enabled', False) if train_cfg else False
    
    # [Ex14+ Feature] 檢查是否在 ControlNet 中使用顏色
    use_color_on_lightmap = train_cfg.get('data', {}).get('use_color_on_lightmap', False) if train_cfg else False
    
    resolution = train_cfg.get('data', {}).get('resolution', 512) if train_cfg else 512
    print(f"  > Using resolution: {resolution}x{resolution}")

    # 選擇 Relighting API 版本
    if relighting_impl == 'gemini_amb':
        api = relighting_api_gemini_amb
        returns_ambient = True
    elif relighting_impl == 'gemini':
        api = relighting_api_gemini
        returns_ambient = False
    else:
        api = relighting_api_v2
        returns_ambient = False
        
    EncoderClass = CustomEncoderAmb if use_ambient_cond else CustomEncoderV1

    # Load Models (UNet, ControlNet, VAE, etc.)
    base_model_path = "stabilityai/stable-diffusion-2-1"
    
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    _replace_unet_conv_in(unet) 
    unet.load_state_dict(torch.load(os.path.join(ckpt_path, "custom_unet.pth")), strict=False)
    unet.cuda().eval()
    
    controlnet = ControlNetModel.from_pretrained(os.path.join(ckpt_path, "controlnet"), torch_dtype=torch.float32)
    cond_encoder = EncoderClass.from_pretrained(os.path.join(ckpt_path, "custom_encoder"), torch_dtype=torch.float32)
    
    albedo_wrapper = None
    if use_pred_albedo and os.path.exists(os.path.join(ckpt_path, "albedo_estimator")):
        albedo_wrapper = AlbedoWrapper.from_pretrained(os.path.join(ckpt_path, "albedo_estimator"), low_cpu_mem_usage=False)
        albedo_wrapper.to('cuda').eval()

    # Pipeline 初始化
    if any(v in args.version for v in ['ex2_3', 'ex2_4', 'ex2_5']):
        from pipeline_cn_ex2_3 import CustomControlNetPipeline
    else:
        from pipeline_cn import CustomControlNetPipeline

    pipe = CustomControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, cond_encoder=cond_encoder, unet=unet, torch_dtype=torch.float32
    )
    pipe.to('cuda')

    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
    vae.to('cuda')
    
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    # Transforms
    img_transform = transforms.Compose([
        transforms.Resize((resolution, resolution), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    cond_transform = transforms.Compose([
        transforms.Resize((resolution, resolution), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    return {
        "pipe": pipe,
        "unet": unet,
        "controlnet": controlnet,
        "cond_encoder": cond_encoder,
        "vae": vae,
        "scheduler": noise_scheduler,
        "albedo_wrapper": albedo_wrapper,
        "api": api,
        "returns_ambient": returns_ambient,
        "use_ambient_cond": use_ambient_cond,
        "use_color_on_lightmap": use_color_on_lightmap,
        "img_transform": img_transform,
        "cond_transform": cond_transform,
        "resolution": resolution,
        "COND_CONFIG": { 
            "lightmap": ['train_ex2_2', 'train_ex2_3', 'train_ex2_4', 'train_ex4', 'train_ex4_1', 'train_ex5', 
                         'train_ex5_1', 'train_ex6', 'train_ex6_1', 'train_ex6_2', 'train_ex6_3', 'train_ex6_4',
                         'train_ex6_5', 'train_ex6_6', 'train_ex6_7', 'train_ex6_8', 'train_ex6_9', 'train_ex6_10',
                         'train_ex6_11', 'train_ex6_12', 'train_ex6_13', 'train_ex7', 'train_ex8', 'train_ex8_1', 
                         'train_ex8_2', 'train_ex8_3', 'train_ex8_4', 'train_ex8_5', 'train_ex8_6', 'train_ex8_7',
                         'train_ex8_8', 'train_ex8_9', 'train_ex8_10', 'train_ex8_11', 'train_ex8_12', 'train_ex8_13', 
                         'train_ex9', 'train_ex9_1', 'train_ex10', 'train_ex10_1'],
            "prompt": ['train_ex2_3', 'train_ex2_4', 'train_ex2_5']
        }
    }


# ==========================================
# [Function] Double Normalized SSIM Guided Generation
# 說明：
#   這是為了 SSIM Guidance 設計的手動去噪迴圈。
#   採用「雙重歸一化」策略：將 Pred 與 Target 都投射到 Mean=0.5, Std=0.2 的空間。
#   這能確保 SSIM 只專注於「結構」，而不會被兩者的亮度或顏色差異誤導。
# ==========================================
def guided_generation(ctx, pipe_kwargs, target_img_tensor, guidance_scale=0.0):
    scheduler = ctx['scheduler']
    unet = ctx['unet']
    controlnet = ctx['controlnet']
    cond_encoder = ctx['cond_encoder']
    vae = ctx['vae']
    
    # 1. 準備 Conditions (Intensity, Color, Ambient)
    intensity = pipe_kwargs['intensity'].cuda()
    color = pipe_kwargs['color'].cuda()
    
    c_cond_tuple = pipe_kwargs['image']
    c_cond_list = []
    for t in c_cond_tuple:
        t = t.cuda()
        if t.ndim == 3: t = t.unsqueeze(0)
        c_cond_list.append(t)
    controlnet_cond = torch.cat(c_cond_list, dim=1)
    
    if ctx['use_ambient_cond']:
        encoder_hidden_states = cond_encoder(intensity, pipe_kwargs['ambient'].cuda(), color)
    else:
        encoder_hidden_states = cond_encoder(intensity, color)

    # 2. 準備 Timesteps
    num_inference_steps = pipe_kwargs['num_inference_steps']
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    
    # 3. 準備 Latents (從 Albedo Latents 加上 Noise)
    albedo_latents = pipe_kwargs['albedo_latents'].clone().cuda() 
    
    generator = pipe_kwargs.get('generator', None)
    latents = torch.randn(albedo_latents.shape, generator=generator, device='cpu', dtype=albedo_latents.dtype).to('cuda')
    latents = latents * scheduler.init_noise_sigma
    
    # SSIM Loss Module
    ssim_criterion = SSIMLoss().cuda()
    target_img_01 = (target_img_tensor.cuda() + 1.0) / 2.0
    
    # [Inner Function] Robust Normalization for SSIM
    def robust_normalize(img_tensor):
        # 1. 轉灰階 (只看亮度結構)
        gray = img_tensor.mean(dim=1, keepdim=True)
        # 2. 計算統計量
        mu = gray.mean(dim=(2, 3), keepdim=True)
        std = gray.std(dim=(2, 3), keepdim=True)
        # 3. 歸一化 (Zero-Mean, Unit-Variance)
        normalized = (gray - mu) / (std + 1e-5)
        # 4. 重映射到 SSIM 友善區間 (0.5中心)
        remapped = normalized * 0.2 + 0.5
        return remapped.clamp(0.0, 1.0)

    # 預先處理 Target，避免重複計算
    target_norm = robust_normalize(target_img_01)

    print(f"  > Starting Guided Inference (Scale: {guidance_scale}, Double Norm: True)...")
    
    # 4. 去噪迴圈
    for t in tqdm(timesteps, desc="Guided Sampling"):
        with torch.enable_grad():
            latents = latents.detach().requires_grad_(True)
            unet_input = torch.cat([latents, albedo_latents], dim=1)
            
            # ControlNet Forward
            down_block_res_samples, mid_block_res_sample, _ = controlnet(
                latents, t, encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond, return_dict=False,
            )

            # UNet Forward
            noise_pred = unet(
                unet_input, t, encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[sample.to(dtype=torch.float32) for sample in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float32),
                return_dict=False,
            )[0]
            
            # --- Guidance Step ---
            if guidance_scale > 0:
                # 預測 x0
                step_output = scheduler.step(noise_pred, t, latents, return_dict=True)
                pred_x0_latents = step_output.pred_original_sample
                # Decode VAE (取得像素空間圖像)
                pred_x0_img = vae.decode(pred_x0_latents / vae.config.scaling_factor).sample
                pred_x0_img = (pred_x0_img + 1.0) / 2.0
                # 歸一化預測圖
                pred_norm = robust_normalize(pred_x0_img)
                
                # 計算 SSIM Loss 並反向傳播
                loss = ssim_criterion(pred_norm, target_norm)
                grads = torch.autograd.grad(loss, latents)[0]
                
                # 更新 Latents
                latents = latents - guidance_scale * grads
                
        latents = latents.detach()
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 5. Decode Final Image
    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    
    image = (image * 255).round().astype("uint8")
    return [Image.fromarray(image[0])]

# ==========================================
# 2. 單次推理模式 (Single Inference)
# 說明：最基本的模式，會詳細輸出 Lightmap, Albedo, Reconstruction 等中間產物方便 Debug。
# ==========================================
def run_single_inference(ctx, args, dataset, indices):
    print("--- Mode: Single Inference ---")
    out_dir = f'./inference/{args.version}/{args.mode}_{args.guidance_scale:04.0f}'
    if args.use_pretrained_alb:
        out_dir += "_prealb"
    if args.auto_exposure:
        out_dir += "_autoexp"
    os.makedirs(out_dir, exist_ok=True)

    for idx in indices:
        item = dataset[idx]
        data = _prepare_batch_data(ctx, item, args, idx)
        
        # 呼叫統一的推論邏輯
        image = _run_inference_internal(ctx, data, args, item)

        # 計算四項品質指標
        target = data['target_pil'] if args.mode == 'lightlab' else data['ori_pil']
        p_score = _calculate_psnr(image, target)
        s_score = _calculate_ssim(image, target)
        n_score = _calculate_niqe(image)
        b_score = _calculate_brisque(image)
        l_score = _calculate_lpips(image, target)
        
        # 儲存結果與對照圖
        suffix = f"{idx}_{args.seed}"
        if args.guidance_scale > 0:
            suffix += f"_gs{args.guidance_scale}"
        
        save_inference_visuals(
            out_dir,
            suffix,
            data["ori_pil"],
            image,
            data["target_pil"],
            mask_pil=data["mask"]  # 有 mask 再放
        )
        
        print(f"[Single] ID {idx}  | PSNR: {p_score:.4f} | SSIM: {s_score:.4f} | NIQE: {n_score:.4f} | BRISQUE: {b_score:.4f} | LPIPS: {l_score:.4f}")
    print(f"Saved to: {out_dir}")

# ==========================================
# 3. [New] Multi-Condition Sweep (Colors & Intensities)
# 說明：針對同一張圖，自動測試多種顏色與光強，並確保 ControlNet Condition 也跟著改變。
# ==========================================
def run_multicond_sweep(ctx, args, dataset, indices):
    print("--- Mode: Multi-Condition Sweep (All Metrics Enabled) ---")
    
    control_tool = RelightingControlMetrics(resolution=ctx['resolution'])
    
    COLORS = {
        "Red": [255, 0, 0], "Yellow": [255, 255, 0], "Green": [0, 255, 0],
        "Cyan": [0, 255, 255], "Blue": [0, 0, 255], "Magenta": [255, 0, 255],
        # "Orange":  [255, 165, 0],
        # "Purple":  [128, 0, 128],
        # "Pink":   [255, 192, 203],
    }
    INTENSITIES = [0.1, 0.2, 0.4, 0.7, 1.0] # 拿掉0.0
    
    out_dir = f'./inference/{args.version}/sweep_multicond'
    os.makedirs(out_dir, exist_ok=True)
    
    for idx in indices:
        item = dataset[idx]
        print(f"\n" + "="*50 + f"\nProcessing Image ID: {idx}\n" + "="*50)
        
        # 1. 準備量測區域 (Lightmap Luma)        
        # 色度基準
        # p_input = control_tool.get_chromaticity_vector(item['image'])

        # --- 第一階段：Color Sweep (計算品質指標 + CA) ---
        print("\n[Phase 1] Sweeping Colors...")
        delta_p_list, target_p_list = [], []
        first_save = True
        
        for c_name, c_val in COLORS.items():
            data = _prepare_batch_data(ctx, item, args, idx, override_color=c_val, override_intensity=1.0)
            image = _run_inference_internal(ctx, data, args, item)
            
            # (1) 原本的品質指標
            s = _calculate_ssim(image, data['ori_pil'])
            n = _calculate_niqe(image)
            b = _calculate_brisque(image)
            l = _calculate_lpips(image, data['ori_pil'])
            
            # (2) CA 指標相關紀錄
            # p_relit = control_tool.get_chromaticity_vector(image)
            # delta_p_list.append(p_relit - p_input)
            # c_t = torch.tensor(c_val, dtype=torch.float32) / 255.0  # (3,)
            # target_rgb = c_t.cpu().numpy().reshape(1, 1, 3).astype(np.float32)
            # target_lab = skcolor.rgb2lab(target_rgb)  # (1,1,3)
            # target_ab = torch.tensor(target_lab[0, 0, 1:3], dtype=torch.float32, device=p_input.device)  # (2,)
            # target_p_list.append(target_ab - p_input)
            
            # ca_single = F.cosine_similarity((p_relit - p_input).unsqueeze(0), (target_ab - p_input).unsqueeze(0),).item()

            ca_single = control_tool.ca_directional_deltaY_roi(
                data["ori_pil"],   # input PIL
                image,             # relit PIL
                c_val,             # target RGB in 0..255 (你原本的 c_val 就是)
                q=92
            )
            
            if first_save:
                prefix = os.path.join(out_dir, f"img{idx}_roi")
                control_tool.save_ca_roi_mask(
                    data["ori_pil"],
                    image,
                    prefix,
                    q=92
                )
                first_save = False
            
            image.save(os.path.join(out_dir, f"img{idx}_color_{c_name}.png"))
            print(f"    Color {c_name:8} | SSIM: {s:.4f} | NIQE: {n:.4f} | BRISQUE: {b:.4f} | LPIPS: {l:.4f} | CA: {ca_single:.4f}")

        # 計算最終 CA
        ca_score = control_tool.calculate_ca(delta_p_list, target_p_list)

        # --- 第二階段：Intensity Sweep (計算品質指標 + IM) ---
        print("\n[Phase 2] Sweeping Intensities...")
        alpha_list, luma_list = [], []
        luma_v_3d = control_tool.luma_weights.view(3, 1, 1).to('cuda')
        
        for ints in INTENSITIES:
            data = _prepare_batch_data(ctx, item, args, idx, override_color=COLORS["Red"], override_intensity=ints)
            image = _run_inference_internal(ctx, data, args, item)
            
            # (1) 原本的品質指標
            s = _calculate_ssim(image, data['ori_pil'])
            n = _calculate_niqe(image)
            b = _calculate_brisque(image)
            l = _calculate_lpips(image, data['ori_pil'])
            
            # (2) IM 指標相關紀錄
            img_t = transforms.ToTensor()(image).to('cuda')
            luma_map = (img_t * luma_v_3d).sum(dim=0)
            avg_luma = luma_map.mean()
            alpha_list.append(ints)
            luma_list.append(avg_luma.item())

            image.save(os.path.join(out_dir, f"img{idx}_intensity_{ints}.png"))
            print(f"    Intensity {ints:<4} | SSIM: {s:.4f} | NIQE: {n:.4f} | BRISQUE: {b:.4f} | LPIPS: {l:.4f}")

        # 計算最終 IM
        im_spearman, im_pearson = control_tool.calculate_im(alpha_list, luma_list)
        
        # 畫 intensity–luminance curve
        plt.figure()
        plt.plot(alpha_list, luma_list, marker='o')
        plt.xlabel("Intensity (alpha)")
        plt.ylabel("Mean luminance (Y)")
        plt.title(f"Intensity Response Curve (Image {idx})")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"img{idx}_intensity_curve.png"), dpi=200)
        plt.close()

        # --- 最後總結列印 ---
        print("\n" + "-"*30)
        print(f"SUMMARY FOR IMAGE ID {idx}:")
        print(f"  > Color Controllability (CA):     {ca_score:.4f} (Ideally 1.0)")
        print(f"  > Intensity Monotonicity (Spearman): {im_spearman:.4f} (Ideally 1.0)")
        print(f"  > Intensity Linearity (Pearson):     {im_pearson:.4f} (Ideally 1.0)")
        print("-"*30 + "\n")
    print(f"\nSweep Completed. Results saved to {out_dir}")

# ==========================================
# 4. Benchmark 模式 (Top 1% SSIM -> NIQE)
# 說明：
#   跑大量數據，不儲存中間圖片與 JSON。
#   採用「一邊生成一邊篩選」的 Min-Heap 策略，最後才對 Top 1% 進行排序並存成 CSV。
# ==========================================
def run_benchmark_phase1_quality(ctx, args, dataset, indices, out_dir):
    """
    Phase 1:
      - 每張圖 inference 一次
      - 算 SSIM/NIQE/BRISQUE/LPIPS
      - 用 heap 維持 TopK：SSIM 高優先，NIQE 低作 tie-break
      - 輸出 topk_quality.csv
    """
    top_k = int(getattr(args, "top_k", 100))
    csv_name = getattr(args, "topk_csv_name", "topk_quality.csv")

    print(f"[Phase1] Selecting Top {top_k} from {len(indices)} images by SSIM (tie-break: NIQE lower)")

    total = {"ssim": 0.0, "niqe": 0.0, "brisque": 0.0, "lpips": 0.0}
    count = 0
    top_heap = []  # store: (ssim, -niqe, idx, niqe, brisque, lpips)

    pbar = tqdm(indices, desc="Phase1: Quality")
    for idx in pbar:
        item = dataset[idx]

        data = _prepare_batch_data(ctx, item, args, idx)
        image = _run_inference_internal(ctx, data, args, item)

        s_score = _calculate_ssim(image, data["ori_pil"])
        n_score = _calculate_niqe(image)
        b_score = _calculate_brisque(image)
        l_score = _calculate_lpips(image, data["ori_pil"])

        total["ssim"] += float(s_score)
        total["niqe"] += float(n_score)
        total["brisque"] += float(b_score)
        total["lpips"] += float(l_score)
        count += 1

        entry = (float(s_score), -float(n_score), int(idx), float(n_score), float(b_score), float(l_score))

        if len(top_heap) < top_k:
            heapq.heappush(top_heap, entry)
        else:
            worst = top_heap[0]
            if (entry[0], entry[1]) > (worst[0], worst[1]):
                heapq.heapreplace(top_heap, entry)

        pbar.set_postfix({"avg_ssim": f"{total['ssim']/count:.4f}"})

    print("\n[Phase1] Full-set Quality Averages:")
    avg_ssim = total["ssim"] / count
    avg_niqe = total["niqe"] / count
    avg_brisque = total["brisque"] / count
    avg_lpips = total["lpips"] / count
    print(f"  SSIM   : {avg_ssim:.6f}")
    print(f"  NIQE   : {avg_niqe:.6f}")
    print(f"  BRISQUE: {avg_brisque:.6f}")
    print(f"  LPIPS  : {avg_lpips:.6f}")

    top_results = sorted(top_heap, key=lambda x: (x[0], x[1]), reverse=True)
    csv_path = _save_topk_quality_csv(out_dir, top_results, filename=csv_name)

    return csv_path, avg_ssim, avg_niqe, avg_brisque, avg_lpips

def run_benchmark_phase2_color_to_hf(ctx, args, dataset, topk_csv_path, out_dir):
    """
    Phase 2A (Color):
      - 讀 topk_quality.csv
      - 對 TopK 做顏色控制（預設 R/G/B），intensity 固定（預設 1.0）
      - 每張圖每個顏色都算 CA
      - 存成 HF Dataset：image_id / input / mask / relit / color_score + phase1 分數
    """
    rows = _load_topk_quality_csv(topk_csv_path)

    color_str = getattr(args, "control_colors", "Red,Green,Blue")
    color_names = [c.strip() for c in color_str.split(",") if c.strip()]
    intensity = float(getattr(args, "control_intensity_for_color", 1.0))

    max_items = getattr(args, "phase2_max_items", None)
    if max_items is not None:
        rows = rows[:int(max_items)]

    COLORS = {
        "Red":   [255, 0, 0],
        "Green": [0, 255, 0],
        "Blue":  [0, 0, 255],
    }

    control_tool = RelightingControlMetrics(resolution=ctx["resolution"])

    records = {
        "image_id": [],
        "input_image": [],
        "mask": [],
        "relit_image": [],
        "color_name": [],
        "intensity": [],
        "color_score": [],
        "ssim": [],
        "niqe": [],
        "brisque": [],
        "lpips": [],
    }

    print(f"[Phase2-Color] colors={color_names}, intensity={intensity}, n_images={len(rows)}")

    for row in tqdm(rows, desc="Phase2: Color"):
        idx = row["image_id"]
        item = dataset[idx]

        # input PIL
        input_pil = item["image"] if (isinstance(item, dict) and "image" in item) else item["image"]

        # mask PIL
        if args.mode == 'lightlab':
            print(f"Warning: Using lightlab mode for benchmark, cancel process. ")
            break
        else:
            mask = item['mask'].convert('L')

        # p_input once per image
        # p_input = control_tool.get_chromaticity_vector(input_pil, None)
        # dev = p_input.device
        dev = "cuda"  # 假設 control_tool 內部會把向量轉到 cuda

        for cname in color_names:
            if cname not in COLORS:
                continue
            rgb = COLORS[cname]

            data = _prepare_batch_data(ctx, item, args, idx,
                                       override_color=rgb,
                                       override_intensity=float(intensity))
            relit_pil = _run_inference_internal(ctx, data, args, item)

            # p_relit = control_tool.get_chromaticity_vector(relit_pil, None)
            # target_ab = _lab_target_ab_from_rgb(rgb, dev)

            # delta_p = p_relit - p_input
            # target_dir = target_ab - p_input

            # ca = F.cosine_similarity(delta_p.unsqueeze(0), target_dir.unsqueeze(0), dim=1).item()
            
            ca = control_tool.ca_directional_deltaY_roi(
                input_pil,
                relit_pil,
                rgb,      # target RGB in 0..255
                q=92
            )

            records["image_id"].append(int(idx))
            records["input_image"].append(input_pil)
            records["mask"].append(mask)
            records["relit_image"].append(relit_pil)
            records["color_name"].append(cname)
            records["intensity"].append(float(intensity))
            records["color_score"].append(float(ca))

            records["ssim"].append(float(row["ssim"]))
            records["niqe"].append(float(row["niqe"]))
            records["brisque"].append(float(row["brisque"]))
            records["lpips"].append(float(row["lpips"]))

    hf_ds = Dataset.from_dict(records)

    save_name = getattr(args, "phase2_hf_name", "phase2_color_hf")
    save_path = os.path.join(out_dir, save_name)
    hf_ds.save_to_disk(save_path)
    print(f"[Phase2-Color] Saved HF dataset -> {save_path}")

    if getattr(args, "push_to_hub", False):
        repo = getattr(args, "hf_repo_name", None)
        if repo is None:
            raise ValueError("push_to_hub=True but hf_repo_name is None")
        hf_ds.push_to_hub(repo)
        print(f"[Phase2-Color] Pushed to hub -> {repo}")

    # print mean CA per color
    avg_ca = []
    for cname in color_names:
        vals = [v for (n, v) in zip(records["color_name"], records["color_score"]) if n == cname]
        if len(vals):
            mean_val = float(np.mean(vals))
            print(f"[Phase2-Color] Mean CA ({cname}) = {mean_val:.4f}")
            avg_ca.append([cname, mean_val])

    return save_path, avg_ca

def run_benchmark_phase2_intensity(ctx, args, dataset, topk_csv_path, out_dir):
    """
    Phase 2B (Intensity):
      - 讀 topk_quality.csv
      - 對 TopK 的每張圖跑 intensity sweep
      - 每張圖把 luma_0.0 ... luma_1.0 存成一列 CSV（方便後續統計）
      - 再算 dataset mean curve + Spearman/Pearson，並輸出 curve CSV
    """
    rows = _load_topk_quality_csv(topk_csv_path)

    max_items = getattr(args, "phase2_max_items", None)
    if max_items is not None:
        rows = rows[:int(max_items)]

    # intensity sweep
    sweep_str = getattr(args, "control_intensity_sweep", "0.0,0.2,0.5,0.8,1.0")
    intensity_list = [float(x) for x in sweep_str.split(",")]

    # intensity evaluation uses white light by default
    intensity_color = [255, 255, 255]

    control_tool = RelightingControlMetrics(resolution=ctx["resolution"])
    luma_v_3d = control_tool.luma_weights.view(3, 1, 1).to("cuda")

    perimg_csv = os.path.join(out_dir, getattr(args, "phase2_intensity_csv", "phase2_intensity_per_image.csv"))
    curve_csv = os.path.join(out_dir, getattr(args, "phase2_intensity_curve_csv", "phase2_intensity_curve.csv"))

    # ---- per-image csv ----
    with open(perimg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["image_id", "ssim", "niqe", "brisque", "lpips"] + [f"luma_{a}" for a in intensity_list]
        w.writerow(header)

        for row in tqdm(rows, desc="Phase2: Intensity(per-image)"):
            idx = row["image_id"]
            item = dataset[idx]

            luma_values = []
            for alpha in intensity_list:
                data_i = _prepare_batch_data(ctx, item, args, idx,
                                             override_color=intensity_color,
                                             override_intensity=float(alpha))
                image_i = _run_inference_internal(ctx, data_i, args, item)

                img_t = transforms.ToTensor()(image_i).to("cuda")
                luma_map = (img_t * luma_v_3d).sum(dim=0)
                luma_values.append(float(luma_map.mean().item()))

            w.writerow([
                int(idx),
                float(row["ssim"]),
                float(row["niqe"]),
                float(row["brisque"]),
                float(row["lpips"]),
                *luma_values
            ])

    print(f"[Phase2-Intensity] Saved per-image intensity CSV -> {perimg_csv}")

    # ---- compute dataset mean curve from per-image csv ----
    per_rows = []
    with open(perimg_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for rr in r:
            per_rows.append(rr)

    mean_curve = []
    for a in intensity_list:
        key = f"luma_{a}"
        vals = [float(rr[key]) for rr in per_rows]
        mean_curve.append(float(np.mean(vals)))

    sp, _ = spearmanr(intensity_list, mean_curve)
    pr, _ = pearsonr(intensity_list, mean_curve)

    # save curve csv
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "mean_luma"])
        for a, y in zip(intensity_list, mean_curve):
            w.writerow([a, y])
        w.writerow([])
        w.writerow(["spearman", float(sp)])
        w.writerow(["pearson", float(pr)])

    print(f"[Phase2-Intensity] Saved mean curve CSV -> {curve_csv}")
    print(f"[Phase2-Intensity] Spearman={float(sp):.4f}, Pearson={float(pr):.4f}")

    return perimg_csv, curve_csv, mean_curve, float(sp), float(pr)

def run_benchmark_three_phase(ctx, args, dataset, indices):
    out_dir = f'./inference/{args.version}/benchmark_{args.guidance_scale:04.0f}'
    if args.use_pretrained_alb:
        out_dir += "_prealb"
    if args.auto_exposure:
        out_dir += "_autoexp"
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Phase 1
    topk_csv_path, avg_ssim, avg_niqe, avg_brisque, avg_lpips = run_benchmark_phase1_quality(ctx, args, dataset, indices, out_dir)

    # Phase 2A: Color -> HF dataset
    _, avg_ca = run_benchmark_phase2_color_to_hf(ctx, args, dataset, topk_csv_path, out_dir)

    # Phase 2B: Intensity -> per-image CSV + mean curve CSV
    _, _, _, spearman_im, pearson_im = run_benchmark_phase2_intensity(ctx, args, dataset, topk_csv_path, out_dir)

    print("\n[Benchmark] Done (Phase1 + Phase2-Color + Phase2-Intensity).")
    
    record_file = f"{out_dir}/benchmark_results.txt"
    with open(record_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"實驗名稱: {args.version}\n")
        f.write(f"總共跑了幾張圖: {len(indices)}\n")
        f.write(f"平均分數結果:\n")
        f.write(f"  SSIM: {avg_ssim:.4f}\n")
        f.write(f"  LPIPS: {avg_lpips:.4f}\n")
        f.write(f"  NIQE: {avg_niqe:.4f}\n")
        f.write(f"  BRISQUE: {avg_brisque:.4f}\n")
        for cname, ca_val in avg_ca:
            f.write(f"  CA (Color Accuracy) - {cname}: {ca_val:.4f}\n")
        f.write(f"  IM (Intensity Monotonicity) - Spearman: {spearman_im:.4f}, Pearson: {pearson_im:.4f}\n")
        f.write(f"{'='*50}\n")
    
    print(f"\n[Done] 數據已自動存入 {record_file}")

def _save_topk_quality_csv(out_dir, top_results, filename="topk_quality.csv"):
    """
    top_results: list of tuples (ssim, neg_niqe, idx, niqe, brisque, lpips)
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "image_id", "ssim", "niqe", "brisque", "lpips"])
        for rank, (ssim, neg_niqe, idx, niqe, brisque, lpips) in enumerate(top_results):
            w.writerow([rank, idx, float(ssim), float(niqe), float(brisque), float(lpips)])
    print(f"[Phase1] Saved TopK CSV -> {csv_path}")
    return csv_path

def _load_topk_quality_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "rank": int(row["rank"]),
                "image_id": int(row["image_id"]),
                "ssim": float(row["ssim"]),
                "niqe": float(row["niqe"]),
                "brisque": float(row["brisque"]),
                "lpips": float(row["lpips"]),
            })
    return rows

# ==========================================
# 5. BigTime資料集推理模式 (BigTime Inference)
# 說明：最基本的模式，會詳細輸出 Lightmap, Albedo, Reconstruction 等中間產物方便 Debug。
# ==========================================
def run_bigtime_test(ctx, args, dataset, gt_configs):
    """
    ctx: 模型組件 (unet, controlnet 等)
    args: 推理設定
    """
    print("--- Mode: BigTime Inference ---")
    out_dir = f'./inference/{args.version}/bigtime_{args.guidance_scale:04.0f}'
    if args.use_pretrained_alb:
        out_dir += "_prealb"
    if args.auto_exposure:
        out_dir += "_autoexp"
    os.makedirs(out_dir, exist_ok=True)
    results = []
    
    for i, cfg in enumerate(tqdm(gt_configs, desc="BigTime Testing")):
        # 1. 取得資料
        input_img_id = cfg['input_id']
        data = _prepare_bigtime_data(ctx, dataset[input_img_id], args, cfg)
        
        # 2. 模型推理
        image = _run_inference_internal(ctx, data, args)

        # 3. 計算分數
        p_score = _calculate_psnr(image, data["gt_image"])
        s_score = _calculate_ssim(image, data["gt_image"])
        n_score = _calculate_niqe(image)
        b_score = _calculate_brisque(image)
        l_score = _calculate_lpips(image, data["gt_image"])
        
        pseudo = data["target_pil"]  # physics API pseudo GT
        pg_psnr = _calculate_psnr(pseudo, data["gt_image"])
        pg_ssim = _calculate_ssim(pseudo, data["gt_image"])
        pg_lpips = _calculate_lpips(pseudo, data["gt_image"])
        
        results.append({
            "input_id": input_img_id,
            "gt_col": cfg['gt_col'],
            "psnr": p_score,
            "ssim": s_score,
            "niqe": n_score,
            "brisque": b_score,
            "lpips": l_score,
            "pseudo_gt_psnr": pg_psnr,
            "pseudo_gt_ssim": pg_ssim,
            "pseudo_gt_lpips": pg_lpips,
        })
        
        image.save(f"{out_dir}/output_{i:03d}.png")
        data['ori_pil'].save(f"{out_dir}/ori_{i:03d}.png")
        data['gt_image'].save(f"{out_dir}/gt_{i:03d}.png")
        data['target_pil'].save(f"{out_dir}/pseudo_gt_{i:03d}.png")

    # 5. 輸出統計
    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]
    niqes = [r["niqe"] for r in results]
    brisques = [r["brisque"] for r in results]
    lpips = [r["lpips"] for r in results]
    pg_psnrs = [r["pseudo_gt_psnr"] for r in results]
    pg_ssims = [r["pseudo_gt_ssim"] for r in results]
    pg_lpips = [r["pseudo_gt_lpips"] for r in results]
    
    mean_psnr = np.mean(psnrs)
    mean_ssim = np.mean(ssims)
    mean_niqe = np.mean(niqes)
    mean_brisque = np.mean(brisques)
    mean_lpips = np.mean(lpips)
    mean_pg_psnr = np.mean(pg_psnrs)
    mean_pg_ssim = np.mean(pg_ssims)
    mean_pg_lpips = np.mean(pg_lpips)

    print(f"\n📊 BigTime Test Summary:")
    print(f"   Samples: {len(results)}")
    print(f"   Mean PSNR: {mean_psnr:.4f}")
    print(f"   Mean SSIM: {mean_ssim:.4f}")
    print(f"   Mean LPIPS: {mean_lpips:.4f}")
    print(f"   Mean NIQE: {mean_niqe:.4f}")
    print(f"   Mean BRISQUE: {mean_brisque:.4f}")
    print(f"[PseudoGT vs GT] Mean PSNR:  {mean_pg_psnr:.4f}")
    print(f"[PseudoGT vs GT] Mean SSIM:  {mean_pg_ssim:.4f}")
    print(f"[PseudoGT vs GT] Mean LPIPS: {mean_pg_lpips:.4f}")
    
    txt_path = os.path.join(out_dir, "bigtime_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("BigTime Test Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Samples: {len(results)}\n")
        f.write(f"Mean PSNR: {mean_psnr:.6f}\n")
        f.write(f"Mean SSIM: {mean_ssim:.6f}\n")
        f.write(f"Mean NIQE: {mean_niqe:.6f}\n")
        f.write(f"Mean BRISQUE: {mean_brisque:.6f}\n")
        f.write(f"Mean LPIPS: {mean_lpips:.6f}\n")
        f.write(f"[PseudoGT vs GT] Mean PSNR:  {mean_pg_psnr:.6f}\n")
        f.write(f"[PseudoGT vs GT] Mean SSIM:  {mean_pg_ssim:.6f}\n")
        f.write(f"[PseudoGT vs GT] Mean LPIPS: {mean_pg_lpips:.6f}\n")
        f.write("=" * 40 + "\n")

    print(f"\n✅ 結果已儲存至: {txt_path}")
    
    return results

# ==========================================
# 6. Lsun資料集推理模式 (BigTime Inference)
# 說明：最基本的模式，會詳細輸出 Lightmap, Albedo, Reconstruction 等中間產物方便 Debug。
# ==========================================
def run_custom_id_color_test(ctx, args, dataset, image_ids):
    """
    Custom Test:
    - Color controllability metric: CA (cosine similarity in Lab ab direction) averaged over all test images & colors
    - Intensity controllability metric: average Δluma curve (relative to alpha=0.0) over all test images
      and save a mean curve plot (with 95% CI).

    Output:
      ./inference/{args.version}/custom_test_{gs}/
        imgXXXXX/ ... per-image outputs
        intensity_delta_luma_mean_curve.png
        custom_test_summary.txt
    """
    print("--- Mode: Custom Test (IDs x Color + Intensity Metrics) ---")

    control_tool = RelightingControlMetrics(resolution=ctx["resolution"])
    luma_v_3d = control_tool.luma_weights.view(3, 1, 1).to("cuda")

    # TEST_COLOR_LIST = [
    #     ("L_Magenta", [255, 0, 255]),
    #     ("L_Purple",  [125, 0, 255]),
    #     ("L_Cyan",    [0, 255, 255]),
    #     ("L_Green",   [0, 255, 0]),
    #     ("L_Yellow",  [255, 255, 0]),
    # ]
    
    TEST_COLOR_LIST = [
        ("Magenta", [255, 0, 255]),
        ("Blue",  [0, 0, 255]),
        ("Cyan",    [0, 255, 255]),
        ("Green",   [0, 255, 0]),
        ("Yellow",  [255, 255, 0]),
        ("Red",  [255, 0, 0]),
    ]
    # 強度測試：白色 + 6 個強度（包含 0.0 才能算 Δluma）
    TEST_INTENSITY_LIST = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    WHITE = [255, 255, 255]

    out_dir = f'./inference/{args.version}/custom_test_{args.guidance_scale:04.0f}'
    os.makedirs(out_dir, exist_ok=True)

    # ---------
    # Aggregators (across all test images)
    # ---------
    ca_all = []  # all (image,color) CA
    ca_by_color = {name: [] for name, _ in TEST_COLOR_LIST}

    # intensity: collect per-image Δluma arrays shape (N_images, N_intensities)
    delta_luma_all = []

    # =========
    # Loop images
    # =========
    for img_id in image_ids:
        item = dataset[int(img_id)]

        subdir = os.path.join(out_dir, f"img{int(img_id):05d}")
        os.makedirs(subdir, exist_ok=True)

        # Save input
        item["image"].save(os.path.join(subdir, "input.png"))

        # ---------- Color metric (CA) ----------
        # p_input: chromaticity mean of input image in Lab ab
        # p_input = control_tool.get_chromaticity_vector(item["image"], weight_mask=None)
        # dev = p_input.device
        dev = "cuda"  # 假設 control_tool 內部會把向量轉到 cuda

        # fixed intensity for color test
        color_intensity = 1.0
        first_save = True

        for cname, rgb in TEST_COLOR_LIST:
            data = _prepare_batch_data(
                ctx, item, args, int(img_id),
                override_color=rgb,
                override_intensity=color_intensity
            )
            relit = _run_inference_internal(ctx, data, args, item)

            # quality metrics (optional prints)
            s_score = _calculate_ssim(relit, data["ori_pil"])
            n_score = _calculate_niqe(relit)
            b_score = _calculate_brisque(relit)
            l_score = _calculate_lpips(relit, data["ori_pil"])
            
            ca_single = control_tool.ca_directional_deltaY_roi(
                data["ori_pil"],
                relit,
                rgb,      # target RGB in 0..255
                q=92
            )
            
            if first_save:
                prefix = os.path.join(subdir, "roi")
                control_tool.save_ca_roi_mask(
                    data["ori_pil"],
                    relit,
                    prefix,
                    q=92
                )
                first_save = False

            ca_all.append(ca_single)
            ca_by_color[cname].append(ca_single)

            relit.save(os.path.join(subdir, f"color_{cname}.png"))
            data["target_pil"].save(os.path.join(subdir, f"color_{cname}_pseudoGT.png"))
            print(f"[CustomTest-Color] img_id={img_id} | {cname:8} "
                  f"| SSIM:{s_score:.4f} NIQE:{n_score:.4f} BRISQUE:{b_score:.4f} LPIPS:{l_score:.4f} | CA:{ca_single:.4f}")

        # ---------- Intensity metric (Δluma curve) ----------
        # For this part: color fixed to white, sweep intensities
        luma_vals = []
        for alpha in TEST_INTENSITY_LIST:
            data_i = _prepare_batch_data(
                ctx, item, args, int(img_id),
                override_color=WHITE,
                override_intensity=float(alpha)
            )
            relit_i = _run_inference_internal(ctx, data_i, args, item)

            # mean luma (global)
            img_t = transforms.ToTensor()(relit_i).to("cuda")
            luma_map = (img_t * luma_v_3d).sum(dim=0)
            luma_vals.append(float(luma_map.mean().item()))

            relit_i.save(os.path.join(subdir, f"intensity_white_{alpha:.2f}.png"))
            data_i["target_pil"].save(os.path.join(subdir, f"intensity_{alpha:.2f}_pseudoGT.png"))

        # Δluma per-image: subtract luma at 0.0
        base = luma_vals[0]
        delta_vals = [v - base for v in luma_vals]
        delta_luma_all.append(delta_vals)

        print(f"[CustomTest-Intensity] img_id={img_id} | "
              f"Δluma={['{:.4f}'.format(x) for x in delta_vals]}")

    # =========
    # Aggregate & Plot (ALL images mean)
    # =========
    delta_luma_all = np.array(delta_luma_all, dtype=np.float64)  # (N, K)
    mean_delta = delta_luma_all.mean(axis=0)
    # 95% CI
    if delta_luma_all.shape[0] > 1:
        se = delta_luma_all.std(axis=0, ddof=1) / np.sqrt(delta_luma_all.shape[0])
        ci95 = 1.96 * se
    else:
        ci95 = np.zeros_like(mean_delta)

    # --- plot: mean + 95% CI band + individual curves ---
    plt.figure(figsize=(7, 5))

    # # 1) plot per-image curves (thin, transparent)
    # for i in range(delta_luma_all.shape[0]):
    #     plt.plot(
    #         TEST_INTENSITY_LIST,
    #         delta_luma_all[i],
    #         linewidth=1.0,
    #         alpha=0.20
    #     )

    # 2) plot mean curve (thicker)
    plt.plot(
        TEST_INTENSITY_LIST,
        mean_delta,
        marker="o",
        color="deepskyblue",
        linewidth=2.5
    )

    # 3) 95% CI as shaded band (instead of errorbars)
    lower = mean_delta - ci95
    upper = mean_delta + ci95
    plt.fill_between(TEST_INTENSITY_LIST, lower, upper, alpha=0.20)

    plt.xlabel("Condition intensity (alpha)")
    plt.ylabel("Mean Δluma relative to alpha=0.0")
    plt.title(f"Δluma Response")
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(out_dir, "intensity_delta_luma_mean_curve.png")
    plt.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close()

    # =========
    # Summary numbers
    # =========
    mean_ca_all = float(np.mean(ca_all)) if len(ca_all) else 0.0
    mean_ca_by_color = {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in ca_by_color.items()}
    
    # ==============================
    # Plot CA per color (Bar Chart)
    # ==============================

    color_names = [name for name, _ in TEST_COLOR_LIST]
    mean_values = [mean_ca_by_color[name] for name in color_names]

    # 轉成 matplotlib 可用的 RGB (0~1)
    bar_colors = []
    for _, rgb in TEST_COLOR_LIST:
        bar_colors.append([c / 255.0 for c in rgb])

    plt.figure(figsize=(7, 5))
    bars = plt.bar(color_names, mean_values, color=bar_colors)

    plt.ylim(0, 1.0)
    plt.ylabel("Mean CA (Cosine Similarity)")
    plt.title("Color Accuracy per Target Color")
    plt.grid(axis="y", alpha=0.3)

    # 在 bar 上顯示數值
    for i, v in enumerate(mean_values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

    bar_plot_path = os.path.join(out_dir, "ca_per_color_bar.png")
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=250)
    plt.close()

    print(f"📊 CA per color bar chart saved to: {bar_plot_path}")

    print("\n" + "=" * 60)
    print("[CustomTest Summary] (Averaged over all test images)")
    print(f"  Mean CA (all colors): {mean_ca_all:.4f}")
    for cname in [n for n, _ in TEST_COLOR_LIST]:
        print(f"  Mean CA ({cname}): {mean_ca_by_color[cname]:.4f}")
    print(f"  Mean Δluma curve saved to: {plot_path}")
    print("=" * 60)

    # save txt summary
    txt_path = os.path.join(out_dir, "custom_test_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Custom Test Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Version: {args.version}\n")
        f.write(f"Guidance Scale: {args.guidance_scale}\n")
        f.write(f"Num images: {len(image_ids)}\n\n")

        f.write("[Color Metric]\n")
        f.write(f"Mean CA (all colors): {mean_ca_all:.6f}\n")
        for cname in [n for n, _ in TEST_COLOR_LIST]:
            f.write(f"Mean CA ({cname}): {mean_ca_by_color[cname]:.6f}\n")

        f.write("\n[Intensity Metric]\n")
        f.write("Intensity list: " + ",".join([str(a) for a in TEST_INTENSITY_LIST]) + "\n")
        f.write("Mean Δluma: " + ",".join([f"{x:.6f}" for x in mean_delta]) + "\n")
        f.write("95% CI: " + ",".join([f"{x:.6f}" for x in ci95]) + "\n")
        f.write(f"Plot: {plot_path}\n")
        f.write("=" * 60 + "\n")

    print(f"Saved to: {out_dir}")
    print(f"✅ Summary saved: {txt_path}")

# ==========================================
# Helper Functions
# ==========================================
def _replace_unet_conv_in(unet):
    from torch.nn import Conv2d
    from torch.nn.parameter import Parameter
    _weight = unet.conv_in.weight.clone().repeat((1, 2, 1, 1))
    _new_conv_in = Conv2d(8, unet.conv_in.out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(unet.conv_in.bias.clone())
    unet.conv_in = _new_conv_in
    unet.config["in_channels"] = 8

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.manual_seed(seed)

def _calculate_psnr(img1, img2):
    target_size = (512, 512)
    # 確保兩者都是 PIL 且轉為 RGB
    im1 = np.array(img1.convert("RGB").resize(target_size, Image.BILINEAR)).astype(np.float64)
    im2 = np.array(img2.convert("RGB").resize(target_size, Image.BILINEAR)).astype(np.float64)
    mse = np.mean((im1 - im2) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))
    
def _calculate_ssim(img1, img2):
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)
    arr1 = np.array(img1.convert('L'))
    arr2 = np.array(img2.convert('L'))
    return ssim(arr1, arr2, data_range=255)

def _calculate_niqe(img_pil):
    """計算單張圖片的 NIQE (越低越自然)"""
    niqe_val = 0.0
    if niqe_metric is not None:
        # 將 PIL 轉為 Tensor (範圍 0~1) 並搬到 GPU
        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to('cuda')
        with torch.no_grad():
            niqe_val = niqe_metric(img_tensor).item()
    return niqe_val

def _calculate_brisque(img_pil):
    img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to('cuda')
    with torch.no_grad():
        return brisque_metric(img_tensor).item()
    
def _calculate_lpips(img1, img2):
    """計算感知相似度，使用 ori_pil 作為基準"""
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)
    t1 = transforms.ToTensor()(img1).unsqueeze(0).to('cuda')
    t2 = transforms.ToTensor()(img2).unsqueeze(0).to('cuda')
    with torch.no_grad():
        return lpips_metric(t1, t2).item()

def _prepare_batch_data(ctx, item, args, idx, override_color=None, override_intensity=None):
    """
    資料準備函數
    [Update] 支援 override_color 與 override_intensity，用於 Multicond Sweep。
    [Update] 加入靜音邏輯：只有在 single 模式下才儲存 debug 圖。
    """
    ori = item['image']
    # Auto Exposure: 當圖片太暗的時候自動提升亮度，讓 albedo estimator 有更好的輸入（這對於某些特別暗的圖很重要）
    if getattr(args, "auto_exposure", True):
        thr = float(getattr(args, "auto_exposure_thr", 0.25))
        target = float(getattr(args, "auto_exposure_target", 0.35))
        max_gain = float(getattr(args, "auto_exposure_max_gain", 4.0))
        ori, ae_gain, ae_mean_l = _auto_exposure_pil(ori, thr=thr, target=target, max_gain=max_gain)
        if getattr(args, "task", "") == "single":
            print(f"[AutoExposure] img={idx} mean_luma={ae_mean_l:.3f} gain={ae_gain:.2f}")
            
    normal = item['normal']
    depth = item['depth'].convert('L')
    
    if args.mode == 'lightlab' and 'mask_path' in item:
        mask = Image.open(item['mask_path']).convert('L')
    elif args.task == 'custom_test':
        mask_path = f"/mnt/HDD3/miayan/paper/relighting_datasets/lsun/{idx}_mask.png"
        if os.path.exists(mask_path):
            print(f"Loading mask from {mask_path}")
            mask = Image.open(mask_path).convert('L')
        else:
            mask = item['mask'].convert('L')
    else:
        mask = item['mask'].convert('L')

    # 設定 Parameters (支援 Override)
    if override_intensity is not None:
        intensity = override_intensity
    else:
        intensity = 1.0
        
    if override_color is not None:
        # override_color 預期是 list/tuple [R,G,B] 0-255
        color = torch.tensor([c/255.0 for c in override_color], dtype=torch.float32)
    else:
        color = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32) # Default White

    manual_ambient = 0.75

    # Compute Relighting (Physics API)
    cfg_cls = ctx['api'].PhysicalRelightingConfig
    compute_fn = ctx['api'].compute_relighting
    
    p_cfg = cfg_cls(ori, normal, depth)
    p_cfg.add_mask(mask, color, intensity)
    res = compute_fn(p_cfg, manual_ambient)
    
    # -----------------------------------------------------
    # [VISUALIZATION] 靜音邏輯：只有在 single 模式下才存圖
    # -----------------------------------------------------
    # should_save_debug = (args.task == 'single')
    should_save_debug = False

    lightmap_rgb = None
    res_h = ctx['resolution']
    
    if ctx['returns_ambient']:
        ambient_val = res['ambient']
        lightmap_rgb = res['lightmap_rgb']
        if should_save_debug:
            lightmap_rgb.save(f"./inference/{args.version}/debug_illumination_{idx}_{args.seed}.png")
    else:
        ambient_val = getattr(p_cfg, 'ambient', 0.75)
    
    # 根據 Config 選擇正確的 ControlNet Lightmap 來源
    # 這確保了當 override_color 改變時，ControlNet 也能拿到變色的圖
    if ctx['use_color_on_lightmap']:
         # Ex14+ (黑底彩色)，優先抓 raw_rgb
        if 'lightmap_raw_rgb' in res:
             lightmap = res['lightmap_raw_rgb'].convert('RGB')
        elif 'lightmap_rgb' in res:
             lightmap = res['lightmap_rgb'].convert('RGB')
        else:
             lightmap = res['lightmap'].convert('RGB')
    else:
        # Ex10_1 (黑底黑白) 或 Ex8_10 (灰底黑白)
        lightmap = res['lightmap'].convert('RGB')

    if should_save_debug:
        lightmap.save(f"./inference/{args.version}/debug_lightmap_{idx}_{args.seed}.png")
    # -----------------------------------------------------

    target_pil = res['image'] 

    # Albedo Estimation Debug
    if ctx['albedo_wrapper'] and not args.use_pretrained_alb:
        with torch.no_grad():
            ori_t = ctx['img_transform'](ori).unsqueeze(0).to('cuda')
            pred_alb = ctx['albedo_wrapper'](ori_t)
            
            if should_save_debug:
                debug_alb_img = to_pil_image(pred_alb.squeeze(0).cpu().clamp(0, 1))
                debug_alb_img.save(f"./inference/{args.version}/debug_albedo_{idx}_{args.seed}.png")
                
                if lightmap_rgb is not None:
                    l_rgb_resized = lightmap_rgb.resize((res_h, res_h), Image.BILINEAR)
                    illum_tensor = transforms.ToTensor()(l_rgb_resized).to('cuda')
                    phys_recon_tensor = pred_alb * illum_tensor
                    phy_recon = to_pil_image(phys_recon_tensor.squeeze(0).clamp(0, 1).cpu())
                    phy_recon.save(f"./inference/{args.version}/debug_phys_recon_{idx}_{args.seed}.png")
            
            albedo_t = torch.clamp((pred_alb.cpu() * 2.0) - 1.0, -1.0, 1.0)
    else:
        albedo_t = ctx['img_transform'](item['albedo']).unsqueeze(0)

    # Condition Tensors
    norm_t = ctx['cond_transform'](normal)
    dep_t = ctx['cond_transform'](depth)
    mask_t = ctx['cond_transform'](mask)
    map_t = ctx['cond_transform'](lightmap)

    # ControlNet Condition Packing
    if args.version in ctx['COND_CONFIG']['lightmap']:
        cn_cond = (norm_t, map_t)
    else:
        dm = dep_t * mask_t
        cn_cond = (norm_t, dep_t, mask_t, dm)

    return {
        "ori_pil": ori,
        "target_pil": target_pil,
        "albedo": albedo_t,
        "intensity": torch.tensor([intensity]).float().unsqueeze(0),
        "color": color.float().unsqueeze(0),
        "ambient": torch.tensor([ambient_val]).float().unsqueeze(0),
        "controlnet_cond": cn_cond,
        "mask": mask,
    }
     
def _prepare_bigtime_data(ctx, item, args, gt_config):
    ori = item['image'].convert('RGB')
    # Auto Exposure: 當圖片太暗的時候自動提升亮度，讓 albedo estimator 有更好的輸入（這對於某些特別暗的圖很重要）
    if getattr(args, "auto_exposure", True):
        thr = float(getattr(args, "auto_exposure_thr", 0.25))
        target = float(getattr(args, "auto_exposure_target", 0.5))
        max_gain = float(getattr(args, "auto_exposure_max_gain", 4.0))
        ori, ae_gain, ae_mean_l = _auto_exposure_pil(ori, thr=thr, target=target, max_gain=max_gain)
        if ae_gain > 1.0:
            ori.save(f"./inference/{args.version}/bigtime_0100_autoexp/debug_autoexposure_ori_{gt_config['input_id']}.png")
        if getattr(args, "task", "") == "single":
            print(f"[AutoExposure] img={idx} mean_luma={ae_mean_l:.3f} gain={ae_gain:.2f}")
                
    normal = item['normal']
    depth = item['depth'].convert('L')
    mask = item['mask'].convert('L')
    gt_image = item[gt_config['gt_col']].convert('RGB')
    
    intensity = gt_config.get('gt_intensity', 1.0)
    color = torch.tensor(gt_config.get('gt_color', [255, 255, 255]), dtype=torch.float32)/255.0
    manual_ambient = gt_config.get('gt_amb', 0.75)
    
    # -----------------------------------------------------
    # Compute Relighting (Physics API)
    # -----------------------------------------------------
    cfg_cls = ctx['api'].PhysicalRelightingConfig
    compute_fn = ctx['api'].compute_relighting
    
    p_cfg = cfg_cls(ori, normal, depth)
    p_cfg.add_mask(mask, color, intensity)
    res = compute_fn(p_cfg, manual_ambient)
    pseudo_gt = res['image'].convert('RGB')
    pseudo_gt.save(f"./inference/{args.version}/bigtime_0100_autoexp/debug_relighting_result_{gt_config['input_id']}.png")
    
    lightmap_rgb = None
    res_h = ctx['resolution']
    
    if ctx['returns_ambient']:
        ambient_val = res['ambient']
        lightmap_rgb = res['lightmap_rgb']
    else:
        ambient_val = getattr(p_cfg, 'ambient', 0.75)
        
    if ctx['use_color_on_lightmap']:
         # Ex14+ (黑底彩色)，優先抓 raw_rgb
        if 'lightmap_raw_rgb' in res:
             lightmap = res['lightmap_raw_rgb'].convert('RGB')
        elif 'lightmap_rgb' in res:
             lightmap = res['lightmap_rgb'].convert('RGB')
        else:
             lightmap = res['lightmap'].convert('RGB')
    else:
        # Ex10_1 (黑底黑白) 或 Ex8_10 (灰底黑白)
        lightmap = res['lightmap'].convert('RGB')
    
    # -----------------------------------------------------
    # Albedo Estimation
    # -----------------------------------------------------
    if ctx['albedo_wrapper'] and not args.use_pretrained_alb:
        with torch.no_grad():
            ori_t = ctx['img_transform'](ori).unsqueeze(0).to('cuda')
            pred_alb = ctx['albedo_wrapper'](ori_t)
            albedo_t = torch.clamp((pred_alb.cpu() * 2.0) - 1.0, -1.0, 1.0)
    else:
        albedo_t = ctx['img_transform'](item['albedo']).unsqueeze(0)
        
    # Condition Tensors
    norm_t = ctx['cond_transform'](normal)
    dep_t = ctx['cond_transform'](depth)
    mask_t = ctx['cond_transform'](mask)
    map_t = ctx['cond_transform'](lightmap)
    
    # ControlNet Condition Packing
    if args.version in ctx['COND_CONFIG']['lightmap']:
        cn_cond = (norm_t, map_t)
    else:
        dm = dep_t * mask_t
        cn_cond = (norm_t, dep_t, mask_t, dm)

    return {
        "ori_pil": ori,
        "target_pil": pseudo_gt,
        "albedo": albedo_t,
        "intensity": torch.tensor([intensity]).float().unsqueeze(0),
        "color": color.float().unsqueeze(0),
        "ambient": torch.tensor([ambient_val]).float().unsqueeze(0),
        "controlnet_cond": cn_cond,
        "gt_image": gt_image
    }
    
def _run_inference_internal(ctx, data, args, item=None):
    """ 內部 Helper: 負責準備 Latents 並呼叫 Pipeline """
    albedo_latents = ctx['vae'].encode(data['albedo'].to(dtype=torch.float32).cuda()).latent_dist.sample() * ctx['vae'].config.scaling_factor
    albedo_noise = torch.randn_like(albedo_latents)
    timesteps_albedo = torch.randint(50, 51, (1,))  # 50/200
    albedo_latents_noisy = ctx['scheduler'].add_noise(albedo_latents, albedo_noise, timesteps_albedo)

    generator = _set_seed(args.seed)
    pipe_kwargs = {
        "intensity": data['intensity'],
        "color": data['color'],
        "num_inference_steps": 50, # 50/20
        "generator": generator,
        "image": data['controlnet_cond'],
        "albedo_latents": albedo_latents_noisy.cuda(),
        "prompt": item['prompt'] if args.version in ctx['COND_CONFIG']['prompt'] and item is not None else None,
    }
    if ctx['use_ambient_cond']:
        pipe_kwargs["ambient"] = data['ambient']

    if args.guidance_scale > 0:
        target_tensor = ctx['img_transform'](data['ori_pil']).unsqueeze(0)
        images = guided_generation(ctx, pipe_kwargs, target_tensor, guidance_scale=args.guidance_scale)
        return images[0]
    else:
        pipe_output, _, _ = ctx['pipe'](**pipe_kwargs)
        return pipe_output.images[0]

def _lab_target_ab_from_rgb(color_rgb, device):
    c = torch.tensor(color_rgb, dtype=torch.float32) / 255.0
    target_rgb_np = c.numpy().reshape(1, 1, 3).astype(np.float32)
    target_lab = skcolor.rgb2lab(target_rgb_np)
    target_ab = torch.tensor(target_lab[0, 0, 1:3], dtype=torch.float32, device=device)
    return target_ab

def _auto_exposure_pil(img_pil, thr=0.25, target=0.5, max_gain=6.0):
    arr = np.asarray(img_pil.convert("RGB")).astype(np.float32) / 255.0
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

    # ⭐ 用 median 而不是 mean
    median_l = float(np.percentile(luma, 50))

    if median_l >= thr:
        return img_pil, 1.0, median_l

    gain = min(max_gain, target / (median_l + 1e-8))

    arr2 = np.clip(arr * gain, 0.0, 1.0)
    out = Image.fromarray((arr2 * 255.0).round().astype(np.uint8))
    return out, float(gain), median_l

def save_inference_visuals(out_dir, suffix, input_pil, output_pil, target_pil, mask_pil=None):

    imgs = [
        ("Input", input_pil),
        ("Output", output_pil),
        ("Target", target_pil),
    ]

    if mask_pil is not None:
        imgs.append(("Mask", mask_pil))

    # 統一尺寸
    w, h = output_pil.size
    imgs = [(name, img.resize((w, h))) for name, img in imgs]

    title_h = 30
    canvas = Image.new("RGB", (w * len(imgs), h + title_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i, (name, img) in enumerate(imgs):

        x = i * w

        canvas.paste(img.convert("RGB"), (x, title_h))

        # 畫文字
        bbox = draw.textbbox((0,0), name, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + w//2 - tw//2, 5), name, fill=(0,0,0), font=font)

    canvas.save(f"{out_dir}/compare_{suffix}.png")

# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ver', '--version', type=str, required=True, help="Experiment version (e.g., train_ex8)")
    parser.add_argument('-mode', '--mode', type=str, default='standard', choices=['standard', 'lightlab', 'bigtime'])
    
    # [修改] 選項：multicond (多顏色/強度), benchmark (Top 1%), single (單張)
    parser.add_argument('-task', '--task', type=str, default='single', 
                        choices=['single', 'multicond', 'benchmark', 'custom_test'], 
                        help="'single': runs 1 image. 'multicond': sweeps colors/intensities. 'benchmark': saves top 1% dataset. 'custom_test': runs specific image IDs with fixed color list.")
    
    parser.add_argument('-data', '--data', type=int, default=3, help="Number of data items to process")
    parser.add_argument('-data_start', '--data_start', type=int, default=0, help="Starting index of data items to process")
    parser.add_argument('-seed', '--seed', type=int, default=6071)
    parser.add_argument('-pp', '--print_process', action='store_true')
    parser.add_argument('-gs', '--guidance_scale', type=float, default=0.0, help="SSIM Guidance Scale")
    
    # [新增] Auto Exposure 選項
    parser.add_argument('--auto_exposure', action='store_true')
    parser.add_argument("--auto_exposure_thr", type=float, default=0.25, help="Auto exposure threshold for mean luma (0-1)")
    parser.add_argument("--auto_exposure_target", type=float, default=0.35, help="Auto exposure target mean luma (0-1)")
    parser.add_argument("--auto_exposure_max_gain", type=float, default=4.0, help="Auto exposure max gain limit")
    
    # [新增] Benchmark 模式專用選項
    # Phase1
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--topk_csv_name", type=str, default="topk_quality.csv")

    # Phase2 shared
    parser.add_argument("--phase2_max_items", type=int, default=None)  # e.g. 100
    # Phase2 color
    parser.add_argument("--control_colors", type=str, default="Red,Green,Blue")
    parser.add_argument("--control_intensity_for_color", type=float, default=1.0)
    parser.add_argument("--phase2_hf_name", type=str, default="phase2_color_hf")

    # Phase2 intensity
    parser.add_argument("--control_intensity_sweep", type=str, default="0.0,0.2,0.5,0.8,1.0")
    parser.add_argument("--phase2_intensity_csv", type=str, default="phase2_intensity_per_image.csv")
    parser.add_argument("--phase2_intensity_curve_csv", type=str, default="phase2_intensity_curve.csv")
    
    # [新增] BigTime 模式專用選項
    parser.add_argument('--use_pretrained_alb', action='store_true')
    
    # [新增] Custom Test 模式專用選項
    parser.add_argument("--test_ids", type=str, default="23,24,26,29,46,59,73,81")
    # parser.add_argument("--test_ids", type=str, default="66,202725,218817,186980,125466,52308,69843,133576,67014,88258,209463,5,8,48,39")

    # optional HF upload
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_repo_name", type=str, default='Miayan/test-visualize')
    args = parser.parse_args()

    # 1. Setup
    ctx = setup_pipeline(args)
    
    # 2. Load Dataset
    repo = "Miayan/physical-relighting-dataset"
    if args.mode == 'lightlab': repo = "Miayan/physical-relighting-eval-dataset"
    elif args.mode =='bigtime': repo = "Miayan/test-bigtime"
    
    print(f"Loading Dataset: {repo}")
    ds = load_dataset(repo, split="train", cache_dir="/mnt/HDD3/miayan/paper/relighting_datasets/")
    
    for col in ds.column_names:
        if col not in ('color', 'intensity', 'prompt'):
            ds = ds.cast_column(col, HfImage(decode=True))
            
    # Determine indices
    if args.mode == 'lightlab':
        ids = [181, 13, 75, 77] 
        # ids = [181, 13, 75, 77, 8, 95] 
        mask_paths = [
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/0_0.png',
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/1_0.png',
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/2_0.png',
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/3_0.png'
        ]
        indices = ids[:args.data]
        
        mapped_dataset = []
        for i, idx in enumerate(indices):
            item = ds[idx]
            item['mask_path'] = mask_paths[i]
            mapped_dataset.append(item)
        indices = range(len(mapped_dataset)) 
        dataset_to_use = mapped_dataset
    elif args.mode == 'bigtime':
        print("BigTime mode: Using all data without sampling.")
        
        GT_CONFIGS = [
            {"input_id": 0, "gt_col": "gt_0001", "gt_color": [255, 255, 255], "gt_intensity": 1.0, "gt_amb": 0.25},
            {"input_id": 0, "gt_col": "gt_0002", "gt_color": [255, 255, 255], "gt_intensity": 0.7, "gt_amb": 0.25},
            {"input_id": 0, "gt_col": "gt_0003", "gt_color": [255, 255, 255], "gt_intensity": 0.2, "gt_amb": 0.1},
            {"input_id": 1, "gt_col": "gt_0001", "gt_color": [195, 187, 74], "gt_intensity": 3.0, "gt_amb": 0.5},
            {"input_id": 2, "gt_col": "gt_0001", "gt_color": [0, 133, 200], "gt_intensity": 3.0, "gt_amb": 0.6},
            {"input_id": 2, "gt_col": "gt_0002", "gt_color": [255, 255, 0], "gt_intensity": 1.0, "gt_amb": 0.6},
            {"input_id": 3, "gt_col": "gt_0001", "gt_color": [255, 255, 255], "gt_intensity": 1.0, "gt_amb": 0.75},
            {"input_id": 4, "gt_col": "gt_0001", "gt_color": [255, 255, 255], "gt_intensity": 1.5, "gt_amb": 0.75},
            {"input_id": 4, "gt_col": "gt_0002", "gt_color": [255, 255, 255], "gt_intensity": 0.5, "gt_amb": 0.5},
            {"input_id": 5, "gt_col": "gt_0001", "gt_color": [255, 255, 255], "gt_intensity": 0.5, "gt_amb": 0.5},
            {"input_id": 5, "gt_col": "gt_0002", "gt_color": [255, 255, 255], "gt_intensity": 1.0, "gt_amb": 0.75},
            {"input_id": 6, "gt_col": "gt_0001", "gt_color": [255, 255, 0], "gt_intensity": 3.0, "gt_amb": 0.75},
            {"input_id": 7, "gt_col": "gt_0001", "gt_color": [248, 132, 52], "gt_intensity": 1.0, "gt_amb": 0.75},
            {"input_id": 7, "gt_col": "gt_0002", "gt_color": [248, 132, 52], "gt_intensity": 2.0, "gt_amb": 0.75},
            {"input_id": 8, "gt_col": "gt_0001", "gt_color": [255, 0, 255], "gt_intensity": 1.0, "gt_amb": 0.75},
            {"input_id": 9, "gt_col": "gt_0001", "gt_color": [0, 255, 255], "gt_intensity": 1.0, "gt_amb": 0.75},
            {"input_id": 10, "gt_col": "gt_0001", "gt_color": [255, 0, 255], "gt_intensity": 1.0, "gt_amb": 0.75},
        ]
        
        dataset_to_use = ds
        total_len = len(ds)
        run_bigtime_test(ctx, args, dataset_to_use, GT_CONFIGS[:args.data])
    else:
        total_len = len(ds)
        # Standard mode indices
        if args.task == 'benchmark':
            sample_size = min(args.data, total_len)

            print(f"[Benchmark] Randomly sampling {sample_size} images from {total_len}")
            print(f"[Benchmark] Using random seed: {args.seed}")

            # 固定 seed 讓結果可重現
            random.seed(args.seed)

            indices = random.sample(range(total_len), sample_size)
        elif args.task == 'single' and args.mode == 'standard':
            indices = range(args.data_start, min(args.data, total_len))
        else:
            sample_size = min(args.data, total_len)
            indices = range(sample_size)
        dataset_to_use = ds

    # 3. Dispatch Task
    if args.mode != 'bigtime':
        if args.task == 'single':
            run_single_inference(ctx, args, dataset_to_use, indices)
        elif args.task == 'multicond':
            # [New] Color/Intensity Sweep
            run_multicond_sweep(ctx, args, dataset_to_use, indices)
        elif args.task == 'benchmark':
            run_benchmark_three_phase(ctx, args, dataset_to_use, indices)
        elif args.task == 'custom_test':
            if not args.test_ids.strip():
                raise ValueError("custom_test requires --test_ids 12,34,56")
            image_ids = [int(x.strip()) for x in args.test_ids.split(",") if x.strip()]
            run_custom_id_color_test(ctx, args, dataset_to_use, image_ids)