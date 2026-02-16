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
    out_dir = f'./inference/{args.version}'
    os.makedirs(out_dir, exist_ok=True)

    for idx in indices:
        item = dataset[idx]
        data = _prepare_batch_data(ctx, item, args, idx)
        
        # 呼叫統一的推論邏輯
        image = _run_inference_internal(ctx, data, args, item)

        # 計算四項品質指標
        s_score = _calculate_ssim(image, data['ori_pil'])
        n_score = _calculate_niqe(image)
        b_score = _calculate_brisque(image)
        l_score = _calculate_lpips(image, data['ori_pil'])
        
        # 儲存結果與對照圖
        suffix = f"{idx}_{args.seed}"
        if args.guidance_scale > 0:
            suffix += f"_gs{args.guidance_scale}"
        
        image.save(f"{out_dir}/output_{suffix}.png")
        data['ori_pil'].save(f"{out_dir}/ori_{suffix}.png")
        data['target_pil'].save(f"{out_dir}/target_{suffix}.png")
        
        print(f"[Single] ID {idx} | SSIM: {s_score:.4f} | NIQE: {n_score:.4f} | BRISQUE: {b_score:.4f} | LPIPS: {l_score:.4f}")
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
        p_input = control_tool.get_chromaticity_vector(item['image'])

        # --- 第一階段：Color Sweep (計算品質指標 + CA) ---
        print("\n[Phase 1] Sweeping Colors...")
        delta_p_list, target_p_list = [], []
        
        for c_name, c_val in COLORS.items():
            data = _prepare_batch_data(ctx, item, args, idx, override_color=c_val, override_intensity=1.0)
            image = _run_inference_internal(ctx, data, args, item)
            
            # (1) 原本的品質指標
            s = _calculate_ssim(image, data['ori_pil'])
            n = _calculate_niqe(image)
            b = _calculate_brisque(image)
            l = _calculate_lpips(image, data['ori_pil'])
            
            # (2) CA 指標相關紀錄
            p_relit = control_tool.get_chromaticity_vector(image)
            delta_p_list.append(p_relit - p_input)
            c_t = torch.tensor(c_val, dtype=torch.float32) / 255.0  # (3,)
            target_rgb = c_t.cpu().numpy().reshape(1, 1, 3).astype(np.float32)
            target_lab = skcolor.rgb2lab(target_rgb)  # (1,1,3)
            target_ab = torch.tensor(target_lab[0, 0, 1:3], dtype=torch.float32, device=p_input.device)  # (2,)
            target_p_list.append(target_ab - p_input)
            
            ca_single = F.cosine_similarity((p_relit - p_input).unsqueeze(0), (target_ab - p_input).unsqueeze(0),).item()

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
    if getattr(args, "bench_subset", None) is not None:
        indices = indices[:args.bench_subset]
        print(f"[Phase1] Using subset: {len(indices)} images")

    top_k = int(getattr(args, "top_k", 100))
    csv_name = getattr(args, "topk_csv_name", "topk_quality.csv")

    print(f"[Phase1] Selecting Top {top_k} by SSIM (tie-break: NIQE lower)")

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
    print(f"  SSIM   : {total['ssim']/count:.6f}")
    print(f"  NIQE   : {total['niqe']/count:.6f}")
    print(f"  BRISQUE: {total['brisque']/count:.6f}")
    print(f"  LPIPS  : {total['lpips']/count:.6f}")

    top_results = sorted(top_heap, key=lambda x: (x[0], x[1]), reverse=True)
    csv_path = _save_topk_quality_csv(out_dir, top_results, filename=csv_name)

    return csv_path

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
        p_input = control_tool.get_chromaticity_vector(input_pil, None)
        dev = p_input.device

        for cname in color_names:
            if cname not in COLORS:
                continue
            rgb = COLORS[cname]

            data = _prepare_batch_data(ctx, item, args, idx,
                                       override_color=rgb,
                                       override_intensity=float(intensity))
            relit_pil = _run_inference_internal(ctx, data, args, item)

            p_relit = control_tool.get_chromaticity_vector(relit_pil, None)
            target_ab = _lab_target_ab_from_rgb(rgb, dev)

            delta_p = p_relit - p_input
            target_dir = target_ab - p_input

            ca = F.cosine_similarity(delta_p.unsqueeze(0), target_dir.unsqueeze(0), dim=1).item()

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
    for cname in color_names:
        vals = [v for (n, v) in zip(records["color_name"], records["color_score"]) if n == cname]
        if len(vals):
            print(f"[Phase2-Color] Mean CA ({cname}) = {float(np.mean(vals)):.4f}")

    return save_path

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
    out_dir = f'./inference/{args.version}/benchmark'
    os.makedirs(out_dir, exist_ok=True)

    # Phase 1
    topk_csv_path = run_benchmark_phase1_quality(ctx, args, dataset, indices, out_dir)

    # Phase 2A: Color -> HF dataset
    run_benchmark_phase2_color_to_hf(ctx, args, dataset, topk_csv_path, out_dir)

    # Phase 2B: Intensity -> per-image CSV + mean curve CSV
    run_benchmark_phase2_intensity(ctx, args, dataset, topk_csv_path, out_dir)

    print("\n[Benchmark] Done (Phase1 + Phase2-Color + Phase2-Intensity).")

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
    normal = item['normal']
    depth = item['depth'].convert('L')
    
    if args.mode == 'lightlab' and 'mask_path' in item:
         mask = Image.open(item['mask_path']).convert('L')
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
        color = torch.tensor([0, 1.0, 0], dtype=torch.float32) # Default Green

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
    should_save_debug = (args.task == 'single')

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
    if ctx['albedo_wrapper']:
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
        "controlnet_cond": cn_cond
    }
    
def _run_inference_internal(ctx, data, args, item):
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
        "prompt": item['prompt'] if args.version in ctx['COND_CONFIG']['prompt'] else None,
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

# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ver', '--version', type=str, required=True, help="Experiment version (e.g., train_ex8)")
    parser.add_argument('-mode', '--mode', type=str, default='standard', choices=['standard', 'lightlab'])
    
    # [修改] 選項：multicond (多顏色/強度), benchmark (Top 1%), single (單張)
    parser.add_argument('-task', '--task', type=str, default='single', 
                        choices=['single', 'multicond', 'benchmark'], 
                        help="'single': runs 1 image. 'multicond': sweeps colors/intensities. 'benchmark': saves top 1% dataset.")
    
    parser.add_argument('-data', '--data', type=int, default=3, help="Number of data items to process")
    parser.add_argument('-seed', '--seed', type=int, default=6071)
    parser.add_argument('-pp', '--print_process', action='store_true')
    parser.add_argument('-gs', '--guidance_scale', type=float, default=0.0, help="SSIM Guidance Scale")
    
    # [新增] Benchmark 模式專用選項
    # Phase1
    parser.add_argument("--bench_subset", type=int, default=None)
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

    # optional HF upload
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_repo_name", type=str, default=None)
    args = parser.parse_args()

    # 1. Setup
    ctx = setup_pipeline(args)
    
    # 2. Load Dataset
    repo = "Miayan/physical-relighting-dataset"
    if args.mode == 'lightlab': repo = "Miayan/physical-relighting-eval-dataset"
    
    print(f"Loading Dataset: {repo}")
    ds = load_dataset(repo, split="train", cache_dir="/mnt/HDD3/miayan/paper/relighting_datasets/")
    
    for col in ds.column_names:
        if col not in ('color', 'intensity', 'prompt'):
            ds = ds.cast_column(col, HfImage(decode=True))
            
    # Determine indices
    if args.mode == 'lightlab':
        ids = [181, 16, 75, 77] 
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
    else:
        # Standard mode indices
        if args.task == 'benchmark':
            limit = min(args.data, len(ds))
            indices = range(limit)
        else:
            indices = range(args.data)
        dataset_to_use = ds

    # 3. Dispatch Task
    if args.task == 'single':
        run_single_inference(ctx, args, dataset_to_use, indices)
    elif args.task == 'multicond':
        # [New] Color/Intensity Sweep
        run_multicond_sweep(ctx, args, dataset_to_use, indices)
    elif args.task == 'benchmark':
        run_benchmark_three_phase(ctx, args, dataset_to_use, indices)