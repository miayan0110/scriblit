import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import heapq
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from huggingface_hub import snapshot_download
from natsort import natsorted
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim

# Custom Imports
from network_controlnet import ControlNetModel
from datasets import load_dataset, Image as HfImage, Dataset

# Relighting API Imports
import physical_relighting_api_v2 as relighting_api_v2
import physical_relighting_api_v2_gemini as relighting_api_gemini
import physical_relighting_api_v2_gemini_amb as relighting_api_gemini_amb

# Encoders
from light_cond_encoder import CustomEncoder as CustomEncoderV1
from light_cond_encoder_amb import CustomEncoder as CustomEncoderAmb
from albedo_estimator import AlbedoWrapper


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

    for i, idx in enumerate(indices):
        item = dataset[idx]
        data = _prepare_batch_data(ctx, item, args, idx)
        
        # 準備 Albedo Latents (加入部分 Noise)
        albedo_latents = ctx['vae'].encode(data['albedo'].to(dtype=torch.float32).cuda()).latent_dist.sample() * ctx['vae'].config.scaling_factor
        albedo_noise = torch.randn_like(albedo_latents)
        timesteps_albedo = torch.randint(200, 201, (1,))
        albedo_latents_noisy = ctx['scheduler'].add_noise(albedo_latents, albedo_noise, timesteps_albedo)

        generator = _set_seed(args.seed)
        pipe_kwargs = {
            "intensity": data['intensity'],
            "color": data['color'],
            "num_inference_steps": 20, 
            "generator": generator,
            "image": data['controlnet_cond'],
            "albedo_latents": albedo_latents_noisy.cuda(),
            "prompt": item['prompt'] if args.version in ctx['COND_CONFIG']['prompt'] else None,
        }
        if ctx['use_ambient_cond']:
            pipe_kwargs["ambient"] = data['ambient']

        # 執行 Inference (Guided 或 Standard)
        if args.guidance_scale > 0:
            target_tensor = ctx['img_transform'](data['ori_pil']).unsqueeze(0)
            images = guided_generation(ctx, pipe_kwargs, target_tensor, guidance_scale=args.guidance_scale)
            image = images[0]
        else:
            pipe_output, _, _ = ctx['pipe'](**pipe_kwargs)
            image = pipe_output.images[0]

        ssim_score = _calculate_ssim(image, data['ori_pil'])
        
        # 存檔
        suffix = f"{idx}_{args.seed}"
        if args.guidance_scale > 0: suffix += f"_gs{args.guidance_scale}"
        
        image.save(f"{out_dir}/output_{suffix}.png")
        data['ori_pil'].save(f"{out_dir}/ori_{suffix}.png")
        data['target_pil'].save(f"{out_dir}/target_{suffix}.png")
        
        print(f"  [Single] Processed {idx} | SSIM: {ssim_score:.4f} | Saved to {out_dir}")

# ==========================================
# 3. [New] Multi-Condition Sweep (Colors & Intensities)
# 說明：針對同一張圖，自動測試多種顏色與光強，並確保 ControlNet Condition 也跟著改變。
# ==========================================
def run_multicond_sweep(ctx, args, dataset, indices):
    print("--- Mode: Multi-Condition Sweep (Colors & Intensities) ---")
    
    # 定義要測試的顏色 (RGB 0-255)
    COLORS = {
        "White": [255, 255, 255],
        "Red":   [255, 0, 0],
        "Green": [0, 255, 0],
        "Blue":  [0, 0, 255],
        "Warm":  [255, 200, 150],
        "Cool":  [150, 200, 255]
    }
    
    # 定義要測試的強度
    INTENSITIES = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
    
    out_dir = f'./inference/{args.version}/sweep_multicond'
    os.makedirs(out_dir, exist_ok=True)
    
    for i, idx in enumerate(indices):
        item = dataset[idx]
        print(f"\nProcessing Image ID: {idx}")
        
        # 1. 顏色 Sweep (固定 Intensity = 1.0)
        print("  > Sweeping Colors...")
        for c_name, c_val in COLORS.items():
            # 使用 _prepare_batch_data 覆寫顏色，確保 ControlNet Lightmap 重算
            data = _prepare_batch_data(ctx, item, args, idx, override_color=c_val, override_intensity=1.0)
            
            image = _run_inference_internal(ctx, data, args, item)
            
            fname = f"img{idx}_color_{c_name}.png"
            image.save(os.path.join(out_dir, fname))
            print(f"    Saved {fname}")

        # 2. 強度 Sweep (固定 Color = White)
        print("  > Sweeping Intensities...")
        for ints in INTENSITIES:
            data = _prepare_batch_data(ctx, item, args, idx, override_color=COLORS["White"], override_intensity=ints)
            
            image = _run_inference_internal(ctx, data, args, item)
            
            fname = f"img{idx}_intensity_{ints}.png"
            image.save(os.path.join(out_dir, fname))
            print(f"    Saved {fname}")
            
    print(f"\nSweep Completed. Results saved to {out_dir}")

def _run_inference_internal(ctx, data, args, item):
    """ 內部 Helper: 負責準備 Latents 並呼叫 Pipeline """
    albedo_latents = ctx['vae'].encode(data['albedo'].to(dtype=torch.float32).cuda()).latent_dist.sample() * ctx['vae'].config.scaling_factor
    albedo_noise = torch.randn_like(albedo_latents)
    timesteps_albedo = torch.randint(200, 201, (1,))
    albedo_latents_noisy = ctx['scheduler'].add_noise(albedo_latents, albedo_noise, timesteps_albedo)

    generator = _set_seed(args.seed)
    pipe_kwargs = {
        "intensity": data['intensity'],
        "color": data['color'],
        "num_inference_steps": 20, 
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


# ==========================================
# 4. Benchmark 模式 (Top 1% SSIM)
# 說明：
#   跑大量數據，但不儲存中間圖片（節省空間）。
#   只保留 SSIM 分數最高的 Top 1% 結果，並存成 HuggingFace Dataset。
#   使用 Min-Heap 演算法來有效率地篩選 Top K。
# ==========================================
def run_benchmark_inference(ctx, args, dataset, indices):
    print(f"\n=== Benchmark Mode ===")
    print(f"Total Images: {len(indices)}")
    
    top_k_count = max(1, int(len(indices) * 0.01))
    print(f"Goal: Save Top {top_k_count} best images (Top 1%) to Dataset.")

    out_dir = f'./inference/{args.version}/benchmark'
    os.makedirs(out_dir, exist_ok=True)

    # 初始化 Min-Heap (用來維持 Top K)
    # 格式: (ssim, idx, image_pil, target_pil)
    # 因為是 Min-Heap，heap[0] 永遠是這 K 個裡面分數最低的 (門檻值)
    top_k_heap = [] 
    total_ssim = 0.0
    processed_count = 0
    
    pbar = tqdm(indices, desc="Benchmarking")
    for idx in pbar:
        item = dataset[idx]
        data = _prepare_batch_data(ctx, item, args, idx)
        
        image = _run_inference_internal(ctx, data, args, item)

        score = _calculate_ssim(image, data['ori_pil'])
        total_ssim += score
        processed_count += 1
        
        # Top K 篩選邏輯
        if len(top_k_heap) < top_k_count:
            heapq.heappush(top_k_heap, (score, idx, image, data['ori_pil']))
        else:
            # 如果 Heap 滿了，檢查這張圖有沒有比 Heap 裡最爛的那張好
            if score > top_k_heap[0][0]:
                heapq.heapreplace(top_k_heap, (score, idx, image, data['ori_pil']))
        
        min_top_score = top_k_heap[0][0] if top_k_heap else 0.0
        pbar.set_postfix({"Avg": f"{total_ssim/processed_count:.3f}", "Top1%": f"{min_top_score:.3f}"})

    # 整理結果
    final_avg_ssim = total_ssim / processed_count
    top_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
    
    print("\n" + "="*40)
    print(f"Benchmark Report")
    print(f"Total Processed: {processed_count}")
    print(f"Average SSIM:    {final_avg_ssim:.5f}")
    print(f"Best SSIM:       {top_results[0][0]:.5f}")
    print(f"Top 1% Cutoff:   {top_results[-1][0]:.5f}")
    print("="*40)
    
    # 建立 Dataset
    data_dict = {
        "index": [],
        "ssim": [],
        "image": [],
        "target": []
    }
    
    vis_dir = os.path.join(out_dir, "top_images")
    os.makedirs(vis_dir, exist_ok=True)
    
    for score, idx, img, tgt in top_results:
        data_dict["index"].append(idx)
        data_dict["ssim"].append(score)
        data_dict["image"].append(img)
        data_dict["target"].append(tgt)
        img.save(f"{vis_dir}/rank_{idx}_ssim{score:.3f}.png")

    hf_ds = Dataset.from_dict(data_dict)
    save_path = os.path.join(out_dir, "top_1percent_dataset")
    hf_ds.save_to_disk(save_path)
    
    print(f"Top {len(top_results)} images saved to: {vis_dir}")
    print(f"Full Dataset saved to: {save_path}")

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
        color = torch.tensor([0, 1.0, 0], dtype=torch.float32) # Default Blue

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
        indices = ids[:args.data+1]
        
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
            indices = range(args.data + 1)
        dataset_to_use = ds

    # 3. Dispatch Task
    if args.task == 'single':
        run_single_inference(ctx, args, dataset_to_use, indices)
    elif args.task == 'multicond':
        # [New] Color/Intensity Sweep
        run_multicond_sweep(ctx, args, dataset_to_use, indices)
    elif args.task == 'benchmark':
        run_benchmark_inference(ctx, args, dataset_to_use, indices)