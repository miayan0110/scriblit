import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
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

# Custom Imports
from network_controlnet import ControlNetModel
from datasets import load_dataset, Image as HfImage

# Relighting API Imports
import physical_relighting_api_v2 as relighting_api_v2
import physical_relighting_api_v2_gemini as relighting_api_gemini
import physical_relighting_api_v2_gemini_amb as relighting_api_gemini_amb

# Encoders
from light_cond_encoder import CustomEncoder as CustomEncoderV1
from light_cond_encoder_amb import CustomEncoder as CustomEncoderAmb
from albedo_estimator import AlbedoWrapper


# ==========================================
# [新增] Differentiable SSIM for Guidance
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
# 1. 通用設定與模型載入
# ==========================================
def setup_pipeline(args):
    print(f"--- [Setup] Initializing Pipeline for Version: {args.version} ---")
    
    if os.path.exists(args.version) and os.path.isdir(args.version):
        print(f"  > Found local experiment folder: {args.version}")
        experiment_dir = args.version
    else:
        print(f"  > Local folder '{args.version}' not found. Downloading from Hugging Face...")
        local_dir = snapshot_download(repo_id="Miayan/project", repo_type="model", allow_patterns=[f"{args.version}/*"])
        experiment_dir = os.path.join(local_dir, args.version)
    
    checkpoints = [f for f in os.listdir(experiment_dir) if f.startswith('checkpoint')]
    best_ckpt = natsorted(checkpoints)[-1]
    ckpt_path = os.path.join(experiment_dir, best_ckpt)
    print(f"  > Using checkpoint: {ckpt_path}")

    config_path = os.path.join(experiment_dir, "config.yaml")
    train_cfg = OmegaConf.load(config_path) if os.path.exists(config_path) else None
    
    use_ambient_cond = train_cfg.get('model', {}).get('enable_ambient_cond', False) if train_cfg else False
    relighting_impl = train_cfg.get('data', {}).get('relighting_impl', 'v2') if train_cfg else 'v2'
    use_pred_albedo = train_cfg.get('albedo_estimator', {}).get('enabled', False) if train_cfg else False
    
    resolution = train_cfg.get('data', {}).get('resolution', 512) if train_cfg else 512
    print(f"  > Using resolution: {resolution}x{resolution}")

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

    # Load Models
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

    # Note: 我們在 guided generation 中會手動執行，所以 pipe 只在非 guidance 模式下當作容器或備用
    if any(v in args.version for v in ['ex2_3', 'ex2_4', 'ex2_5']):
        from pipeline_cn_ex2_3 import CustomControlNetPipeline
    else:
        from pipeline_cn import CustomControlNetPipeline

    pipe = CustomControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, cond_encoder=cond_encoder, unet=unet, torch_dtype=torch.float32
    )
    # [重要] Guidance 需要保留模型在 GPU 上，不能隨意 offload
    # pipe.enable_model_cpu_offload() 
    pipe.to('cuda')

    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
    vae.to('cuda')
    
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

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
        "img_transform": img_transform,
        "cond_transform": cond_transform,
        "resolution": resolution,
        "COND_CONFIG": { 
            "lightmap": ['train_ex2_2', 'train_ex2_3', 'train_ex2_4', 'train_ex4', 'train_ex4_1', 'train_ex5', 
                         'train_ex5_1', 'train_ex6', 'train_ex6_1', 'train_ex6_2', 'train_ex6_3', 'train_ex6_4',
                         'train_ex6_5', 'train_ex6_6', 'train_ex6_7', 'train_ex6_8', 'train_ex6_9', 'train_ex6_10',
                         'train_ex6_11', 'train_ex6_12', 'train_ex6_13', 'train_ex7', 'train_ex8', 'train_ex8_1', 
                         'train_ex8_2', 'train_ex8_3', 'train_ex8_4', 'train_ex8_5', 'train_ex8_6', 'train_ex8_7',
                         'train_ex8_8', 'train_ex8_9', 'train_ex8_10', 'train_ex8_9'],
            "prompt": ['train_ex2_3', 'train_ex2_4', 'train_ex2_5']
        }
    }


# ==========================================
# [修正版 5.0] Double Normalized SSIM (The Robust Solution)
# ==========================================
def guided_generation(ctx, pipe_kwargs, target_img_tensor, guidance_scale=0.0):
    """
    手動執行的去噪迴圈，支援 SSIM Loss Guidance。
    ** 修正邏輯：
       使用「雙重歸一化 (Double Normalization)」。
       不強迫 Target 去適應 Pred，而是將兩者都投射到一個標準的
       分佈空間 (Mean=0.5, Std=0.2)，在這個公平的空間算 SSIM。
       這能同時保證：
       1. SSIM 分數高 (保留了中低頻結構)。
       2. 顏色完全不偏 (因為亮度和顏色資訊在歸一化時被拿掉了)。
    """
    scheduler = ctx['scheduler']
    unet = ctx['unet']
    controlnet = ctx['controlnet']
    cond_encoder = ctx['cond_encoder']
    vae = ctx['vae']
    
    # 1. Prepare Conditions
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

    # 2. Prepare Timesteps
    num_inference_steps = pipe_kwargs['num_inference_steps']
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    
    # 3. Prepare Latents
    albedo_latents = pipe_kwargs['albedo_latents'].clone().cuda() 
    
    generator = pipe_kwargs.get('generator', None)
    latents = torch.randn(albedo_latents.shape, generator=generator, device='cpu', dtype=albedo_latents.dtype).to('cuda')
    latents = latents * scheduler.init_noise_sigma
    
    # SSIM Loss Module
    ssim_criterion = SSIMLoss().cuda()
    
    # Target (Original Image)
    target_img_01 = (target_img_tensor.cuda() + 1.0) / 2.0
    
    # [Helper] Robust Normalization
    def robust_normalize(img_tensor):
        # 1. Mean Grayscale (B, 1, H, W)
        gray = img_tensor.mean(dim=1, keepdim=True)
        
        # 2. Calculate Stats
        mu = gray.mean(dim=(2, 3), keepdim=True)
        std = gray.std(dim=(2, 3), keepdim=True)
        
        # 3. Normalize to Zero-Mean, Unit-Variance
        # 加上 1e-5 防止除以 0
        normalized = (gray - mu) / (std + 1e-5)
        
        # 4. Remap to strict [0, 1] range for SSIM
        # 假設大部分像素在 +/- 2.5 std 內，我們將其縮放以適應 [0, 1]
        # Mean -> 0.5, Std -> 0.2
        remapped = normalized * 0.2 + 0.5
        
        # Clamp 掉極端值 (Highlights/Shadows)，避免數值爆炸
        return remapped.clamp(0.0, 1.0)

    # Pre-process target once (Standardized Structure)
    target_norm = robust_normalize(target_img_01)

    print(f"  > Starting Guided Inference (Scale: {guidance_scale}, Double Norm: True)...")
    
    for t in tqdm(timesteps, desc="Guided Sampling"):
        with torch.enable_grad():
            latents = latents.detach().requires_grad_(True)
            
            unet_input = torch.cat([latents, albedo_latents], dim=1)
            
            down_block_res_samples, mid_block_res_sample, _ = controlnet(
                latents, t, encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond, return_dict=False,
            )

            noise_pred = unet(
                unet_input, t, encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[sample.to(dtype=torch.float32) for sample in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float32),
                return_dict=False,
            )[0]
            
            # --- Guidance Step ---
            if guidance_scale > 0:
                step_output = scheduler.step(noise_pred, t, latents, return_dict=True)
                pred_x0_latents = step_output.pred_original_sample
                
                # Decode
                pred_x0_img = vae.decode(pred_x0_latents / vae.config.scaling_factor).sample
                pred_x0_img = (pred_x0_img + 1.0) / 2.0
                
                # [关键] Normalize Prediction to the SAME standard space
                pred_norm = robust_normalize(pred_x0_img)
                
                # Calculate SSIM on Normalized Maps
                loss = ssim_criterion(pred_norm, target_norm)
                
                grads = torch.autograd.grad(loss, latents)[0]
                
                # Optional: Clip gradients slightly to ensure stability
                # torch.nn.utils.clip_grad_norm_(grads, 1.0)
                
                latents = latents - guidance_scale * grads
                
        latents = latents.detach()
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    
    image = (image * 255).round().astype("uint8")
    return [Image.fromarray(image[0])]

# ==========================================
# 2. 單次推理模式 (Single Inference)
# ==========================================
def run_single_inference(ctx, args, dataset, indices):
    print("--- Mode: Single Inference ---")
    
    out_dir = f'./inference/{args.version}'
    os.makedirs(out_dir, exist_ok=True)
    if args.print_process:
        os.makedirs(f'{out_dir}/denoise', exist_ok=True)
        os.makedirs(f'{out_dir}/cal', exist_ok=True)

    for i, idx in enumerate(indices):
        item = dataset[idx]
        data = _prepare_batch_data(ctx, item, args, idx)
        
        # Prepare Albedo Latents
        albedo_latents = ctx['vae'].encode(data['albedo'].to(dtype=torch.float32).cuda()).latent_dist.sample() * ctx['vae'].config.scaling_factor
        albedo_noise = torch.randn_like(albedo_latents)
        timesteps_albedo = torch.randint(200, 201, (1,))
        albedo_latents_noisy = ctx['scheduler'].add_noise(albedo_latents, albedo_noise, timesteps_albedo)

        # Common Args
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

        # Dispatch
        if args.guidance_scale > 0:
            # Use Manual Guided Loop
            # Target for SSIM: Original Image (data['ori_pil'] -> Tensor)
            target_tensor = ctx['img_transform'](data['ori_pil']).unsqueeze(0)
            images = guided_generation(ctx, pipe_kwargs, target_tensor, guidance_scale=args.guidance_scale)
            image = images[0]
            denoise_lst, cal_lst = [], [] 
        else:
            # Use Standard Pipeline
            pipe_output, denoise_lst, cal_lst = ctx['pipe'](**pipe_kwargs)
            image = pipe_output.images[0]

        # [NEW] Calculate and Print SSIM
        ssim_score = _calculate_ssim(image, data['ori_pil'])
        
        # Save
        suffix = f"{idx}_{args.seed}"
        if args.guidance_scale > 0: suffix += f"_gs{args.guidance_scale}"
        
        image.save(f"{out_dir}/output_{suffix}.png")
        data['ori_pil'].save(f"{out_dir}/ori_{suffix}.png")
        data['target_pil'].save(f"{out_dir}/target_{suffix}.png")
        
        print(f"  [Single] Processed {idx} | SSIM: {ssim_score:.4f} | Saved to {out_dir}")

# ==========================================
# 3. 掃描實驗模式 (Sweep Inference)
# ==========================================
def run_sweep_inference(ctx, args, dataset, indices):
    print("--- Mode: Sweep Experiment ---")
    
    EXPERIMENT_CONFIG = {
        "albedo_noise_step": [20, 50, 100, 200],
        "num_inference_steps": [20, 50]
    }
    
    exp_dir = f"./inference/{args.version}/experiment_sweep_{args.mode}"
    os.makedirs(exp_dir, exist_ok=True)
    
    csv_path = os.path.join(exp_dir, "results_ssim.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'albedo_noise_step', 'num_inference_steps', 'ssim_score'])
    
    for i, idx in enumerate(indices):
        item = dataset[idx]
        data = _prepare_batch_data(ctx, item, args, idx)
        
        # Base Latents
        albedo_latents_base = ctx['vae'].encode(data['albedo'].to(dtype=torch.float32).cuda()).latent_dist.sample() * ctx['vae'].config.scaling_factor
        
        for a_step in EXPERIMENT_CONFIG["albedo_noise_step"]:
            for inf_step in EXPERIMENT_CONFIG["num_inference_steps"]:
                
                curr_latents = albedo_latents_base.clone()
                noise = torch.randn_like(curr_latents)
                t_tensor = torch.tensor([a_step], device=curr_latents.device).long()
                curr_latents_noisy = ctx['scheduler'].add_noise(curr_latents, noise, t_tensor)
                
                generator = _set_seed(args.seed)
                pipe_kwargs = {
                    "intensity": data['intensity'],
                    "color": data['color'],
                    "num_inference_steps": inf_step, 
                    "generator": generator,
                    "image": data['controlnet_cond'],
                    "albedo_latents": curr_latents_noisy.cuda(),
                    "prompt": item['prompt'] if args.version in ctx['COND_CONFIG']['prompt'] else None
                }
                if ctx['use_ambient_cond']:
                    pipe_kwargs["ambient"] = data['ambient']
                
                # Use Standard Pipeline for Sweep
                pipe_output, _, _ = ctx['pipe'](**pipe_kwargs)
                output_image = pipe_output.images[0]
                
                score = _calculate_ssim(output_image, data['ori_pil'])
                
                fname = f"{a_step}_{inf_step}_{idx}.png"
                output_image.save(os.path.join(exp_dir, fname))
                
                with open(csv_path, mode='a', newline='') as f:
                    csv.writer(f).writerow([idx, a_step, inf_step, f"{score:.4f}"])
                
                print(f"  [Sweep] {fname} | SSIM: {score:.4f}")

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

def _prepare_batch_data(ctx, item, args, idx):
    """
    處理單張資料的共通邏輯：計算光照、Albedo Estimator、Transforms。
    包含實驗視覺化圖片的儲存。
    """
    ori = item['image']
    normal = item['normal']
    depth = item['depth'].convert('L')
    
    if args.mode == 'lightlab' and 'mask_path' in item:
         mask = Image.open(item['mask_path']).convert('L')
    else:
         mask = item['mask'].convert('L')

    # Fixed lighting parameters
    intensity = 1.0
    color = torch.tensor([0, 1.0, 0], dtype=torch.float32) # Blue
    manual_ambient = 0.75

    # Compute Relighting
    cfg_cls = ctx['api'].PhysicalRelightingConfig
    compute_fn = ctx['api'].compute_relighting
    
    p_cfg = cfg_cls(ori, normal, depth)
    p_cfg.add_mask(mask, color, intensity)
    res = compute_fn(p_cfg, manual_ambient)
    
    # -----------------------------------------------------
    # [VISUALIZATION] 這裡補回你的視覺化與存檔邏輯
    # -----------------------------------------------------
    lightmap_rgb = None
    res_h = ctx['resolution']
    if ctx['returns_ambient']:
        ambient_val = res['ambient']
        # 存 illumination
        lightmap_rgb = res['lightmap_rgb']
        lightmap_rgb.save(f"./inference/{args.version}/debug_illumination_{idx}_{args.seed}.png")
    else:
        ambient_val = getattr(p_cfg, 'ambient', 0.75)
    
    lightmap = res['lightmap'].convert('RGB')
    # 存 gray lightmap
    lightmap.save(f"./inference/{args.version}/debug_lightmap_{idx}_{args.seed}.png")
    # -----------------------------------------------------

    target_pil = res['image'] 

    # Albedo Estimation
    if ctx['albedo_wrapper']:
        with torch.no_grad():
            ori_t = ctx['img_transform'](ori).unsqueeze(0).to('cuda')
            pred_alb = ctx['albedo_wrapper'](ori_t)
            
            # -----------------------------------------------------
            # [VISUALIZATION] 存 Albedo & Phys Recon
            # -----------------------------------------------------
            debug_alb_img = to_pil_image(pred_alb.squeeze(0).cpu().clamp(0, 1))
            debug_alb_img.save(f"./inference/{args.version}/debug_albedo_{idx}_{args.seed}.png")
            
            if lightmap_rgb is not None:
                l_rgb_resized = lightmap_rgb.resize((res_h, res_h), Image.BILINEAR)
                illum_tensor = transforms.ToTensor()(l_rgb_resized).to('cuda')
                phys_recon_tensor = pred_alb * illum_tensor
                phy_recon = to_pil_image(phys_recon_tensor.squeeze(0).clamp(0, 1).cpu())
                phy_recon.save(f"./inference/{args.version}/debug_phys_recon_{idx}_{args.seed}.png")
            # -----------------------------------------------------

            albedo_t = torch.clamp((pred_alb.cpu() * 2.0) - 1.0, -1.0, 1.0)
    else:
        albedo_t = ctx['img_transform'](item['albedo']).unsqueeze(0)

    # Condition Tensors
    norm_t = ctx['cond_transform'](normal)
    dep_t = ctx['cond_transform'](depth)
    mask_t = ctx['cond_transform'](mask)
    map_t = ctx['cond_transform'](lightmap)

    # ControlNet Condition
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
    parser.add_argument('-task', '--task', type=str, default='single', choices=['single', 'sweep'], help="Choose 'single' for 1 image, 'sweep' for experiments")
    parser.add_argument('-data', '--data', type=int, default=3, help="Number of data items to process")
    parser.add_argument('-seed', '--seed', type=int, default=6071)
    parser.add_argument('-pp', '--print_process', action='store_true')
    parser.add_argument('-gs', '--guidance_scale', type=float, default=0.0, help="SSIM Guidance Scale (e.g. 500.0). 0 to disable.")
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
            
    # Determine indices & Pre-handle Mask Paths (for LightLab)
    if args.mode == 'lightlab':
        ids = [181, 16, 75, 77] # 12/16 
        mask_paths = [
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/0_0.png',
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/1_0.png',
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/2_0.png',
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/3_0.png'
        ]
        indices = ids[:args.data+1]
        
        # 建立一個臨時的 mapping 讓 helper function 找得到 mask
        mapped_dataset = []
        for i, idx in enumerate(indices):
            item = ds[idx]
            item['mask_path'] = mask_paths[i]
            mapped_dataset.append(item)
        indices = range(len(mapped_dataset)) 
        dataset_to_use = mapped_dataset
    else:
        indices = range(args.data + 1)
        dataset_to_use = ds

    # 3. Dispatch Task
    if args.task == 'single':
        run_single_inference(ctx, args, dataset_to_use, indices)
    elif args.task == 'sweep':
        # 注意: 目前 Sweep 模式還沒整合 Guidance 參數，會用 default pipeline
        run_sweep_inference(ctx, args, dataset_to_use, indices)