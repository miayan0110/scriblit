import os
import argparse
import torch
import numpy as np
import random
import csv
from PIL import Image
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
# 1. 通用設定與模型載入 (Common Setup)
# ==========================================
def setup_pipeline(args):
    """
    負責所有模型載入、Config 讀取、Pipeline 組裝。
    回傳一個包含所有必要物件的字典 context。
    """
    print(f"--- [Setup] Initializing Pipeline for Version: {args.version} ---")
    
    # 1.1 Download / Locate Model
    local_dir = snapshot_download(repo_id="Miayan/project", repo_type="model", allow_patterns=[f"{args.version}/*"])
    experiment_dir = os.path.join(local_dir, args.version)
    
    # Find best checkpoint
    checkpoints = [f for f in os.listdir(experiment_dir) if f.startswith('checkpoint')]
    best_ckpt = natsorted(checkpoints)[-1]
    ckpt_path = os.path.join(experiment_dir, best_ckpt)
    print(f"  > Using checkpoint: {best_ckpt}")

    # 1.2 Load Config
    config_path = os.path.join(experiment_dir, "config.yaml")
    train_cfg = OmegaConf.load(config_path) if os.path.exists(config_path) else None
    
    # Determine Settings
    use_ambient_cond = train_cfg.get('model', {}).get('enable_ambient_cond', False) if train_cfg else False
    relighting_impl = train_cfg.get('data', {}).get('relighting_impl', 'v2') if train_cfg else 'v2'
    use_pred_albedo = train_cfg.get('albedo_estimator', {}).get('enabled', False) if train_cfg else False

    # 1.3 Import Relighting API & Encoder
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

    # =========================================================
    # 1.4 動態 Import Pipeline (保留你的特殊邏輯)
    # =========================================================
    if any(v in args.version for v in ['ex2_3', 'ex2_4', 'ex2_5']):
        print(f"  > Using Pipeline with PROMPT support for {args.version}")
        from pipeline_cn_ex2_3 import CustomControlNetPipeline
    else:
        print(f"  > Using Standard Pipeline for {args.version}")
        from pipeline_cn import CustomControlNetPipeline

    # 1.5 Load Models
    base_model_path = "stabilityai/stable-diffusion-2-1"
    
    # UNet
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    _replace_unet_conv_in(unet) 
    unet.load_state_dict(torch.load(os.path.join(ckpt_path, "custom_unet.pth")), strict=False)
    unet.cuda().eval()
    
    # ControlNet & Encoder
    controlnet = ControlNetModel.from_pretrained(os.path.join(ckpt_path, "controlnet"), torch_dtype=torch.float32)
    cond_encoder = EncoderClass.from_pretrained(os.path.join(ckpt_path, "custom_encoder"), torch_dtype=torch.float32)
    
    # Albedo Estimator (Optional)
    albedo_wrapper = None
    if use_pred_albedo and os.path.exists(os.path.join(ckpt_path, "albedo_estimator")):
        albedo_wrapper = AlbedoWrapper.from_pretrained(os.path.join(ckpt_path, "albedo_estimator"), low_cpu_mem_usage=False)
        albedo_wrapper.to('cuda').eval()

    # Pipeline Instantiation
    pipe = CustomControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, cond_encoder=cond_encoder, unet=unet, torch_dtype=torch.float32
    )
    pipe.enable_model_cpu_offload()

    # VAE & Scheduler
    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    # Transforms
    img_transform = transforms.Compose([
        transforms.Resize((512, 512), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    cond_transform = transforms.Compose([
        transforms.Resize((512, 512), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    return {
        "pipe": pipe,
        "vae": vae,
        "scheduler": noise_scheduler,
        "albedo_wrapper": albedo_wrapper,
        "api": api,
        "returns_ambient": returns_ambient,
        "use_ambient_cond": use_ambient_cond,
        "img_transform": img_transform,
        "cond_transform": cond_transform,
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
        
        # 1. Prepare Data
        # [修改] 傳入 idx 以便 _prepare_batch_data 做存檔命名
        data = _prepare_batch_data(ctx, item, args, idx)
        
        # 2. Encode Albedo Latents
        albedo_latents = ctx['vae'].encode(data['albedo'].to(dtype=torch.float32)).latent_dist.sample() * ctx['vae'].config.scaling_factor
        albedo_noise = torch.randn_like(albedo_latents)
        timesteps_albedo = torch.randint(200, 201, (1,))
        albedo_latents_noisy = ctx['scheduler'].add_noise(albedo_latents, albedo_noise, timesteps_albedo)

        # 3. Pipeline Args
        generator = _set_seed(args.seed)
        pipe_kwargs = {
            "intensity": data['intensity'],
            "color": data['color'],
            "num_inference_steps": 20, 
            "generator": generator,
            "image": data['controlnet_cond'],
            "albedo_latents": albedo_latents_noisy.cuda(),
            "prompt": item['prompt'] if args.version in ctx['COND_CONFIG']['prompt'] else None
        }
        if ctx['use_ambient_cond']:
            pipe_kwargs["ambient"] = data['ambient']

        # 4. Run
        pipe_output, denoise_lst, cal_lst = ctx['pipe'](**pipe_kwargs)
        image = pipe_output.images[0]

        # 5. Save Results
        suffix = f"{idx}_{args.seed}"
        image.save(f"{out_dir}/output_{suffix}.png")
        data['ori_pil'].save(f"{out_dir}/ori_{suffix}.png")
        data['target_pil'].save(f"{out_dir}/target_{suffix}.png")

        if args.print_process:
            for j, (den, cal) in enumerate(zip(denoise_lst, cal_lst)):
                den[0].resize((256,256)).save(f"{out_dir}/denoise/{j}_{args.seed}.png")
                cal[0].resize((256,256)).save(f"{out_dir}/cal/{j}_{args.seed}.png")
        
        print(f"  [Single] Processed {idx}, saved to {out_dir}")


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
    
    print(f"  Configs: {EXPERIMENT_CONFIG}")
    print(f"  Saving to: {exp_dir}")

    for i, idx in enumerate(indices):
        item = dataset[idx]
        
        # [修改] 傳入 idx 以便 _prepare_batch_data 做存檔命名
        data = _prepare_batch_data(ctx, item, args, idx)
        
        # Base Latents
        albedo_latents_base = ctx['vae'].encode(data['albedo'].to(dtype=torch.float32)).latent_dist.sample() * ctx['vae'].config.scaling_factor
        
        for a_step in EXPERIMENT_CONFIG["albedo_noise_step"]:
            for inf_step in EXPERIMENT_CONFIG["num_inference_steps"]:
                
                # 1. Noise Injection
                curr_latents = albedo_latents_base.clone()
                noise = torch.randn_like(curr_latents)
                t_tensor = torch.tensor([a_step], device=curr_latents.device).long()
                curr_latents_noisy = ctx['scheduler'].add_noise(curr_latents, noise, t_tensor)
                
                # 2. Pipeline Args
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
                
                # 3. Run
                pipe_output, _, _ = ctx['pipe'](**pipe_kwargs)
                output_image = pipe_output.images[0]
                
                # 4. Metric & Save
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
    color = torch.tensor([0, 0, 1.0], dtype=torch.float32) # Blue
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
                l_rgb_resized = lightmap_rgb.resize((512, 512), Image.BILINEAR)
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
# CUDA_VISIBLE_DEVICES=1 python inference.py --version train_ex8 --data 3 --task single --mode standard
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ver', '--version', type=str, required=True, help="Experiment version (e.g., train_ex8)")
    parser.add_argument('-mode', '--mode', type=str, default='standard', choices=['standard', 'lightlab'])
    parser.add_argument('-task', '--task', type=str, default='single', choices=['single', 'sweep'], help="Choose 'single' for 1 image, 'sweep' for experiments")
    parser.add_argument('-data', '--data', type=int, default=3, help="Number of data items to process")
    parser.add_argument('-seed', '--seed', type=int, default=6071)
    parser.add_argument('-pp', '--print_process', action='store_true')
    args = parser.parse_args()

    # 1. Setup
    ctx = setup_pipeline(args)
    
    # 2. Load Dataset
    repo = "Miayan/physical-relighting-dataset"
    if args.mode == 'lightlab': repo = "Miayan/physical-relighting-eval-dataset"
    
    print(f"Loading Dataset: {repo}")
    # cache_dir 請依你的環境修改
    ds = load_dataset(repo, split="train", cache_dir="/mnt/HDD3/miayan/paper/relighting_datasets/")
    
    for col in ds.column_names:
        if col not in ('color', 'intensity', 'prompt'):
            ds = ds.cast_column(col, HfImage(decode=True))
            
    # Determine indices & Pre-handle Mask Paths (for LightLab)
    if args.mode == 'lightlab':
        ids = [181, 12, 75, 77] 
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
        run_sweep_inference(ctx, args, dataset_to_use, indices)