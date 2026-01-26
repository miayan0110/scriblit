from diffusers.utils import load_image
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    DDIMScheduler
)
import os
import json
from network_controlnet import ControlNetModel
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from natsort import natsorted
from PIL import Image
import argparse
import numpy as np
import random
from omegaconf import OmegaConf

from datasets import load_from_disk, Image as HfImage, Dataset as DS, load_dataset
from huggingface_hub import snapshot_download
from physical_relighting_api_v2 import compute_relighting, PhysicalRelightingConfig


parser = argparse.ArgumentParser(description='Training Monitor Lighting on Various Dataset.')
parser.add_argument('-n', '--n', type=str, default='', help=("pretrained model path"))
parser.add_argument('-data', '--data', type=int, default=0, help=("inference data amount"))
parser.add_argument('-dp', '--path', type=str, default='lsun_train_new_phys', help=("inference data path"))
parser.add_argument('-ver', '--version', type=str, default='ex2_2', help=("experiment version"))
parser.add_argument('-pp', '--print_process', action="store_true", help=("print denoising process image"))
parser.add_argument('-seed', '--seed', type=int, default=6071)
parser.add_argument('-mode', '--mode', type=str, default='standard', choices=['standard', 'lightlab'], help="Switch between 'standard' test data and 'lightlab' out-of-domain data")
args = parser.parse_args()

if any(v in args.version for v in ['ex2_3', 'ex2_4', 'ex2_5']):
    print(f"Using Pipeline with PROMPT support for {args.version}")
    from pipeline_cn_ex2_3 import CustomControlNetPipeline
else:
    print(f"Using Standard Pipeline for {args.version}")
    from pipeline_cn import CustomControlNetPipeline

# CUDA_VISIBLE_DEVICES=1 python inference.py --version train_ex6 --data 3 --mode standard

def image_to_tensor(image):
    image = image / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
    return image.float()

def replace_unet_conv_in(unet):
    # replace the first layer to accept 8 in_channels
    _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()  # [320]
    _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)

    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(
        8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    
    # replace config
    unet.config["in_channels"] = 8
    return

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.manual_seed(seed)


## Inference Start
# === 1. Prepare Paths ===
local_dir = snapshot_download(
    repo_id="Miayan/project",
    repo_type="model",
    allow_patterns=[
        f"{args.version}/*",                 # 下載 args.version 底下全部
    ]
)
if os.path.isdir(f'{local_dir}/{args.version}'):
    args.n = f'{local_dir}/'

base_model_path = "stabilityai/stable-diffusion-2-1"
experiment = f'{args.n}{args.version}'

# Find best checkpoint
best_weight = os.listdir(experiment)
best_weight = [f for f in best_weight if f.startswith('checkpoint')]
best_weight = natsorted(best_weight)[-1]	# get the last checkpoint
print(f"Using checkpoint: {best_weight}")

# Paths
controlnet_path = "%s/%s/controlnet"%(experiment, best_weight)
unet_path = "%s/%s/custom_unet.pth"%(experiment, best_weight)
custom_encoder_path = "%s/%s/custom_encoder"%(experiment, best_weight)
albedo_estimator_path = "%s/%s/albedo_estimator"%(experiment, best_weight)


# === 2. Load Config & Determine Settings ===
config_path = os.path.join(experiment, "config.yaml")
train_cfg = None

# Default settings (Fallback for legacy experiments without config)
use_ambient_cond = False
relighting_impl = 'v2'
use_pred_albedo = False

if os.path.exists(config_path):
    print(f"[Info] Found config.yaml in {config_path}, loading settings...")
    train_cfg = OmegaConf.load(config_path)
    
    # 2.1 Determine Ambient Condition
    if train_cfg.get('model', {}).get('enable_ambient_cond', False):
        use_ambient_cond = True
    
    # 2.2 Determine Relighting API
    if train_cfg.get('data', {}).get('relighting_impl', None):
        relighting_impl = train_cfg.data.relighting_impl
        
    # 2.3 Determine Albedo Estimator
    if train_cfg.get('albedo_estimator', {}).get('enabled', False):
        use_pred_albedo = True
else:
    print(f"[Warning] Config.yaml not found. Using legacy defaults (No ambient, v2 API).")
    
print(f"--- Inference Settings ---")
print(f"  > Ambient Condition: {use_ambient_cond}")
print(f"  > Relighting API:    {relighting_impl}")
print(f"  > Use Pred Albedo:   {use_pred_albedo}")
print(f"--------------------------")


# === 3. Dynamic Imports ===

# 3.1 Import Encoder (根據 Config 決定)
if use_ambient_cond:
    from light_cond_encoder_amb import CustomEncoder
else:
    from light_cond_encoder import CustomEncoder

# 3.2 Import Relighting API (根據 Config 決定)
if relighting_impl == 'gemini_amb':
    import physical_relighting_api_v2_gemini_amb as relighting_api
    returns_ambient = True
elif relighting_impl == 'gemini':
    import physical_relighting_api_v2_gemini as relighting_api
    returns_ambient = False
else:
    import physical_relighting_api_v2 as relighting_api
    returns_ambient = False
    
PhysicalRelightingConfig = relighting_api.PhysicalRelightingConfig
compute_relighting = relighting_api.compute_relighting


# === 4. Load Models ===
unet = UNet2DConditionModel.from_pretrained(
    base_model_path, subfolder="unet", revision=None, variant=None
)
replace_unet_conv_in(unet)
unet.load_state_dict(torch.load(unet_path), strict=False)
unet.cuda().eval()

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32)
cond_encoder = CustomEncoder.from_pretrained(custom_encoder_path, torch_dtype=torch.float32)

albedo_wrapper = None
if use_pred_albedo and os.path.exists(albedo_estimator_path):
    from albedo_estimator import AlbedoWrapper
    print("Loading Albedo Estimator...")
    albedo_wrapper = AlbedoWrapper.from_pretrained(albedo_estimator_path, low_cpu_mem_usage=False)
    albedo_wrapper.to('cuda').eval()

pipe = CustomControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, cond_encoder=cond_encoder, torch_dtype=torch.float32, unet = unet
)

# # speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# # remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()


# === 5. Transforms & Preps ===
image_transforms = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

conditioning_image_transforms = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

vae = AutoencoderKL.from_pretrained(
    base_model_path, subfolder="vae", revision=None, variant=None
)
noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")


os.makedirs('./inference/%s'%args.version, exist_ok=True)
if args.print_process:
    os.makedirs('./inference/%s/denoise'%args.version, exist_ok=True)
    os.makedirs('./inference/%s/cal'%args.version, exist_ok=True)

num = 0
LIGHTLAB_CONFIG = {
    'ids': [181, 12, 75, 77],   # 12/16, 45, 53
    'masks': [
             '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/0_0.png',
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/1_0.png',
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/2_0.png',
            '/mnt/HDD3/miayan/paper/scriblit/eval_data/images_flatten/mask/3_0.png']
}

COND_CONFIG = {
    "lightmap": ['train_ex2_2', 'train_ex2_3', 'train_ex2_4', 
                'train_ex4', 'train_ex4_1', 'train_ex5', 
                'train_ex5_1', 'train_ex6', 'train_ex6_1', 
                'train_ex6_2', 'train_ex6_3', 'train_ex6_4',
                'train_ex6_5', 'train_ex6_6', 'train_ex6_7',
                'train_ex6_8', 'train_ex6_9', 'train_ex6_10',
                'train_ex6_11', 'train_ex6_12', 'train_ex6_13',
                'train_ex7', 'train_ex8', 'train_ex8_1', 
                'train_ex8_2', 'train_ex8_3', 'train_ex8_4',
                'train_ex8_5', 'train_ex8_6', 'train_ex8_7',
                'train_ex8_8', 'train_ex8_9', 'train_ex8_10'],
    "prompt": ['train_ex2_3', 'train_ex2_4', 'train_ex2_5']
}

# experiment setting
weight_dtype = torch.float32
ds_hf_repo = "Miayan/physical-relighting-dataset"
ds_split = "train"
indices_to_run = range(args.data + 1)
if args.mode == 'lightlab':
    print("Mode: LightLab Evaluation")
    ds_hf_repo = "Miayan/physical-relighting-eval-dataset"
    indices_to_run = LIGHTLAB_CONFIG['ids'][:args.data + 1]
    mask_paths = LIGHTLAB_CONFIG['masks'][:args.data + 1]

ds = load_dataset(ds_hf_repo, split=ds_split, cache_dir=f"/mnt/HDD3/miayan/paper/relighting_datasets/")
for col in ds.column_names:
    if col not in ('color', 'intensity', 'prompt'):
        ds = ds.cast_column(col, HfImage(decode=True))

for i, idx in enumerate(indices_to_run):
    item = ds[idx]
 
    albedo = item['albedo']
    normal = item['normal']
    depth = item['depth'].convert('L')
    ori = item['image']
     
    if args.mode == 'lightlab':
        mask_path = mask_paths[i]
        mask = Image.open(mask_path).convert('L')
    else:
        mask = item['mask'].convert('L')
 
    # # use color and intensity from dataset
    # intensity = item['intensity'][-1]
    # color = item['color']

    # custom light color and intensity
    intensity = 1.0
    color = torch.tensor([0/255.0, 0/255.0, 255/255.0], dtype=torch.float32)
    manual_ambient=0.75
 
    config = PhysicalRelightingConfig(ori, normal, depth)
    config.add_mask(mask, color, intensity)
    result = compute_relighting(config, manual_ambient)
    
    ambient_val = manual_ambient
    if returns_ambient:
        ambient_val = result['ambient']
        # for experiment visualization
        lightmap_rgb = result['lightmap_rgb']
        lightmap_rgb.save(f"./inference/{args.version}/debug_illumination_{idx}_{args.seed}.png")
    else:
        lightmap_rgb = None
        ambient_val = getattr(config, 'ambient', 0.75)
    
    lightmap = result['lightmap'].convert('RGB')
    lightmap.save(f"./inference/{args.version}/debug_lightmap_{idx}_{args.seed}.png")   # for experiment visualization
    
    
    if albedo_wrapper is not None:
        ori_tensor = image_transforms(ori).unsqueeze(0).to('cuda')
        with torch.no_grad():
            pred_albedo = albedo_wrapper(ori_tensor)
            
        # for experiment visualization
        debug_alb_img = to_pil_image(pred_albedo.squeeze(0).cpu().clamp(0, 1))
        debug_alb_img.save(f"./inference/{args.version}/debug_albedo_{idx}_{args.seed}.png")
        if lightmap_rgb is not None:
            lightmap_rgb = lightmap_rgb.resize((512, 512), Image.BILINEAR)
            illum_tensor = transforms.ToTensor()(lightmap_rgb).to('cuda')
            phys_recon_tensor = pred_albedo * illum_tensor
            phy_recon = to_pil_image(phys_recon_tensor.squeeze(0).clamp(0, 1).cpu())
            phy_recon.save(f"./inference/{args.version}/debug_phys_recon_{idx}_{args.seed}.png")
        
        albedo = (pred_albedo.cpu() * 2.0) - 1.0
        albedo = torch.clamp(albedo, -1.0, 1.0)
    else:
        albedo = image_transforms(albedo).unsqueeze(0)

        
    normal = conditioning_image_transforms(normal)
    depth = conditioning_image_transforms(depth)
    mask = conditioning_image_transforms(mask)
    lightmap = conditioning_image_transforms(lightmap)
    color = torch.tensor(color, dtype=torch.float32).unsqueeze(0)
    intensity = torch.tensor([intensity], dtype=torch.float32).unsqueeze(0)
    ambient = torch.tensor([ambient_val], dtype=torch.float32).unsqueeze(0)
 
    albedo_latents = vae.encode(albedo.to(dtype=weight_dtype)).latent_dist.sample()
    albedo_latents = albedo_latents * vae.config.scaling_factor
    albedo_noise = torch.randn_like(albedo_latents)#pls
    timesteps_albedo = torch.randint(200, 201, (1,))
    albedo_latents = noise_scheduler.add_noise(albedo_latents, albedo_noise, timesteps_albedo)
 
    # generate image
    generator = set_seed(args.seed)
    if args.version in COND_CONFIG['lightmap']:
        controlnet_cond = (normal, lightmap)
    else:
        DM = depth * mask
        controlnet_cond = (normal, depth, mask, DM)

    pipe_kwargs = {
        "intensity": intensity,
        "color": color,
        "num_inference_steps": 20,
        "generator": generator,
        "image": controlnet_cond,
        "albedo_latents": albedo_latents.cuda()
    }
    pipe_kwargs["prompt"] = item['prompt'] if args.version in COND_CONFIG['prompt'] else None
    
    if use_ambient_cond:
        pipe_kwargs["ambient"] = ambient

    pipe_output, denoise_lst, cal_lst = pipe(**pipe_kwargs)
    image = pipe_output.images[0]
    
    # with torch.no_grad():
    #     phys_albedo = albedo.detach().clone() * 0.5 + 0.5
    #     phys_lightmap = lightmap.unsqueeze(0).to(phys_albedo.device)
    #     phys_recon_tensor = phys_albedo * phys_lightmap
    #     phys_recon_img = to_pil_image(phys_recon_tensor.squeeze(0).clamp(0, 1).cpu())
    #     phys_recon_img.save(f"./inference/{args.version}/phys_check_{idx}_{args.seed}.png")

    if args.print_process:
        for i, (denoise_img, cal_img) in enumerate(zip(denoise_lst, cal_lst)):
            denoise_img = denoise_img[0].resize((256,256))
            cal_img = cal_img[0].resize((256,256))
            denoise_img.save("./inference/%s/denoise/%d_%d.png"%(args.version, i, args.seed))
            cal_img.save("./inference/%s/cal/%d_%d.png"%(args.version, i, args.seed))
    
    image.save("./inference/%s/output_%d_%d.png"%(args.version, idx, args.seed))
    ori.save("./inference/%s/ori_%d_%d.png"%(args.version, idx, args.seed))
    result['image'].save("./inference/%s/target_%d_%d.png"%(args.version, idx, args.seed))
    # mask.save("./inference/%s/mask_%d_%d.png"%(args.version, idx, args.seed))
print(f'Inference version: {args.version}. Done.')
 
