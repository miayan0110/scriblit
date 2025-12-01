from diffusers.utils import load_image
import torch
from PIL import Image
import torchvision.transforms as transforms
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

from datasets import load_from_disk, Image as HfImage, Dataset as DS
from light_cond_encoder import CustomEncoder
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description='Training Monitor Lighting on Various Dataset.')
parser.add_argument('-n', '--n', type=str, default='', help=("pretrained model path"))
parser.add_argument('-data', '--data', type=int, default=0, help=("inference data amount"))
parser.add_argument('-dp', '--path', type=str, default='lsun_train_new_phys', help=("inference data path"))
parser.add_argument('-ver', '--version', type=str, default='ex2_2', help=("experiment version"))
parser.add_argument('-pp', '--print_process', action="store_true", help=("print denoising process image"))
parser.add_argument('-seed', '--seed', type=int, default=6071)
args = parser.parse_args()

if any(v in args.version for v in ['ex2_2', 'ex3', 'ex4', 'ex5', 'ex6']):
    from pipeline_cn import CustomControlNetPipeline
else:
	from pipeline_cn_ex2_3 import CustomControlNetPipeline

# CUDA_VISIBLE_DEVICES=7 python inference.py -n scribblelight_controlnet -data data

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


## Inference
local_dir = snapshot_download(
    repo_id="Miayan/project",
    repo_type="model",
    allow_patterns=[
        f"{args.version}/*",                 # 下載 args.version 底下全部
    ]
)
args.n = local_dir

base_model_path = "stabilityai/stable-diffusion-2-1"
experiment = f'{args.n}/{args.version}'
best_weight = os.listdir(experiment)
best_weight = [f for f in best_weight if f.startswith('checkpoint')]
best_weight = natsorted(best_weight)[-1]
print(best_weight)
controlnet_path = "%s/%s/controlnet"%(experiment, best_weight)
unet_path = "%s/%s/custom_unet.pth"%(experiment, best_weight)
custom_encoder_path = "%s/%s/custom_encoder"%(experiment, best_weight)

unet = UNet2DConditionModel.from_pretrained(
	base_model_path, subfolder="unet", revision=None, variant=None
)

replace_unet_conv_in(unet)
unet.load_state_dict(torch.load(unet_path), strict=False)
unet.cuda().eval()

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32)
cond_encoder = CustomEncoder.from_pretrained(custom_encoder_path, torch_dtype=torch.float32)

pipe = CustomControlNetPipeline.from_pretrained(
	base_model_path, controlnet=controlnet, cond_encoder=cond_encoder, torch_dtype=torch.float32, unet = unet
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

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

file_path = 'dataset/%s/prompt.json'%args.version


num = 0

os.makedirs('./inference/%s'%args.version, exist_ok=True)
if args.print_process:
	os.makedirs('./inference/%s/denoise'%args.version, exist_ok=True)
	os.makedirs('./inference/%s/cal'%args.version, exist_ok=True)

weight_dtype = torch.float32
ds = load_from_disk(f'/mnt/HDD7/miayan/paper/relighting_datasets/{args.path}')
for col in ds.column_names:
	if col not in ('color', 'intensity', 'prompt'):
		ds = ds.cast_column(col, HfImage(decode=True))

for idx, item in enumerate(ds):
	if idx > args.data:
		break
    
	albedo = item['albedo']
	normal = item['normal']
	prompt = item['prompt']
	depth = item['depth'].convert('L')
	mask = item['mask'].convert('L')
	# intensity = item['intensity'][-1]
	color = item['color']
	intensity = 1.0
	# color = torch.tensor([255/255.0, 0/255.0, 0/255.0], dtype=torch.float32)
	lightmap = item['lightmap'].convert('RGB')
 
	ori = item['image']
	target = item['pseudo_gt']
 
	albedo = image_transforms(albedo).unsqueeze(0)
	normal = conditioning_image_transforms(normal)
	depth = conditioning_image_transforms(depth)
	mask = conditioning_image_transforms(mask)
	lightmap = conditioning_image_transforms(lightmap)
	color = torch.tensor(color, dtype=torch.float32).unsqueeze(0)
	intensity = torch.tensor([intensity], dtype=torch.float32).unsqueeze(0)
 
	albedo_latents = vae.encode(albedo.to(dtype=weight_dtype)).latent_dist.sample()
	albedo_latents = albedo_latents * vae.config.scaling_factor
	albedo_noise = torch.randn_like(albedo_latents)#pls
	timesteps_albedo = torch.randint(200, 201, (1,))
	albedo_latents = noise_scheduler.add_noise(albedo_latents, albedo_noise, timesteps_albedo)
 
	# generate image
	generator = set_seed(args.seed)
	if args.version in ['train_ex2_2', 'train_ex2_3', 'train_ex2_4', 'train_ex4', 'train_ex4_1', 'train_ex5', 'train_ex5_1', 'train_ex6']:
		controlnet_cond = (normal, lightmap)
	else:
		DM = depth * mask
		controlnet_cond = (normal, depth, mask, DM)
  
	cond_prompt = prompt if args.version in ['train_ex2_3', 'train_ex2_4', 'train_ex2_5'] else None
 
	pipe_output, denoise_lst, cal_lst = pipe(
			intensity=intensity, color=color, prompt=cond_prompt, num_inference_steps=20, generator=generator, image=controlnet_cond, albedo_latents=albedo_latents.cuda()
		)
	image = pipe_output.images[0]

	if args.print_process:
		for i, (denoise_img, cal_img) in enumerate(zip(denoise_lst, cal_lst)):
			denoise_img = denoise_img[0].resize((256,256))
			cal_img = cal_img[0].resize((256,256))
			denoise_img.save("./inference/%s/denoise/%d_%d.png"%(args.version, i, args.seed))
			cal_img.save("./inference/%s/cal/%d_%d.png"%(args.version, i, args.seed))
	# image = image.resize((256,256))
	image.save("./inference/%s/output_%d_%d.png"%(args.version, idx, args.seed))
	# ori.save("./inference/%s/ori_%d_%d.png"%(args.version, idx, args.seed))
	# target.save("./inference/%s/target_%d_%d.png"%(args.version, idx, args.seed))
	# item['mask'].save("./inference/%s/mask_%d_%d.png"%(args.version, idx, args.seed))
print(f'Inference version: {args.version}. Done.')
 


# with open(file_path, 'r') as file:
# 	for number, line in enumerate(file):
# 		data = json.loads(line)
		
# 		normal = data.get('normal')
# 		shading = data.get('shading')
# 		albedo = data.get('albedo')
# 		target = data.get('target')
# 		prompt = data.get('prompt')
# 		control_image_normal = load_image(normal)
# 		control_image_shading = load_image(shading)
# 		control_image_albedo = load_image(albedo)

# 		control_image_normal = control_image_normal.resize((512,512))
# 		control_image_shading = control_image_shading.resize((512,512))

# 		image = image_transforms(control_image_albedo)
# 		image = image.unsqueeze(0)
# 		albedo_latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample()
# 		albedo_latents = albedo_latents * vae.config.scaling_factor
# 		albedo_noise = torch.randn_like(albedo_latents)#pls
# 		timesteps_albedo = torch.randint(200, 201, (1,))
# 		albedo_latents = noise_scheduler.add_noise(albedo_latents, albedo_noise, timesteps_albedo)

# 		# generate image
# 		generator = set_seed(args.seed)
# 		image = pipe(
# 			prompt, num_inference_steps=20, generator=generator, image=(control_image_normal, control_image_shading), albedo_latents= albedo_latents.cuda()
# 		).images[0]

# 		image.save("./inference/%s/output_%d_%d.png"%(args.version,number,args.seed))

# 		normal = Image.open(normal)
# 		normal.save("./inference/%s/normal_%d.png"%(args.version,number))
# 		shading = Image.open(shading).convert('RGB')
# 		shading.save("./inference/%s/shading_%d.png"%(args.version,number))
# 		albedo = Image.open(albedo)
# 		albedo.save("./inference/%s/albedo_%d.png"%(args.version,number))
# 		target = Image.open(target)
# 		target.save("./inference/%s/target_%d.png"%(args.version,number))