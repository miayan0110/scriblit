import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
	AutoencoderKL,
	DDPMScheduler,
	StableDiffusionControlNetPipeline,
	UNet2DConditionModel,
	UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from dataloader_new import Indoor_dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision

from light_cond_encoder import CustomEncoder as CustomEncoderV1
from light_cond_encoder_amb import CustomEncoder as CustomEncoderAmb

from dataset.intrinsic.pipeline import load_models, run_pipeline
from my_utils import alb_to_pil, gen_albedo, depth_estimation, normal_estimation_sn, img_transforms, _move_to_device, calculate_pred_lightmap
from transformers import pipeline as depth_pipeline
from dep_nor.models.dsine.v02 import DSINE_v02 as DSINE
import dep_nor.utils.utils as dep_nor_utils

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from network_controlnet import ControlNetModel
from pipeline_cn import CustomControlNetPipeline
from albedo_estimator import AlbedoWrapper


logger = get_logger(__name__)

## Tensorboard logging
def write_tb_log(image, tag, n_img, log_writer, i):

	output_to_show = image.cpu().data[0: n_img, ...]
	output_to_show = (output_to_show)
	output_to_show[output_to_show > 1] = 1
	output_to_show[output_to_show < 0] = 0
	grid = torchvision.utils.make_grid(output_to_show, nrow=n_img)

	log_writer.add_image(tag, grid, i + 1)
 
def write_tb_log_source(image, tag, n_img, log_writer, i):

    output_to_show = image.cpu().data[0: n_img, ...]
    output_to_show = (output_to_show + 1) / 2
    output_to_show[output_to_show > 1] = 1
    output_to_show[output_to_show < 0] = 0
    grid = torchvision.utils.make_grid(output_to_show, nrow=n_img)

    log_writer.add_image(tag, grid, i + 1)
 
def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
	img_str = ""
	if image_logs is not None:
		img_str = "You can find some example images below.\n\n"
		for i, log in enumerate(image_logs):
			images = log["images"]
			validation_prompt = log["validation_prompt"]
			validation_image = log["validation_image"]
			validation_image.save(os.path.join(repo_folder, "image_control.png"))
			img_str += f"prompt: {validation_prompt}\n"
			images = [validation_image] + images
			image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
			img_str += f"![images_{i})](./images_{i}.png)\n"

	model_description = f"""
        # controlnet-{repo_id}

        These are controlnet weights trained on {base_model} with new type of conditioning.
        {img_str}
        """
	model_card = load_or_create_model_card(
		repo_id_or_path=repo_id,
		from_training=True,
		license="creativeml-openrail-m",
		base_model=base_model,
		model_description=model_description,
		inference=True,
	)

	tags = [
		"stable-diffusion",
		"stable-diffusion-diffusers",
		"text-to-image",
		"diffusers",
		"controlnet",
		"diffusers-training",
	]
	model_card = populate_model_card(model_card, tags=tags)

	model_card.save(os.path.join(repo_folder, "README.md"))
 

## Model import function
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
	text_encoder_config = PretrainedConfig.from_pretrained(
		pretrained_model_name_or_path,
		subfolder="text_encoder",
		revision=revision,
	)
	model_class = text_encoder_config.architectures[0]

	if model_class == "CLIPTextModel":
		from transformers import CLIPTextModel

		return CLIPTextModel
	elif model_class == "RobertaSeriesModelWithTransformation":
		from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

		return RobertaSeriesModelWithTransformation
	else:
		raise ValueError(f"{model_class} is not supported.")

def replace_unet_conv_in(unet):
	# replace the first layer to accept 8 in_channels
	_weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
	_bias = unet.conv_in.bias.clone()  # [320]
	_weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
	_weight[:, 4:8, :, :] = 0

	# new conv_in channel
	_n_convin_out_channel = unet.conv_in.out_channels
	_new_conv_in = Conv2d(
		8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
	)
	_new_conv_in.weight = Parameter(_weight)
	_new_conv_in.bias = Parameter(_bias)
	unet.conv_in = _new_conv_in
	logging.info("Unet conv_in layer is replaced")

	# replace config
	unet.config["in_channels"] = 8
	logging.info("Unet config is updated")

	unet.conv_in.requires_grad_(True)

	return

def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    from diffusers.utils.torch_utils import is_compiled_module
    model = model._orig_mod if is_compiled_module(model) else model
    return model


## Predict x0 from model output
def pred_x0_latents_batchsize_1(noise_scheduler, model_pred, x_t, t):
    """ v2
    model_pred: UNet 對應 prediction_type 的輸出 (ε or v)，shape = (B,4,H,W)
    x_t:       噪聲後的 latents (B,4,H,W) —— 注意是「影像 latents」，不是 8 通道 cat
    t:         (B,) timesteps
    回傳：z0_hat (B,4,H,W)
    """
    step_out = noise_scheduler.step(model_pred, t, x_t, return_dict=True)
    return step_out.pred_original_sample  # 就是 \hat{x}_0

def pred_x0_latents(noise_scheduler, model_pred, x_t, t):
    """
    model_pred: (B, 4, H, W)  来自 UNet
    x_t:       (B, 4, H, W)  noisy latents
    t:         (B,) or scalar timestep(s)
    回傳:      (B, 4, H, W) 对应每个 sample 的 \hat{x}_0
    """
    # t 是 tensor 且有 batch 維度 → 每張圖用自己的 t
    if isinstance(t, torch.Tensor) and t.ndim > 0:
        B = t.shape[0]
        preds = []
        for i in range(B):
            # 每一次只取一張圖 + 一個 scalar t
            ti = int(t[i].item())
            step_out = noise_scheduler.step(
                model_pred[i:i+1],  # (1,4,H,W)
                ti,                 # scalar
                x_t[i:i+1],         # (1,4,H,W)
                return_dict=True,
            )
            preds.append(step_out.pred_original_sample)  # (1,4,H,W)

        return torch.cat(preds, dim=0)  # (B,4,H,W)

    # 否則 t 本來就是 scalar 的情況
    t_scalar = int(t) if not isinstance(t, torch.Tensor) else int(t.item())
    step_out = noise_scheduler.step(model_pred, t_scalar, x_t, return_dict=True)
    return step_out.pred_original_sample


## Utils for training
# Transform RGB image to gray-scale luminance
def get_luminance(img):
    return (img[:, 0:1] * 0.299 + img[:, 1:2] * 0.587 + img[:, 2:3] * 0.114)

def compute_self_recon_loss(cfg, batch, pred_ori_alb, gt_ori_alb, global_step, num_update_steps_per_epoch, weight_dtype):
    """
    計算物理重建 Loss，包含以下功能：
    1. 支援多種 Target 模式: 'original', 'teacher', 'staged'
    2. 支援 Affine Alignment (Bias & Scale) - 使用手算 Normal Equation 省顯存
    3. 支援 Epsilon-Insensitive Loss
    """
    loss_cfg = cfg.losses.get('self_recon_loss', {'enabled': False})
    
    # 基本檢查
    if not loss_cfg.get('enabled', False):
        return torch.tensor(0.0, device=pred_ori_alb.device, dtype=weight_dtype), 0.0
    if "lightmap_rgb" not in batch or batch["lightmap_rgb"] is None:
        return torch.tensor(0.0, device=pred_ori_alb.device, dtype=weight_dtype), 0.0

    gt_illumination = batch["lightmap_rgb"].to(dtype=weight_dtype)
    mode = loss_cfg.get('mode', 'teacher')
    
    # --- 1. 準備 Target Image ---
    target_recon_img = None
    current_weight = loss_cfg.get('weight', 1.0)

    if mode == "original":
        # Target = Original Image
        target_recon_img = (batch["pixel_values"].to(dtype=weight_dtype) + 1.0) / 2.0
    
    elif mode == "teacher":
        # Target = Teacher Albedo * Lightmap
        target_recon_img = gt_ori_alb * gt_illumination
        
    elif mode == "staged":
        # 分階段：先排毒(灰階) -> 後上色(彩色)
        current_epoch = global_step // num_update_steps_per_epoch
        switch_epoch = loss_cfg.get('staging', {}).get('switch_epoch', 5)
        
        if current_epoch < switch_epoch:
            # 階段一：排毒
            teacher_luma = get_luminance(gt_ori_alb)
            target_albedo = teacher_luma.repeat(1, 3, 1, 1)
            current_weight = loss_cfg.get('staging', {}).get('stage1_weight', 2.0)
        else:
            # 階段二：上色
            target_albedo = gt_ori_alb
            current_weight = loss_cfg.get('staging', {}).get('stage2_weight', 0.5)
        
        target_recon_img = target_albedo * gt_illumination

    if target_recon_img is None:
        return torch.tensor(0.0, device=pred_ori_alb.device, dtype=weight_dtype), 0.0

    # --- 2. 計算 Prediction (加入 Bias & Scale) ---
    use_affine = loss_cfg.get('affine_alignment', False)
    
    if use_affine:
        # 使用 Normal Equation 手動解線性回歸: y = s*x1 + b*x2
        # Term 1: Albedo * Light (x1)
        term1 = pred_ori_alb * gt_illumination
        # Term 2: Albedo (x2) - 模擬環境光
        term2 = pred_ori_alb
        
        # Flatten: (B, N)
        # 把 RGB 通道也展平，當作一樣的 s, b 來解 (Global Scale/Bias)
        x1 = term1.view(term1.shape[0], -1) 
        x2 = term2.view(term2.shape[0], -1) 
        y  = target_recon_img.view(target_recon_img.shape[0], -1) 
        
        # 計算矩陣 M 的元素 (Dot products)
        s11 = (x1 * x1).sum(dim=1) # (B,)
        s12 = (x1 * x2).sum(dim=1) # (B,)
        s22 = (x2 * x2).sum(dim=1) # (B,)
        
        # 計算向量 V 的元素
        sy1 = (x1 * y).sum(dim=1)  # (B,)
        sy2 = (x2 * y).sum(dim=1)  # (B,)
        
        # 構建 2x2 矩陣 M 和 2x1 向量 V
        M = torch.stack([
            torch.stack([s11, s12], dim=1),
            torch.stack([s12, s22], dim=1)
        ], dim=1) # (B, 2, 2)
        
        V = torch.stack([sy1, sy2], dim=1).unsqueeze(2) # (B, 2, 1)
        
        # 加入微小雜訊避免 Singular Matrix
        eye = torch.eye(2, device=M.device).unsqueeze(0) * 1e-6
        M = M + eye
        
        # 解方程 [s, b] = inv(M) * V
        M_inv = torch.linalg.inv(M) 
        params = torch.bmm(M_inv, V) # (B, 2, 1)
        
        s_opt = params[:, 0, 0].view(-1, 1, 1, 1) # (B, 1, 1, 1)
        b_opt = params[:, 1, 0].view(-1, 1, 1, 1) # (B, 1, 1, 1)
        
        student_recon_img = term1 * s_opt + term2 * b_opt
    else:
        student_recon_img = pred_ori_alb * gt_illumination

    # --- 3. 計算 Loss (含 Epsilon) ---
    epsilon = loss_cfg.get('epsilon', 0.0)
    diff = torch.abs(student_recon_img - target_recon_img)
    
    if epsilon > 0.0:
        loss_val = torch.clamp(diff - epsilon, min=0.0).mean()
    else:
        loss_val = diff.mean()
    
    final_loss = loss_val * current_weight
    
    return final_loss, final_loss.item()

def calculate_latent_ssim_loss(x, y, C1=0.01**2, C2=0.03**2):
    """
    計算 Latent Space 的 SSIM Loss (1 - SSIM)。
    x, y: (B, 4, H, W) -> 也就是 pred_z0 和 latents
    使用 AvgPool2d 來模擬 SSIM 的局部視窗計算。
    """
    # 使用 3x3 window
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x**2, 3, 1, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    
    # 回傳 1 - SSIM (因為我們要 minimize loss)
    return 1.0 - (ssim_n / ssim_d).mean()


## Config parser
def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet training script with Config.")
    parser.add_argument(
        "--config",
        type=str,
        default="/mnt/HDD3/miayan/paper/scriblit/config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override experiment.output_dir from CLI",
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    
    if args.output_dir is not None:
        # 覆寫 config 中的設定
        print(f"[Info] Overriding output_dir from CLI: {args.output_dir}")
        conf.experiment.output_dir = args.output_dir
    
    # Check for existing config in output_dir
    backup_config_path = os.path.join(conf.experiment.output_dir, "config.yaml")
    
    should_resume = conf.model.resume_from_checkpoint is not None
    backup_exists = os.path.exists(backup_config_path)
    
    if should_resume and backup_exists:
        print(f"[-] Resume detected: '{conf.model.resume_from_checkpoint}'")
        print(f"[-] Found backup config in: {backup_config_path}")
        print(f"[-] Loading configuration from BACKUP file to ensure consistency...")
        
        backup_conf = OmegaConf.load(backup_config_path)
        backup_conf.model.resume_from_checkpoint = conf.model.resume_from_checkpoint
        
        return backup_conf
    return conf



def main(cfg):
    ### 1. Setup Directories & Logging
    tb_dir = os.path.join(cfg.experiment.output_dir, 'tb_summary')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    log_writer = SummaryWriter(tb_dir)
    
    n_img = 3    
    logging_dir = Path(cfg.experiment.output_dir, cfg.experiment.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.experiment.output_dir, logging_dir=logging_dir)
    
    
    ### 2. Initialize Accelerator (using cfg.training params)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.experiment.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    if accelerator.is_main_process:
        if cfg.experiment.output_dir is not None:
            os.makedirs(cfg.experiment.output_dir, exist_ok=True)
            # Backup config
            OmegaConf.save(cfg, os.path.join(cfg.experiment.output_dir, "config.yaml"))
            
            
    ### 3. Load Models & Tokenizer
    if cfg.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name, revision=cfg.model.revision, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="tokenizer", revision=cfg.model.revision, use_fast=False,
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(cfg.model.pretrained_model_name_or_path, cfg.model.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.model.revision, variant=cfg.model.variant
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.model.pretrained_model_name_or_path, subfolder="vae", revision=cfg.model.revision, variant=cfg.model.variant
    )

    if cfg.model.hf_version is not None:
        unet = UNet2DConditionModel.from_pretrained(cfg.model.hf_repo_id, subfolder=f'{cfg.model.hf_version}/checkpoint-235000/custom_unet', torch_dtype=torch.float32)
        cond_encoder = CustomEncoder.from_pretrained(cfg.model.hf_repo_id, subfolder=f'{cfg.model.hf_version}/checkpoint-235000/custom_encoder', torch_dtype=torch.float32)
        controlnet = ControlNetModel.from_pretrained(cfg.model.hf_repo_id, subfolder=f'{cfg.model.hf_version}/checkpoint-235000/controlnet', torch_dtype=torch.float32)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="unet", revision=cfg.model.revision, variant=cfg.model.variant
        )
        if cfg.model.get('enable_ambient_cond', False):
            logger.info("Initializing CustomEncoderAmb (Intensity + Ambient + Color)")
            cond_encoder = CustomEncoderAmb(int(unet.config.cross_attention_dim), K=cfg.light_encoder.K, Bi=cfg.light_encoder.Bi, Bc=cfg.light_encoder.Bc, Ba=cfg.light_encoder.Ba)
        else:
            logger.info("Initializing CustomEncoderV1 (Intensity + Color)")
            cond_encoder = CustomEncoderV1(int(unet.config.cross_attention_dim), K=cfg.light_encoder.K, Bi=cfg.light_encoder.Bi, Bc=cfg.light_encoder.Bc)

        # Initialize controlnet from unet
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=6)
        
    albedo_wrapper = None
    if "albedo_estimator" in cfg:
        if cfg.albedo_estimator.enabled:
            logger.info("Initializing Albedo Estimator for fine-tuning...")
            albedo_wrapper = AlbedoWrapper(cfg.albedo_estimator)
            albedo_wrapper.to(accelerator.device)
        
    
    ### 4. Custom Saving Hooks
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    unwrapped = unwrap_model(accelerator, model)
                    
                    if isinstance(unwrapped, ControlNetModel):
                        sub_dir = "controlnet"
                    elif unwrapped.__class__.__name__ == 'CustomEncoder':
                        sub_dir = "custom_encoder"
                    elif isinstance(unwrapped, AlbedoWrapper):
                        sub_dir = "albedo_estimator"
                    else:
                        sub_dir = None

                    if sub_dir:
                        unwrapped.save_pretrained(os.path.join(output_dir, sub_dir))
                    i -= 1

        def load_model_hook(models, input_dir):
            i = len(models) - 1
            while i >= 0:
                model = models[i]
                unwrapped = unwrap_model(accelerator, model)

                if isinstance(unwrapped, ControlNetModel):
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                    unwrapped.register_to_config(**load_model.config)
                    unwrapped.load_state_dict(load_model.state_dict())
                    del load_model
                    models.pop(i)
                elif unwrapped.__class__.__name__ == 'CustomEncoder':
                    load_model = unwrapped.__class__.from_pretrained(input_dir, subfolder="custom_encoder")
                    unwrapped.register_to_config(**load_model.config)
                    unwrapped.load_state_dict(load_model.state_dict())
                    del load_model
                    models.pop(i)
                elif isinstance(unwrapped, AlbedoWrapper):
                    load_model = AlbedoWrapper.from_pretrained(input_dir, subfolder="albedo_estimator", low_cpu_mem_usage=False)
                    unwrapped.register_to_config(**load_model.config)
                    unwrapped.load_state_dict(load_model.state_dict(), strict=False)
                    del load_model
                    models.pop(i)
                i -= 1

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        
        
    # 5. Model Configuration (Freeze/Unfreeze)
    vae.requires_grad_(False)
    if 8 != unet.config["in_channels"]:
        replace_unet_conv_in(unet)
   
    if cfg.model.hf_version is not None:
        # Load custom pretrained unet if path provided
        unet_path = "./%s/custom_unet.pth" % cfg.model.pretrain_unet_path
        unet.load_state_dict(torch.load(unet_path), strict=False)

    unet.requires_grad_(False)
    for name, p in unet.named_parameters():
        if name.startswith(("conv_in", "mid_block.attentions", "mid_block.resnets", "down_blocks.0.resnets", "up_blocks.3.resnets")):
            p.requires_grad = True
        elif ("attentions" in name or "attn" in name) and "transformer_blocks" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    text_encoder.requires_grad_(False)
    controlnet.train()

    # Enable xformers & gradient checkpointing
    if is_xformers_available(): # xformers usually default enabled in diffusers logic, checking if available
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

    # # Note: enable_gradient_checkpointing is boolean in config? usually standard to enable if true
    # controlnet.enable_gradient_checkpointing()
    # unet.enable_gradient_checkpointing()
    
    
    ### 6. Optimizer (using cfg.training params)
    if cfg.training.scale_lr:
        cfg.training.learning_rate = (
            cfg.training.learning_rate * cfg.training.gradient_accumulation_steps * cfg.training.batch_size * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW
    if cfg.training.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")

    params_to_optimize = list(controlnet.parameters()) + list(cond_encoder.parameters())
    trainable_unet_params = [p for p in unet.parameters() if p.requires_grad]
    if len(trainable_unet_params) > 0:
        params_to_optimize += trainable_unet_params
        
    if albedo_wrapper is not None:
        params_to_optimize += list(albedo_wrapper.get_trainable_parameters())

    optimizer = optimizer_class(
        params_to_optimize,
        lr=cfg.training.learning_rate,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.training.adam_weight_decay,
        eps=cfg.training.adam_epsilon,
    )
    
    
    ### 7. Dataloader (using cfg.training.batch_size)
    train_dataset = Indoor_dataset(tokenizer, cfg.data)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        num_workers=cfg.data.dataloader_num_workers, 
        batch_size=cfg.training.batch_size, 
        shuffle=True
    )
    
    
    ### 8. Scheduler (using cfg.training params)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if cfg.training.max_train_steps is None:
        cfg.training.max_train_steps = cfg.training.num_epochs * num_update_steps_per_epoch
    
    cfg.training.num_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.training.max_train_steps * accelerator.num_processes,
    )

    # Prepare with Accelerator
    if albedo_wrapper is not None:
        controlnet, cond_encoder, optimizer, train_dataloader, lr_scheduler, albedo_wrapper = accelerator.prepare(
            controlnet, cond_encoder, optimizer, train_dataloader, lr_scheduler, albedo_wrapper
        )
    else:
        controlnet, cond_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, cond_encoder, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.experiment.project_name)
        
        
    # 9. Load Aux Models (Only if Recon Loss is enabled)
    albedo_model, normal_model, depth_pipe = None, None, None
    keep_aux_on_gpu = False
    if 'keep_aux_models_on_gpu' in cfg.losses.recon_loss:
        keep_aux_on_gpu = cfg.losses.recon_loss.keep_aux_models_on_gpu    # default to False (move to cpu)
    if cfg.losses.recon_loss.enabled:
        logger.info("***** Loading Aux Models for Reconstruction Loss *****")            
        from transformers import pipeline as depth_pipeline
        # Ensure paths match your system
        normal_model = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
        _move_to_device(normal_model, 'cpu')
        depth_pipe = depth_pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device='cpu')
        if albedo_wrapper is not None:
            keep_aux_on_gpu = True
            albedo_model = albedo_wrapper.models_dict
            logger.info("Shared Albedo Model from AlbedoWrapper for Recon Loss.")
        else:
            albedo_model = load_models(path='v2', model_dir='/mnt/HDD7/miayan/paper/scriblit/dataset/iid', device='cpu')
        
        if keep_aux_on_gpu:
            logger.info(f"NOTE: Auxiliary models will be kept on GPU {accelerator.device} for speed.")
            _move_to_device(albedo_model, accelerator.device)
            _move_to_device(normal_model, accelerator.device)
            depth_pipe.model.to(accelerator.device)
            depth_pipe.device = accelerator.device
        else:
            logger.info("NOTE: Auxiliary models will be offloaded to CPU to save VRAM (Slow).")
    else:
        logger.info("***** Recon Loss Disabled: Skipping Aux Models *****")
        
        
    ### 10. Training State
    global_step = 0
    first_epoch = 0

    if cfg.model.resume_from_checkpoint:
        if cfg.model.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.model.resume_from_checkpoint)
        else:
            dirs = os.listdir(cfg.experiment.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.experiment.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
        else:
             accelerator.print("Checkpoint not found. Starting new training.")

    # [Log] Training Information Logging
    total_batch_size = cfg.training.batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {cfg.training.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.training.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.training.max_train_steps}")
    logger.info(f"  Noise Scheduler prediction type = {noise_scheduler.config.prediction_type}")

    # Loss Accumulator
    accumulated_loss = {k: 0.0 for k in cfg.losses.keys()}
    accumulated_loss['total'] = 0.0

    progress_bar = tqdm(range(global_step, cfg.training.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(first_epoch, cfg.training.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                
                # --- Forward Pass ---
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                
                # Use predicted albedo from Albedo Estimator during training at a rate of 50%
                if (cfg.albedo_estimator.enabled and albedo_wrapper is not None) and (random.random() < 0.5):
                    with torch.no_grad():
                        pred_albedo = albedo_wrapper((batch["ori_img"].to(dtype=weight_dtype) + 1.0) / 2.0)
                        pred_albedo = pred_albedo * 2.0 - 1.0  # Scale back to [-1, 1]
                    albedo_latents = vae.encode(pred_albedo).latent_dist.sample() * vae.config.scaling_factor
                else:
                    albedo_latents = vae.encode(batch["albedo"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                albedo_noise = torch.randn_like(albedo_latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps_albedo = torch.randint(200, 201, (bsz,), device=latents.device)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                albedo_latents = noise_scheduler.add_noise(albedo_latents, albedo_noise, timesteps_albedo)

                if cfg.model.get('enable_ambient_cond', False):
                    encoder_hidden_states = cond_encoder(batch['intensity'], batch['ambient'], batch['color'])
                else:
                    encoder_hidden_states = cond_encoder(batch['intensity'], batch['color'])
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample, controlnet_cond_recon = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=controlnet_image, return_dict=False
                )

                cat_latents = torch.cat([noisy_latents, albedo_latents], dim=1).to(dtype=weight_dtype)

                model_pred = unet(
                    cat_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # [Memory] Clear intermediate tensors after UNet forward
                del down_block_res_samples, mid_block_res_sample, cat_latents, encoder_hidden_states, controlnet_image

                pred_z0 = pred_x0_latents(noise_scheduler, model_pred, noisy_latents, timesteps)
                
                # [Memory] noisy_latents no longer needed after pred_z0 calculation
                del noisy_latents
                torch.cuda.empty_cache()

                # --- Loss Calculation ---
                total_loss = 0.0
                
                # 1. Physical Loss
                if cfg.losses.phys_loss.enabled:
                    phys_loss = F.mse_loss(pred_z0.float(), latents.float(), reduction="mean")
                    total_loss += phys_loss * cfg.losses.phys_loss.weight
                    accumulated_loss['phys_loss'] += phys_loss.item()
                    
                # Latent SSIM Loss
                if cfg.losses.get('ssim_loss', {}).get('enabled', False):
                    # pred_z0: 預測的去噪 latent
                    # latents: 真實的 GT latent
                    loss_ssim = calculate_latent_ssim_loss(pred_z0.float(), latents.float())
                    
                    w_ssim = cfg.losses.ssim_loss.get('weight', 0.2)
                    total_loss += loss_ssim * w_ssim
                    accumulated_loss['ssim_loss'] = accumulated_loss.get('ssim_loss', 0.0) + loss_ssim.item()

                # Lazy decode
                I_relit = None
                if (cfg.losses.area_loss.enabled or cfg.losses.intensity_loss.enabled or cfg.losses.recon_loss.enabled or (cfg.albedo_estimator.enabled and albedo_wrapper is not None)):
                    I_relit = vae.decode(pred_z0.detach() / vae.config.scaling_factor).sample
                    pseudo_gt = batch["pixel_values"].to(dtype=weight_dtype)
                    
                    # [Memory] pred_z0 is no longer needed after decoding (unless Latent loss needs it, but it uses model_pred)
                    del pred_z0 

                # 2. Area Loss
                if cfg.losses.area_loss.enabled:
                    mask = batch["mask"].to(dtype=weight_dtype).repeat(1, 3, 1, 1)
                    masked_I_relit = I_relit * mask
                    masked_pseudo_gt = pseudo_gt * mask
                    diff = (masked_I_relit - masked_pseudo_gt).abs()
                    denom = mask.sum(dim=(1,2,3)) + 1e-8
                    per_sample = diff.sum(dim=(1,2,3)) / denom
                    area_loss = per_sample.mean()
                    
                    total_loss += area_loss * cfg.losses.area_loss.weight
                    accumulated_loss['area_loss'] += area_loss.item()
                    
                    # [Memory] Clean Area Loss specific vars
                    del diff, denom, per_sample, mask, masked_pseudo_gt
                    torch.cuda.empty_cache()

                # 3. Intensity Loss
                if cfg.losses.intensity_loss.enabled:
                    lightmap_gt = batch["lightmap"].to(dtype=weight_dtype)
                    lightmap_pred = calculate_pred_lightmap(I_relit, pseudo_gt)
                    intensity_loss = F.l1_loss(lightmap_pred, lightmap_gt)
                    
                    total_loss += intensity_loss * cfg.losses.intensity_loss.weight
                    accumulated_loss['intensity_loss'] += intensity_loss.item()

                    # [Memory] Clean Intensity Loss specific vars
                    del lightmap_pred, lightmap_gt
                    # pseudo_gt is shared, so only delete if not needed by others, 
                    # but here we keep it simple or delete if it's the last usage.
                    del pseudo_gt 
                    torch.cuda.empty_cache()
                
                # ** Albedo Estiamtor Fine-tuning Supervised Loss **
                if cfg.albedo_estimator.enabled and albedo_wrapper is not None:
                    gt_ori_alb = (batch["albedo"].to(dtype=weight_dtype) + 1.0) / 2.0
                    input_ori_img = (batch["ori_img"].to(dtype=weight_dtype) + 1.0) / 2.0
                    
                    pred_ori_alb = albedo_wrapper(input_ori_img)
                    
                    loss_est_total = 0.0
                    # Structure-only Supervision:
                    # transform student and teacher albedo to gray-scale to reduce color bias
                    # learn image content structure
                    loss_cfg = cfg.losses.get('sup_structure_loss', {'enabled': False})
                    if loss_cfg.get('enabled', False):
                        pred_luma = get_luminance(pred_ori_alb)
                        gt_luma = get_luminance(gt_ori_alb)
                        img_luma = get_luminance(input_ori_img)
                        confidence_weight = torch.clamp(img_luma, 0.1, 1.0) ** 2    # 0.0 -> 0.1 讓很暗的房間區域也有一點權重
                    
                        # weighted loss between pseudo gt albedo (predict by teacher model) and predicted albedo (predict by student model)
                        loss_sup_structure = F.l1_loss(pred_luma, gt_luma, reduction='none')
                        loss_sup_val = (loss_sup_structure * confidence_weight).mean()
                        
                        # [權重衰減邏輯]
                        current_weight = loss_cfg.get('weight', 1.0)
                        if loss_cfg.get('enable_decay', False):
                            w_start = loss_cfg.get('start_weight', 1.0)
                            w_end = loss_cfg.get('end_weight', 0.0)
                            d_steps = loss_cfg.get('decay_steps', 5000)
                            
                            if global_step < d_steps:
                                progress = global_step / d_steps
                                current_weight = w_start - (w_start - w_end) * progress
                            else:
                                current_weight = w_end
                        
                        loss_est_total += loss_sup_val * current_weight
                        
                        if accelerator.is_main_process and global_step % 100 == 0:
                            accelerator.log({"struct_w": current_weight}, step=global_step)
                    
                    # Physics Reconstruction Loss (learn image content color)
                    # Original_Image ≈ Pred_Albedo * (Lightmap_Intensity * Light_Color)
                    loss_self_recon, loss_self_recon_item = compute_self_recon_loss(
                        cfg, batch, pred_ori_alb, gt_ori_alb, 
                        global_step, num_update_steps_per_epoch, weight_dtype
                    )
                    
                    loss_est_total += loss_self_recon
                    accumulated_loss['recon_self'] = accumulated_loss.get('recon_self', 0.0) + loss_self_recon_item
                    
                    # Consistency loss between original image predicted albedo and relit image predicted albedo
                    loss_cfg = cfg.losses.get('consistency_loss', {'enabled': False})
                    if loss_cfg.get('enabled', False):
                        input_relit_img = (I_relit.detach() + 1.0) / 2.0
                        pred_relit_alb = albedo_wrapper(input_relit_img)
                        loss_cons = F.mse_loss(pred_ori_alb, pred_relit_alb)
                        
                        w_cons = loss_cfg.get('weight', 1.0)
                        loss_est_total += loss_cons * w_cons
                    
                    total_loss += loss_est_total
                    accumulated_loss['albedo_est'] = accumulated_loss.get('albedo_est', 0.0) + loss_est_total.item()
                    if isinstance(loss_self_recon, torch.Tensor):
                        accumulated_loss['recon_self'] = accumulated_loss.get('recon_self', 0.0) + loss_self_recon.item()
                    del input_ori_img, pred_ori_alb
                    if 'loss_sup_structure' in locals(): del loss_sup_structure
                    if 'pred_relit_alb' in locals(): del pred_relit_alb

                # 4. Recon Loss
                if cfg.losses.recon_loss.enabled:
                    should_offload = not keep_aux_on_gpu
                    
                    # Albedo
                    if albedo_wrapper is not None:
                        input_relit_img = (I_relit + 1.0) / 2.0
                        batch_inputs = []
                        for b_idx in range(input_relit_img.shape[0]):
                            processed = albedo_wrapper._preprocess_frozen_stages(input_relit_img[b_idx])
                            batch_inputs.append(processed)
                        batch_model_inputs = torch.cat(batch_inputs, dim=0) # (B, 9, H, W)
                        
                        diff_model_input = torch.cat([input_relit_img, batch_model_inputs[:, 3:, ...]], dim=1)
                        recon_alb = albedo_wrapper.train_model(diff_model_input)
                        gt_alb = (batch["albedo"].to(dtype=weight_dtype) + 1.0) / 2.0
                        alb_loss = F.mse_loss(recon_alb, gt_alb)
                        del batch_model_inputs, diff_model_input, recon_alb, gt_alb
                    else:
                        recon_alb = gen_albedo(alb_to_pil(I_relit), albedo_model, latents.device, offload=should_offload)
                        if recon_alb.shape[-1] != cfg.data.resolution:
                            recon_alb = F.interpolate(recon_alb, size=(cfg.data.resolution, cfg.data.resolution), mode='bilinear', align_corners=False)
                        recon_alb_latents = vae.encode(recon_alb.to(dtype=weight_dtype).to(latents.device)).latent_dist.sample() * vae.config.scaling_factor
                        alb_loss = F.mse_loss(recon_alb_latents.float(), albedo_latents.float(), reduction="mean")
                        del recon_alb, recon_alb_latents # Immediate cleanup
                    
                    # Normal
                    recon_normal = normal_estimation_sn(alb_to_pil(I_relit), normal_model, latents.device, offload=should_offload)
                    recon_normal = img_transforms(recon_normal, 'side_cond').unsqueeze(0).to(dtype=weight_dtype).to(latents.device)
                    if recon_normal.shape[-1] != cfg.data.resolution:
                        recon_normal = F.interpolate(recon_normal, size=(cfg.data.resolution, cfg.data.resolution), mode='bilinear', align_corners=False)
                    normal_loss = F.mse_loss(recon_normal.float(), batch["normal"].to(dtype=weight_dtype).float().to(latents.device), reduction="mean")
                    del recon_normal # Immediate cleanup
                    
                    # Depth
                    recon_depth = depth_estimation(alb_to_pil(I_relit), depth_pipe, latents.device, offload=should_offload)
                    recon_depth = img_transforms(recon_depth, 'side_cond').unsqueeze(0).to(dtype=weight_dtype).to(latents.device)
                    if recon_depth.shape[-1] != cfg.data.resolution:
                        recon_depth = F.interpolate(recon_depth, size=(cfg.data.resolution, cfg.data.resolution), mode='bilinear', align_corners=False)
                    depth_loss = F.mse_loss(recon_depth.float(), batch["depth"].to(dtype=weight_dtype).float().to(latents.device), reduction="mean")
                    del recon_depth # Immediate cleanup

                    recon_loss_val = alb_loss + normal_loss + depth_loss
                    total_loss += recon_loss_val * cfg.losses.recon_loss.weight
                    accumulated_loss['recon_loss'] += recon_loss_val.item()
                    
                    # [Memory] Cleanup Recon
                    del alb_loss, normal_loss, depth_loss
                    torch.cuda.empty_cache()

                # 5. Latent Loss
                if cfg.losses.latent_loss.enabled:
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise
                        
                    latent_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    total_loss += latent_loss * cfg.losses.latent_loss.weight
                    accumulated_loss['latent_loss'] += latent_loss.item()

                # 6. Image Loss
                if cfg.losses.image_loss.enabled:
                    control_target = batch["control_target"].to(dtype=weight_dtype)
                    image_loss = F.mse_loss(controlnet_cond_recon, control_target)
                    total_loss += image_loss * cfg.losses.image_loss.weight
                    accumulated_loss['image_loss'] += image_loss.item()

                accumulated_loss['total'] += total_loss.item()

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, cfg.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=cfg.training.set_grads_to_none)

                # [Memory] Final cleanup for step
                # Check what still exists in locals() before deleting to avoid errors if some blocks were skipped
                if 'model_pred' in locals(): del model_pred
                if 'latents' in locals(): del latents
                if 'albedo_latents' in locals(): del albedo_latents
                if 'I_relit' in locals() and I_relit is not None: del I_relit
                if 'controlnet_cond_recon' in locals(): del controlnet_cond_recon
                torch.cuda.empty_cache()

            # End of Step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process and global_step % cfg.training.checkpointing_steps == 0:
                    for loss_name, loss_val in accumulated_loss.items():
                        log_writer.add_scalars(loss_name.replace('_loss', ' Loss').title(), {'train': loss_val / cfg.training.checkpointing_steps}, global_step)
                        accumulated_loss[loss_name] = 0.0
                    
                    write_tb_log_source(batch["ori_img"], 'img_src', n_img, log_writer, global_step)
                    write_tb_log_source(batch['pixel_values'], 'pseudo_GT', n_img, log_writer, global_step)
                    write_tb_log(batch["conditioning_pixel_values"][:,:3,:,:], 'normal', n_img, log_writer, global_step)
                    write_tb_log(batch["conditioning_pixel_values"][:,3:6,:,:], 'cond_lightmap', n_img, log_writer, global_step)
                    
                    if cfg.training.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(cfg.experiment.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= cfg.training.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - cfg.training.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(cfg.experiment.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(cfg.experiment.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    
                    unet_save = unwrap_model(accelerator, unet)
                    torch.save(unet_save.state_dict(), os.path.join(save_path, 'custom_unet.pth'))

            logs = {"loss": total_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= cfg.training.max_train_steps:
                break
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        last_save_path = os.path.join(cfg.experiment.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(last_save_path)
        logger.info(f"Saved final training state to {last_save_path}")
        
        # 補上 Custom UNet 的手動儲存 (因為它不在你的 save_model_hook 裡面)
        unet_save = unwrap_model(accelerator, unet)
        torch.save(unet_save.state_dict(), os.path.join(last_save_path, 'custom_unet.pth'))
        
        controlnet = unwrap_model(accelerator, controlnet)
        controlnet.save_pretrained(cfg.experiment.output_dir)
        
        save_model_card(
            cfg.experiment.project_name,
            base_model=cfg.model.pretrained_model_name_or_path,
            repo_folder=cfg.experiment.output_dir,
        )

    accelerator.end_training()
    
    

if __name__ == "__main__":
    conf = parse_args()
    main(conf)
