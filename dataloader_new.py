import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import random
from PIL import Image, ImageFilter
import numpy as np
from omegaconf import OmegaConf

from datasets import load_from_disk, Image as HfImage, Dataset as DS, load_dataset


class Indoor_dataset(Dataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        
        # === 0. Relighting Implementation Version 選擇 ===
        self.relighting_impl = cfg.get('relighting_impl', 'v2')
        print(f"Using Relighting Implementation: {self.relighting_impl}")
        
        if self.relighting_impl == 'gemini_amb':
            import physical_relighting_api_v2_gemini_amb as api
            self.compute_relighting = api.compute_relighting
            self.PhysicalRelightingConfig = api.PhysicalRelightingConfig
            self.returns_ambient = True # 這個版本會回傳 (img, map, amb)
        elif self.relighting_impl == 'gemini':
            import physical_relighting_api_v2_gemini as api
            self.compute_relighting = api.compute_relighting
            self.PhysicalRelightingConfig = api.PhysicalRelightingConfig
            self.returns_ambient = False # 你的 gemini v2 檔案目前只回傳 2 個值
        else:
            import physical_relighting_api_v2 as api
            self.compute_relighting = api.compute_relighting
            self.PhysicalRelightingConfig = api.PhysicalRelightingConfig
            self.returns_ambient = False
            
        
        # === 1. 讀取 Dataset ===
        # 對應 YAML: data_hf_repo_id, data_split, dataset_cache_dir
        repo_id = cfg.data_hf_repo_id
        split = cfg.data_split
        cache_dir = cfg.dataset_cache_dir
        
        print(f"Loading {repo_id} ({split}) from {cache_dir}...")
        self.ds: DS = load_dataset(repo_id, split=split, cache_dir=cache_dir)
        for col in self.ds.column_names:
            if col not in ('color', 'intensity', 'prompt'):
                self.ds = self.ds.cast_column(col, HfImage(decode=True))
                
        # === 2. Resolution 設定 ===
        resolution = cfg.resolution
        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.conditioning_image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        # === 3. Colors 設定 ===
        colors_list = OmegaConf.to_container(cfg.colors, resolve=True) if hasattr(cfg.colors, 'to_container') else cfg.colors
        self.colors = torch.tensor(colors_list, dtype=torch.float32) / 255.0
        
        # === 4. Intensities 設定 ===
        intensities_list = OmegaConf.to_container(cfg.intensities, resolve=True) if hasattr(cfg.intensities, 'to_container') else cfg.intensities
        self.intensity = torch.tensor(intensities_list, dtype=torch.float32)
        
        # === 5. Condition Mode 設定 ===
        self.condition_mode = cfg.get('condition_mode', 'lightmap_normal')
        print(f"Dataset Condition Mode: {self.condition_mode}")
        
        # === 6. 讀取 ControlNet 設定 ===
        # 預設為 True (推薦)，即使用 RGB Lightmap
        self.use_ambient_in_controlnet = cfg.get('use_ambient_in_controlnet', True)
        print(f"ControlNet uses Ambient in Lightmap: {self.use_ambient_in_controlnet}")
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        
        if np.array(item['lightmap']).max() != 0:
            ori = item['image']
            albedo = item['albedo']
            normal = item['normal']
            prompt = item['prompt']
            
            depth = item['depth'].convert('L')
            mask = item['mask'].convert('L')

            intensity = self.intensity[idx % len(self.intensity)]
            color = self.colors[idx % len(self.colors)]

            config = self.PhysicalRelightingConfig(ori, normal, depth)
            config.add_mask(mask, color, intensity)
            result = self.compute_relighting(config)
            if self.returns_ambient:
                ambient_scalar = result['ambient']
                lightmap_rgb = result['lightmap_rgb'].convert('RGB')
                lightmap_rgb_tensor = self.conditioning_image_transforms(lightmap_rgb)
            else:
                lightmap_rgb_tensor = None
                ambient_scalar = getattr(config, 'ambient', 0.75) # 預設值
            
            # 根據 Config 選擇 ControlNet 的 Lightmap
            if self.use_ambient_in_controlnet:
                # Case A: 使用「含 Ambient」的 (灰底)
                # 對應 config: true
                lightmap = result['lightmap'].convert('RGB')
            else:
                # Case B: 使用「不含 Ambient」的 (黑底)
                # 對應 config: false (推薦)
                # 使用 .get() 做 fallback，防止舊版 API 沒有 lightmap_raw 報錯
                lightmap = result.get('lightmap_raw', result['lightmap']).convert('RGB')
            
            
            # Transforms
            normal_tensor = self.conditioning_image_transforms(normal)
            depth_tensor = self.conditioning_image_transforms(depth)
            mask_tensor = self.conditioning_image_transforms(mask)
            lightmap_tensor = self.conditioning_image_transforms(lightmap)
            
            ori_tensor = self.image_transforms(ori)
            albedo_tensor = self.image_transforms(albedo)
            i_phys_tensor = self.image_transforms(result['image'])
            
            color = torch.tensor(color, dtype=torch.float32)
            intensity = torch.tensor([intensity], dtype=torch.float32)
            ambient = torch.tensor([ambient_scalar], dtype=torch.float32)
            
            if self.condition_mode == "lightmap_normal":
                # 模式 1: Normal + Lightmap
                source = torch.cat((normal_tensor, lightmap_tensor), dim=0)
                control_target = torch.cat((normal_tensor, lightmap_tensor), dim=0)
            elif self.condition_mode == "depth_mask_normal":
                # 模式 2: Normal + Depth + Mask + (Depth*Mask)
                DM = depth_tensor * mask_tensor
                source = torch.cat((normal_tensor, depth_tensor, mask_tensor, DM), dim=0)
                control_target = torch.cat((normal_tensor, depth_tensor, mask_tensor, DM), dim=0)
                
            prompt = self.tokenizer(
				prompt, max_length=self.tokenizer.model_max_length-4, padding="max_length", truncation=True, return_tensors="pt"
			)
            
            return dict(
				pixel_values=i_phys_tensor, 
				input_ids=prompt.input_ids, 
				albedo=albedo_tensor, 
				intensity=intensity, 
				ambient=ambient, 
				color=color, 
				conditioning_pixel_values=source, 
				control_target=control_target, 
				ori_img=ori_tensor, 
				normal=normal_tensor, 
				depth=depth_tensor, 
				mask=mask_tensor, 
				lightmap=lightmap_tensor,
                lightmap_rgb=lightmap_rgb_tensor
			)

