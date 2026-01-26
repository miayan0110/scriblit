import torch
import torch.nn as nn
import numpy as np
from intrinsic.pipeline import load_models, run_pipeline
from chrislib.general import invert
from omegaconf import OmegaConf, DictConfig, ListConfig
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

class AlbedoWrapper(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, cfg):
        """
        cfg: 對應 yaml 裡的 model.albedo_estimator 區塊
        """
        super().__init__()
        if isinstance(cfg, (DictConfig, ListConfig)):
            clean_cfg = OmegaConf.to_container(cfg, resolve=True)
        else:
            clean_cfg = cfg
        self.register_to_config(cfg=clean_cfg)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[AlbedoWrapper] Loading models version {clean_cfg['version']} up to Stage {clean_cfg['load_stage']}...")
        
        # 1. 載入模型 (這會包含所有 stage 的模型)
        # load_models function
        raw_models = load_models(
            clean_cfg['version'], 
            stage=clean_cfg['load_stage'], 
            device=device
        )
        
        # 我們把 raw_models 存成 dictionary，不註冊為 submodule，
        # 避免 PyTorch 自動把它們全部加入 optimizer (我們要手動控制)
        self.models_dict = raw_models
        
        # 2. 設定凍結與訓練
        frozen_list = clean_cfg.get('frozen_components', ["ord_model", "iid_model", "col_model"])
        trainable_list = clean_cfg.get('trainable_components', ["alb_model"])
        
        self.trainable_albedo_net = None
        
        for name, model in self.models_dict.items():
            if name in frozen_list:
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
            
            elif name in trainable_list:
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                # 這是我們要訓練的主角 (Stage 3 Albedo Net)
                self.trainable_albedo_net = model 
                # 把它註冊為 Module，這樣存檔時才會被存到
                self.add_module('trainable_albedo_net', model)
                
        if self.trainable_albedo_net is None:
            raise ValueError("Config 裡沒有指定任何 trainable_components，無法進行 Finetune！")

    @property
    def train_model(self):
        return self.trainable_albedo_net
    
    def get_trainable_parameters(self):
        """
        Helper function: 給 Main Optimizer 用的，只回傳需要更新的參數
        """
        return self.trainable_albedo_net.parameters()

    def _preprocess_frozen_stages(self, img_tensor):
        """
        處理 Stage 0 -> Stage 2 的不可微分部分
        Input: 單張 Tensor (C, H, W) range [0, 1]
        Output: Albedo Net 需要的 Input Tensor (1, 9, H, W)
        """
        target_device = img_tensor.device
        # 自動搬移凍結模型 (因為它們沒有被 accelerator 管理)
        # 檢查第一個凍結模型是否在正確的 device
        if 'ord_model' in self.models_dict:
            first_frozen = self.models_dict['ord_model']
            if next(first_frozen.parameters()).device != target_device:
                # 搬移所有凍結模型到 GPU
                for name, model in self.models_dict.items():
                    if name != 'trainable_albedo_net': # trainable_albedo_net 已經被自動管理了，跳過
                        model.to(target_device)
        
        # 轉成 Numpy (H, W, C) 給 pipeline 吃
        img_tensor = img_tensor.clamp(0.0, 1.0)
        img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
        
        device = str(target_device).split(':')[0]
        # 1. 跑 pipeline 前半段 (Numpy, No Grad)
        # run_pipeline function logic
        with torch.no_grad():
            res_stage2 = run_pipeline(
                self.models_dict, 
                img_np, 
                stage=2, # 只跑到 Chroma
                resize_conf=None, 
                device=device
            )
        
        # 2. 準備 Stage 3 的 Input
        # Line 240-247 logic
        raw_img = res_stage2['lin_img']
        rough_alb = res_stage2['lr_alb']
        rough_shd = res_stage2['lr_shd']
        
        # 轉回 Tensor
        t_img = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        t_rough_alb = torch.from_numpy(rough_alb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        t_rough_shd = torch.from_numpy(rough_shd).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # 組合 Input: [Image, Inverted_Shading, Rough_Albedo]
        # 注意: 如果 chrislib 的 invert 有問題，可以用 1.0 / (x + 1e-5) - 1.0 替代
        model_input = torch.cat([t_img, invert(t_rough_shd), t_rough_alb], dim=1)
        
        return model_input

    def forward(self, images):
        """
        End-to-End Forward Pass
        Input: Batch Images (B, 3, H, W), 數值範圍建議 [0, 1]
        Output: Predicted Albedo (B, 3, H, W)
        """
        batch_inputs = []
        
        # 1. 針對 Batch 裡的每一張圖跑前處理 (因為中間依賴 Numpy pipeline，無法向量化)
        # 這部分雖然是 for loop，但在 Finetune batch size 很小 (1~4) 時效能是可以接受的
        for i in range(images.shape[0]):
            single_img = images[i] # (3, H, W)
            
            # 檢查數值範圍，Intrinsic Pipeline 預期 [0, 1]
            if single_img.min() < 0: # 假設是 [-1, 1]
                single_img = (single_img + 1) / 2.0
            
            # 取得 Stage 3 的輸入特徵
            processed_input = self._preprocess_frozen_stages(single_img)
            batch_inputs.append(processed_input)
            
        # 堆疊回 Batch Tensor: (B, 9, H, W)
        batch_inputs = torch.cat(batch_inputs, dim=0)
        
        # 2. 跑可訓練的 Albedo Model (這步帶有梯度！)
        pred_albedo = self.trainable_albedo_net(batch_inputs)
        
        return pred_albedo