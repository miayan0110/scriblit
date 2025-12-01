## Checkpoint discription

### train_v1
- 描述：只 train controlnet 和 custom encoder，用 lightmap 製造 scribble
- 問題：用 lightmap 製造出來的 scribble 有些是全黑的，因為 lightmap 不一定會有全亮 (白色) 區域，感覺 condition 不能用 scribble
- training script：train_controlnet.py

### train_v2
- 描述：把 v1 的 scribble 換成用 lightmap 直接當 condition 輸入
- 問題：因為目前的架構把 text encoder 拿掉了，所以感覺好像不能只 train controlnet
- training script：train_controlnet.py

### train_v3
- 描述：除了 controlnet 和 custom encoder，還把 unet.conv_in 打開一起訓練
- 問題：其實我不知道為啥 gpt 要建議我打開 unet.conv_in，不過目前效果不好，還是會像之前那樣碎碎的
- training script：train_controlnet.py

### train_v4
- 描述：換成把 cross-attn 打開訓練，unet.conv_in 關掉了，感覺這是目前最合理的訓練方式
- 問題：訓練很不穩定
- training script：train_controlnet.py

### train_v5 [tmux v3] 
- 描述：換了新的 cycle，目前嘗試沒 cycle 的版本
- 問題：不知道為啥最後學出來的Irelit都跟原圖一樣
- training script：train_controlnet_ori.py

### train_v5_2
- 描述：把 unet target 改成 pseudo gt
- 問題：很糊
- training script：train_controlnet_ori.py

### train_ex1 [tmux ex1]
- 描述：把 x0 也改成 pseudo gt 看看
- training script：train_controlnet_ex1.py

### train_ex2
- 描述：加上 recon loss (albedo)
- 問題：很糊、不知道為啥 OOM 了
- training script：train_controlnet_ex1.py

### train_ex2_2
- 描述：加上 recon loss (全部加好了)
- training script：train_controlnet_ex1.py

### train_ex2_3
- 描述：ex2_2 把 text prompt 加回去
- training script：train_controlnet_ex2_3.py

### train_ex2_4
- 描述：ex2_3 把所有新加的 loss 拿掉
- training script：train_controlnet_ex2_4.py

### train_ex2_5
- 描述：ex2_4 的 lightmap 換掉，改成 normal, depth, mask
- training script：train_controlnet_ex2_4.py

### train_ex3
- 描述：把 prompt 拿掉，lightmap 換成 normal, depth, mask
- training script：train_controlnet_ex3.py, dataloader_ex3.py

### train_ex3_1
- 描述：在 ex_3 加上針對 mask 區域顏色的 loss
- training script：train_controlnet_ex3.py, dataloader_ex3.py

### train_ex3_2
- 描述：把 ex_3_1 的 cond_loss*10
- training script：train_controlnet_ex3.py, dataloader_ex3.py


### lightmapper [tmux v4]


## Data preprocessing

### Depth & Normal
```python
cd /mnt/HDD7/miayan/paper/DSINE/projects/dsine
conda activate /mnt/HDD7/miayan/paper/envs/DSINE
CUDA_VISIBLE_DEVICES=6 python test_minimal.py ./experiments/exp001_cvpr2024/dsine.txt
```

### Mask
```
cd /mnt/HDD7/miayan/paper/Grounded-Segment-Anything
conda activate /mnt/HDD7/miayan/paper/envs/gsam
CUDA_VISIBLE_DEVICES=7 python gen_light_masks.py
```

### Albedo
```
cd /mnt/HDD7/miayan/paper/scriblit/dataset
conda activate /mnt/HDD7/miayan/paper/envs/cv
python gen_albedo.py
```

### Push to huggingface
```python
cd /mnt/HDD7/miayan/paper/relighting_datasets
conda activate /mnt/HDD7/miayan/paper/envs/cv
python push_to_hub.py
```



CUDA_VISIBLE_DEVICES=7 proxychains4 accelerate launch --main_process_port=0 train_controlnet_ex1.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1"  --output_dir=train_ex1 --pretrain_unet_path=scribblelight_controlnet/checkpoint-10000 --validation_image "./imnormal_1.png" --validation_intensity 1.0 --validation_color 1.0 0.0 0.0

CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port=0 train_controlnet_ex1.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1"  --output_dir=train_ex2_2 --pretrain_unet_path=scribblelight_controlnet/checkpoint-10000 --validation_image "./imnormal_1.png" --validation_intensity 1.0 --validation_color 1.0 0.0 0.0


# gpu id | real id
#    0   |    2
#    1   |    4
#    2   |    6
#    3   |    0
#    4   |    1
#    5   |    3
#    6   |    5
#    7   |    7