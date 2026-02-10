import os
import argparse
import torch
import numpy as np
import csv
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from datasets import load_dataset, Image as HfImage

# [新增] 品質指標庫 (pyiqa)
import pyiqa

# ==========================================
# 1. 初始化指標模型
# ==========================================
# 分數越低代表影像品質越自然、失真越少
niqe_metric = pyiqa.create_metric('niqe', device=torch.device('cuda'))
brisque_metric = pyiqa.create_metric('brisque', device=torch.device('cuda'))

# ==========================================
# 2. 指標計算函數
# ==========================================
def _calculate_niqe(img_pil):
    """計算單張圖片的 NIQE"""
    img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to('cuda')
    with torch.no_grad():
        return niqe_metric(img_tensor).item()

def _calculate_brisque(img_pil):
    """計算單張圖片的 BRISQUE"""
    img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to('cuda')
    with torch.no_grad():
        return brisque_metric(img_tensor).item()

# ==========================================
# 3. 核心執行邏輯
# ==========================================
def run_quality_check(args, dataset, indices):
    print(f"--- [Quality Check] Mode: {args.mode} ---")
    out_dir = f'./inference/quality_report/{args.mode}'
    os.makedirs(out_dir, exist_ok=True)

    results = []
    total_niqe = 0.0
    total_brisque = 0.0
    count = 0

    pbar = tqdm(indices, desc="Calculating Metrics")
    for idx in pbar:
        item = dataset[idx]
        # 取得影像 (對齊你原本 Dataset 的 key 'image')
        img_pil = item['image'].convert('RGB').resize((512, 512))

        # 計算指標
        n_score = _calculate_niqe(img_pil)
        b_score = _calculate_brisque(img_pil)

        total_niqe += n_score
        total_brisque += b_score
        count += 1

        results.append({
            'Image_ID': idx,
            'NIQE': f"{n_score:.5f}",
            'BRISQUE': f"{b_score:.5f}"
        })

        pbar.set_postfix({"Avg_N": f"{total_niqe/count:.3f}", "Avg_B": f"{total_brisque/count:.3f}"})

    # 寫入 CSV 報告
    csv_path = os.path.join(out_dir, "quality_metrics.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Image_ID', 'NIQE', 'BRISQUE'])
        writer.writeheader()
        writer.writerows(results)

    # 印出最終報告
    print(f"\n" + "="*40)
    print(f"Finished processing {count} images.")
    print(f"Global Average NIQE:    {total_niqe/count:.5f}")
    print(f"Global Average BRISQUE: {total_brisque/count:.5f}")
    print(f"Report saved to: {csv_path}")
    print("="*40)

# ==========================================
# 4. Main 入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', '--mode', type=str, default='standard', choices=['standard', 'lightlab'])
    parser.add_argument('-data', '--num_data', type=int, default=100, help="要計算的資料數量")
    args = parser.parse_args()

    # 載入數據集 (對齊你原本的路徑與邏輯)
    repo = "Miayan/physical-relighting-dataset" if args.mode == 'standard' else "Miayan/physical-relighting-eval-dataset"
    print(f"Loading Dataset: {repo}")
    
    ds = load_dataset(repo, split="train", cache_dir="/mnt/HDD3/miayan/paper/relighting_datasets/")
    
    # 確保圖片正確解碼
    ds = ds.cast_column("image", HfImage(decode=True))
            
    # 設定範圍
    if args.mode == 'lightlab':
        # LightLab 特定 ID 邏輯
        indices = [181, 16, 75, 77][:args.num_data + 1]
    else:
        indices = range(min(args.num_data, len(ds)))

    run_quality_check(args, ds, indices)