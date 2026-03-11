import os
from datasets import load_from_disk

def upload_selected_experiments(root_dir, experiment_list, repo_id, token=None):
    """
    root_dir: /mnt/HDD7/miayan/paper/scriblit/inference
    experiment_list: ["train_ex11", "train_ex11-1"]
    """
    
    for ex_folder in experiment_list:
        ex_path = os.path.join(root_dir, ex_folder)
        
        if not os.path.exists(ex_path):
            print(f"❌ 找不到主實驗資料夾: {ex_path}，跳過。")
            continue
            
        # 提取名稱 (例如 train_ex11 -> ex11)
        ex_name = ex_folder.replace('train_', '')
        print(f"🚀 開始處理主實驗: {ex_folder}")

        # 掃描 benchmark 子資料夾
        sub_folders = [f for f in os.listdir(ex_path) if os.path.isdir(os.path.join(ex_path, f))]
        
        for sub in sub_folders:
            if sub.startswith('benchmark_'):
                bench_name = sub.replace('benchmark_', '')
                # dataset 所在的目錄路徑
                target_dataset_dir = os.path.join(ex_path, sub, "phase2_color_hf")
                
                # 檢查該路徑是否存在且包含 dataset 檔案 (通常會有 dataset_info.json)
                if not os.path.exists(target_dataset_dir):
                    print(f"⚠️ 跳過 {sub}: 找不到 phase2_color_hf 資料夾")
                    continue
                
                # 設定 Subset 名稱: ex11_01
                subset_name = f"{ex_name}_{bench_name}_v2"
                print(f"📦 正在從磁碟載入 Dataset: {subset_name}...")

                try:
                    # 1. 從本地磁碟載入現有的 Dataset
                    ds = load_from_disk(target_dataset_dir)
                    
                    print(f"📤 正在上傳至 Hub Subset: {subset_name}...")
                    
                    # 2. 上傳到特定的 Subset (config_name) 和 Split
                    ds.push_to_hub(
                        repo_id,
                        config_name=subset_name,
                        split="train",  # 統一放在 train 分割區
                        token=token,
                        private=False
                    )
                    print(f"✅ {subset_name} 上傳成功！")
                    
                except Exception as e:
                    print(f"❌ 載入或上傳 {subset_name} 時發生錯誤: {e}")

# --- 執行設定 ---
if __name__ == "__main__":
    # 建議先在終端機執行 export HF_TOKEN=你的Token
    # 或者直接在這裡填入字串
    # MY_TOKEN = "xxx"
    
    ROOT_DIR = "/mnt/HDD3/miayan/paper/scriblit/inference"
    # 你想要上傳的實驗清單
    MY_EXPERIMENTS = ["train_ex11"]
    
    TARGET_REPO = "Miayan/test-visualize"
    
    upload_selected_experiments(ROOT_DIR, MY_EXPERIMENTS, TARGET_REPO)