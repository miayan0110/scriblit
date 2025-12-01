from huggingface_hub import upload_folder

repo_id = "Miayan/project"

base_dir = "/mnt/HDD7/miayan/paper/scriblit"
experiments = ["train_ex2_3",
               "train_ex2_4",
               "train_ex2_5", 
               "train_ex3",
               "train_ex3_1",
               "train_ex3_2",
               "train_ex4",
               "train_ex4_1",
               "train_ex5",
               "train_ex5_1",
               "train_ex6"]  # 你有幾個就列幾個

for exp in experiments:
    # 這裡改成整個 experiment 的資料夾
    folder_path = f"{base_dir}/{exp}"
    path_in_repo = exp  # 上到 HF 後的根目錄：train_ex2_2/ ...

    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        # 只上傳這三個：checkpoint-235000 整包 + 同層的兩個檔案
        allow_patterns=[
            "checkpoint-235000/**",                # 粉色框整個資料夾
            "config.json",                         # 紅色框檔案 1
            "diffusion_pytorch_model.safetensors", # 紅色框檔案 2
        ],
        commit_message=f"Add checkpoint-235000 for {exp}",
    )