from huggingface_hub import upload_folder
import os
from natsort import natsorted

repo_id = "Miayan/project"

base_dir = "/mnt/HDD3/miayan/paper/scriblit"
experiments = natsorted([ex for ex in os.listdir(base_dir) if ex.startswith("train_ex")])
print("Found experiments:", experiments)
exclude_exp = ["train_ex8_8_bs32",
               "train_ex8_7_bs32",
               "train_ex9_1",
               "train_ex8_11",]

for exp in experiments:
    if exp in exclude_exp:
        print(f"Skipping excluded experiment: {exp}")
        continue
    print(f"Uploading experiment: {exp}")
    # 這裡改成整個 experiment 的資料夾
    folder_path = f"{base_dir}/{exp}"
    path_in_repo = exp  # 上到 HF 後的根目錄：train_ex2_2/ ...

    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        # 只上傳這三個：checkpoint-** 整包 + 同層的兩個檔案
        allow_patterns=[
            "checkpoint-[0-9]*/**",
            "config.yaml",
        ],
        commit_message=f"Add checkpoint for {exp}",
    )