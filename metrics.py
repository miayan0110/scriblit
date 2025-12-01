import os
import csv
from typing import Dict, Optional, List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ------------------------------------------------------------------
# Global config
# ------------------------------------------------------------------

# 這幾個 env flag 只是避免 xformers / flash attention 的相容性問題，
# 不會影響 metric 計算本身。
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DIFFUSERS_USE_PYTORCH_ATTENTION", "1")
os.environ.setdefault("USE_FLASH_ATTENTION_2", "0")
os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")

try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    # 舊版 torch 沒這些設定就略過
    pass

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# metric accumulation for final averages
_METRIC_SUMS: Dict[str, float] = {}
_METRIC_COUNT: int = 0

# cache for CSV sample_ids (per csv_path)
_CSV_ID_CACHE: Dict[str, set] = {}

# 目前只算 normal 幾何的四個指標
SUPPORTED_METRICS = ["normal_mae_deg", "normal_med_deg", "normal_lt_11_25", "normal_lt_30"]


# ------------------------------------------------------------------
# StableNormal prediction helper
# ------------------------------------------------------------------

def normal_estimation(path, predictor):
    """
    用 StableNormal (或相容的 normal predictor) 預測一張 RGB 圖的 normal。

    Args:
        path: 圖片路徑 (str)
        predictor: 例如
            predictor = torch.hub.load("Stable-X/StableNormal",
                                       "StableNormal",
                                       trust_repo=True)

    Returns:
        model 的原始輸出 tensor (通常為 1x3xHxW, 範圍約 [-1,1] 或 [0,1])
    """
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        normal = predictor(img)
    return normal


def _to_unit_normal(x):
    """
    將 normal 轉成 1x3xHxW 的單位向量 tensor，放到 DEVICE 上。

    支援：
      - PIL.Image (StableNormal 回傳的圖)
      - numpy.ndarray
      - torch.Tensor
    """
    # 1) 如果是 PIL.Image
    if isinstance(x, Image.Image):
        arr = np.asarray(x, dtype=np.float32)  # HxWxC 或 HxW

        if arr.ndim == 2:
            # 灰階 -> 3 通道
            arr = np.stack([arr] * 3, axis=-1)   # HxWx3

        # 映射到 [0,1]（有可能本來就是 0~1，但除一次也沒關係）
        if arr.max() > 1.0:
            arr = arr / 255.0

        # HWC -> CHW -> NCHW
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW

    # 2) 如果是 numpy array
    elif isinstance(x, np.ndarray):
        arr = x.astype(np.float32)

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)  # HxWx3

        if arr.ndim == 3 and arr.shape[-1] in (1, 3):  # HWC
            arr = arr.transpose(2, 0, 1)  # CHW
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW
            pass
        else:
            raise ValueError(f"Unsupported numpy normal shape: {arr.shape}")

        t = torch.from_numpy(arr).unsqueeze(0)  # 1x3xHxW

    # 3) 如果是 torch.Tensor
    elif torch.is_tensor(x):
        t = x.detach().float()

        if t.ndim == 4:
            # NCHW 或 NHWC
            if t.shape[1] not in (1, 3) and t.shape[-1] in (1, 3):
                # NHWC -> NCHW
                t = t.permute(0, 3, 1, 2)
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)

        elif t.ndim == 3:
            # CHW 或 HWC
            if t.shape[0] in (1, 3):
                t = t.unsqueeze(0)  # 1xCxHxW
            elif t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
            else:
                raise ValueError(f"Unsupported normal tensor shape: {t.shape}")

        elif t.ndim == 2:
            # HxW -> 1x3xHxW
            t = t.unsqueeze(0).unsqueeze(0)
            t = t.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Unsupported normal tensor ndim: {t.ndim}")

    else:
        raise TypeError(f"Unsupported normal type: {type(x)}")

    # 移到 DEVICE，並做單位化
    t = t.to(DEVICE)
    t = F.normalize(t, dim=1, eps=1e-6)
    return t



# ------------------------------------------------------------------
# Geometry metrics on normals
# ------------------------------------------------------------------

def compute_normal_metrics(gt_img_path, pred_img_path, predictor):
    """
    用同一個 normal predictor 對 GT / relight 圖片預測 normal，
    然後計算它們之間的角度誤差統計。

    回傳:
        {
            "normal_mae_deg": mean angular error (degree),
            "normal_med_deg": median angular error,
            "normal_lt_11_25": 角度 < 11.25° 的像素比例,
            "normal_lt_30":   角度 < 30° 的像素比例,
        }
    """
    # 1) 用 StableNormal 預測兩張圖的 normal
    gt_raw = normal_estimation(gt_img_path, predictor)
    pred_raw = normal_estimation(pred_img_path, predictor)

    # 2) 轉成 1x3xHxW 的單位 normal
    gt_n = _to_unit_normal(gt_raw)
    pred_n = _to_unit_normal(pred_raw)

    # 3) 算角度誤差（degree）
    dot = (gt_n * pred_n).sum(dim=1).clamp(-1.0, 1.0)   # 1xHxW
    ang = torch.acos(dot) * 180.0 / np.pi               # degree

    ang_flat = ang.view(-1)

    mae = ang_flat.mean().item()
    med = ang_flat.median().item()
    pct_11 = (ang_flat < 11.25).float().mean().item()
    pct_30 = (ang_flat < 30.0).float().mean().item()

    return {
        "normal_mae_deg": float(mae),
        "normal_med_deg": float(med),
        "normal_lt_11_25": float(pct_11),
        "normal_lt_30": float(pct_30),
    }


def compute_all_metrics(
    gt_img_path,
    pred_img_path,
    predictor,
    metric_names=None,
):
    """
    一次計算多個 geometry metrics（目前只算 normal 相關），
    並更新全域的累積和與計數，用來之後在全部跑完時算平均。
    """
    global _METRIC_SUMS, _METRIC_COUNT

    if metric_names is None:
        metric_names = SUPPORTED_METRICS

    nm = compute_normal_metrics(gt_img_path, pred_img_path, predictor)

    results: Dict[str, float] = {}
    for name in metric_names:
        if name not in nm:
            raise ValueError(f"Unsupported metric name '{name}'. Supported: {list(nm.keys())}")
        results[name] = nm[name]

    _METRIC_COUNT += 1
    for k, v in results.items():
        _METRIC_SUMS[k] = _METRIC_SUMS.get(k, 0.0) + float(v)

    return results


def get_metric_averages(metric_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    回傳目前為止所有已計算樣本的 metric 平均值。
    通常在整個資料集跑完後呼叫一次。
    """
    if _METRIC_COUNT == 0:
        return {}

    if metric_names is None:
        metric_names = SUPPORTED_METRICS

    avgs: Dict[str, float] = {}
    for name in metric_names:
        if name in _METRIC_SUMS:
            avgs[name] = _METRIC_SUMS[name] / _METRIC_COUNT

    return avgs


# ------------------------------------------------------------------
# CSV logging with resume support
# ------------------------------------------------------------------

def append_metrics_to_csv(
    gt_img_path,
    pred_img_path,
    csv_path,
    predictor,
    sample_id: Optional[str] = None,
    metric_names: Optional[List[str]] = None,
    skip_if_exists: bool = True,
) -> Dict[str, float]:
    """
    計算 geometry metrics 並將結果 append 到 CSV。

    - gt_img_path, pred_img_path: RGB 圖片路徑
    - predictor: StableNormal model
    - csv_path: CSV 檔路徑
    - sample_id: 該 sample 的唯一 ID（建議用檔名不含副檔名）
    - skip_if_exists:
        - True: 若該 csv 檔裡已經有同樣 sample_id，就直接跳過，不再計算/寫入。
        - False: 一樣會再算一次並寫入（小心重複列）。
    """
    global _CSV_ID_CACHE

    if metric_names is None:
        metric_names = SUPPORTED_METRICS

    # 1) 準備 ID cache：讀一次 CSV，把已經有的 id 記起來
    if csv_path not in _CSV_ID_CACHE:
        existing_ids = set()
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "id" in row and row["id"]:
                        existing_ids.add(row["id"])
        _CSV_ID_CACHE[csv_path] = existing_ids
    else:
        existing_ids = _CSV_ID_CACHE[csv_path]

    # 如果要 skip 且 id 已存在，直接返回空 dict
    if skip_if_exists and sample_id is not None and sample_id in existing_ids:
        return {}

    # 2) 計算 metrics
    metrics = compute_all_metrics(
        gt_img_path, pred_img_path, predictor, metric_names=metric_names
    )

    # 3) 準備寫入資料
    row: Dict[str, float] = {}
    row["id"] = sample_id if sample_id is not None else ""
    for name in metric_names:
        row[name] = float(metrics[name])

    # 4) 判斷要不要寫 header
    file_exists = os.path.exists(csv_path)
    write_header = not file_exists

    # 5) 寫入 CSV（append 模式）
    fieldnames = ["id"] + metric_names
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # 更新 cache
    if sample_id is not None:
        existing_ids.add(sample_id)

    # 回傳結果（方便在外面印或 debug）
    out: Dict[str, float] = {"id": row["id"]}
    out.update(metrics)
    return out


# ------------------------------------------------------------------
# CLI: evaluate folders of images
# ------------------------------------------------------------------

def main():
    """
    命令列用法示例：

    python metrics.py \\
        --gt_dir /path/to/gt_images \\
        --pred_dir /path/to/relight_images \\
        --csv_path /path/to/metrics_normal_ex5.csv \\
        --metrics normal_mae_deg normal_med_deg normal_lt_11_25 normal_lt_30 \\
        --resume

    - 會列出 gt_dir 底下所有 RGB 影像（依副檔名過濾）
    - 假設 pred_dir 裡面有對應「同檔名」的 relight 圖
    - 每一組 (gt, pred) 會先用 StableNormal 預測 normal，再算幾何 metrics，
      最後 append 到 csv_path
    - 若加上 --resume，已經存在於 CSV 的 id 會被跳過（方便斷點續跑）
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/mnt/HDD7/miayan/paper/scriblit/metric_test/gt",
        help="GT RGB 圖片資料夾路徑",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="/mnt/HDD7/miayan/paper/scriblit/metric_test/pred/ex2_2",
        help="Relight RGB 圖片資料夾路徑（檔名需與 gt 對應）",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="metrics_normal_geom_ex2_2.csv",
        help="輸出 CSV 檔案路徑",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=SUPPORTED_METRICS,
        help=f"要計算的指標名稱列表（預設: {SUPPORTED_METRICS}）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若加上此 flag，遇到 CSV 中已存在的 sample_id 會自動跳過（斷點續跑用）",
    )
    parser.add_argument(
        "--exts",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
        help="要掃描的影像副檔名",
    )

    args = parser.parse_args()

    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    csv_path = args.csv_path
    metric_names = args.metrics
    skip_if_exists = args.resume
    exts = [e.lower() for e in args.exts]

    # 1) 初始化 StableNormal（只做一次）
    print("Loading StableNormal predictor from torch.hub ...")
    predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
    predictor = predictor.to(DEVICE)
    print("Predictor loaded.")

    # 2) 掃描 GT 資料夾
    gt_files = [
        f for f in sorted(os.listdir(gt_dir))
        if any(f.lower().endswith(ext) for ext in exts)
    ]
    if not gt_files:
        print(f"[WARN] No images found in gt_dir={gt_dir} with exts={exts}")
        return

    print(f"Found {len(gt_files)} GT images.")
    print(f"Metrics: {metric_names}")
    print(f"Output CSV: {csv_path}")
    print(f"Resume mode: {'ON' if skip_if_exists else 'OFF'}")

    # 3) 一張一張算
    for idx, fname in enumerate(gt_files):
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        if not os.path.exists(pred_path):
            print(f"[SKIP] missing relight image for {fname}")
            continue

        sample_id = os.path.splitext(fname)[0]

        result = append_metrics_to_csv(
            gt_img_path=gt_path,
            pred_img_path=pred_path,
            csv_path=csv_path,
            predictor=predictor,
            sample_id=sample_id,
            metric_names=metric_names,
            skip_if_exists=skip_if_exists,
        )

        if result:
            print(f"[{idx+1}/{len(gt_files)}] {sample_id}: {result}")
        else:
            print(f"[{idx+1}/{len(gt_files)}] {sample_id}: skipped (already in CSV)")

    # 4) 全部跑完後印出平均
    if _METRIC_COUNT > 0:
        final_avgs = get_metric_averages(metric_names=metric_names)
        avg_str = ", ".join(f"{k}={v:.4f}" for k, v in final_avgs.items())
        print(f"\n[FINAL AVERAGE over {_METRIC_COUNT} samples] {avg_str}")
    else:
        print("\n[INFO] No samples were evaluated (maybe all skipped in resume mode).")


if __name__ == "__main__":
    main()
