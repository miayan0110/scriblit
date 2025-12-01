# import some helper functions from chrislib (will be installed by the intrinsic repo)
from chrislib.general import show, view, invert
from chrislib.data_util import load_image

# import model loading and running the pipeline
from intrinsic.pipeline import load_models, run_pipeline

import os, io, argparse
from datasets import load_from_disk, Dataset, Image, load_dataset
from PIL import Image as PILImage
import numpy as np


def pil_from_any(obj):
    if isinstance(obj, PILImage.Image):
        return obj
    if isinstance(obj, dict) and "bytes" in obj:
        return PILImage.open(io.BytesIO(obj["bytes"]))
    if isinstance(obj, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(obj))
    raise TypeError(f"Unsupported image type: {type(obj)}")

def to_rgb_numpy(img):
    """
    將輸入統一成 np.uint8 的 HxWx3 陣列。
    支援 PIL.Image 與 numpy.ndarray。
    """
    if isinstance(img, PILImage.Image):
        img = img.convert("RGB")
        arr = np.array(img)  # HxWx3, uint8
    elif isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:  # 灰階 -> 疊三通道
            arr = np.stack([arr]*3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> 丟 alpha
            arr = arr[:, :, :3]
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unsupported ndarray shape: {arr.shape}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    return arr

def to_pil_rgb(arr):
    """把輸入統一轉成 PIL RGB。支援 numpy/PIL；自動處理灰階、RGBA、float(0~1/0~255)。"""
    if isinstance(arr, PILImage.Image):
        return arr.convert("RGB")

    # 不是 PIL，就當成 numpy
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    # 灰階 -> 疊 3 通道
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    # RGBA -> 丟 alpha
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    # 浮點 → uint8
    if np.issubdtype(arr.dtype, np.floating):
        # 如果最大值 <= 1 視為 0~1，放大到 0~255
        if arr.max() <= 1.0:
            arr = (arr * 255.0).round()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    return PILImage.fromarray(arr, mode="RGB")

def to_jpeg_256(pil_img, quality=90):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((256, 256), resample=PILImage.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return {"bytes": buf.getvalue(), "path": None}

def encode_all(ex, quality=90):
    out = {}
    out["image"]      = to_jpeg_256(pil_from_any(ex["image"]), quality)
    out["mask"]       = to_jpeg_256(pil_from_any(ex["mask"]), quality)
    out["normal"]     = to_jpeg_256(pil_from_any(ex["normal"]), quality)
    out["depth"]      = to_jpeg_256(pil_from_any(ex["depth"]), quality)
    out["pseudo_gt"]  = to_jpeg_256(pil_from_any(ex["pseudo_gt"]), quality)
    out["lightmap"]   = to_jpeg_256(pil_from_any(ex["lightmap"]), quality)
    out["albedo"]     = to_jpeg_256(pil_from_any(ex["albedo"]), quality)
    return out

def gen_alb(ex, intrinsic_model, device):
    out = {}
    pil_img = pil_from_any(ex["image"])
    np_rgb = to_rgb_numpy(pil_img)

    # 關鍵「小改」：轉成 float32 並縮放到 0~1，避免下游變成 float64
    np_rgb = np_rgb.astype(np.float32) / 255.0
    print(np_rgb.shape)

    result = run_pipeline(
        intrinsic_model,
        np_rgb,   # 傳 numpy float32，torch.from_numpy 會維持 float32
        device=device
    )
    # chrislib.view 可能回傳 numpy / PIL，保險起見轉回 PIL
    alb = to_pil_rgb(view(result["hr_alb"]))

    out['albedo'] = alb
    return out


def main(args):
    device = args.device
    intrinsic_model = load_models('v2', device=device)

    ds = load_from_disk(args.src_image)
    print(ds)
    for col in ds.column_names:
        if col not in ('color', 'intensity'):
            ds = ds.cast_column(col, Image(decode=True))

    gen_alb(ds[0], intrinsic_model=intrinsic_model, device=device)

    # # generate albedo
    # ds = ds.map(
    #     lambda ex: gen_alb(ex, intrinsic_model=intrinsic_model, device=device),
    #     num_proc=args.num_proc,
    #     batched=False,
    #     writer_batch_size=1000,
    #     desc="Generating albedo..."
    # )

    # # 全部 resize + jpeg
    # print("[INFO] resizing to 256x256 and encoding as JPEG...")
    # ds = ds.map(
    #     lambda ex: encode_all(ex, quality=args.quality),
    #     num_proc=args.num_proc,
    #     batched=False,
    #     writer_batch_size=1000,
    #     desc="Resize + JPEG encode"
    # )
    
    # os.makedirs(args.out, exist_ok=True)
    # ds.save_to_disk(args.out)
    # print(f"[DONE] saved to {args.out}")
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_image", default='/mnt/HDD7/miayan/paper/relighting_datasets/lsun_train_gt', help="原dataset路徑")
    ap.add_argument("--out", default='/mnt/HDD7/miayan/paper/relighting_datasets/lsun_train', help="輸出 dataset 路徑")
    ap.add_argument("--num_proc", type=int, default=1)
    ap.add_argument("--device", default='cuda:4', help="gpu")
    ap.add_argument("--quality", type=int, default=90)  # 補上，給 encode_all 用
    args = ap.parse_args()

    main(args)

    # ds = load_from_disk(args.out)
    # print(ds)
    # for col in ds.column_names:
    #     if col not in ('color', 'intensity'):
    #         ds = ds.cast_column(col, Image(decode=True))
    
    # ds[0]['image'].save('image.jpg')
    # ds[0]['pseudo_gt'].save('pseudo_gt.jpg')
    # ds[0]['albedo'].save('albedo.jpg')
