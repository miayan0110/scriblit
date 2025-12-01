import io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dataset.intrinsic.pipeline import run_pipeline

from dep_nor.utils.projection import intrins_from_fov
import torchvision.transforms as transforms
import dep_nor.utils.utils as utils

## Utils
# 將模型(或模型字典)搬到指定 device
def _move_to_device(obj, device):
    if hasattr(obj, "to"):
        obj.to(device)
    elif isinstance(obj, dict):
        for v in obj.values():
            _move_to_device(v, device)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _move_to_device(v, device)

def alb_to_pil(tensor):
	# ---- 處理成 PIL ----
	# 去掉 batch 維度
	if tensor.ndim == 4:
		tensor = tensor[0]

	# 把值域從 [-1,1] 或 [0,1] 轉成 [0,255]
	if tensor.min() < 0:
		img = (tensor.clamp(-1, 1) + 1) / 2
	else:
		img = tensor.clamp(0, 1)

	# 轉成 CPU numpy
	img = img.detach().cpu().permute(1, 2, 0).numpy()  # [H,W,C]
	img = (img * 255).astype("uint8")

	# 轉成 PIL Image
	pil_img = Image.fromarray(img)
	return pil_img

def img_transforms(img, mode='input_cond'):
    if mode == 'input_cond':
        image_transforms = transforms.Compose([
			transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
			transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
		])
    elif mode == 'side_cond':
        image_transforms = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        
    if type(img) != torch.Tensor:
        img = transforms.ToTensor()(img)
        
    return image_transforms(img)



## Albedo Estimation
def pil_from_any(obj):
    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, dict) and "bytes" in obj:
        return Image.open(io.BytesIO(obj["bytes"]))
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj))
    raise TypeError(f"Unsupported image type: {type(obj)}")

def to_rgb_numpy(img):
    """
    將輸入統一成 np.uint8 的 HxWx3 陣列。
    支援 PIL.Image 與 numpy.ndarray。
    """
    if isinstance(img, Image.Image):
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
    if isinstance(arr, Image.Image):
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

    return Image.fromarray(arr, mode="RGB")

def gen_albedo(ex, intrinsic_model, device):
	pil_img = pil_from_any(ex)
	np_rgb = to_rgb_numpy(pil_img)

	# 關鍵「小改」：轉成 float32 並縮放到 0~1，避免下游變成 float64
	np_rgb = np_rgb.astype(np.float32) / 255.0

	_move_to_device(intrinsic_model, device)

	result = run_pipeline(
		intrinsic_model,
		np_rgb,   # 傳 numpy float32，torch.from_numpy 會維持 float32
		device=device
	)

	_move_to_device(intrinsic_model, 'cpu')
	torch.cuda.empty_cache()

	def view(img, p=100):
		img = img ** (1/2.2)
		return (img / np.percentile(img, p)).clip(0, 1)

	alb = view(result["hr_alb"])
	alb_tensor = img_transforms(torch.from_numpy(alb).permute(2, 0, 1).contiguous().unsqueeze(0))
	return alb_tensor


## DSINE: depth & normal estimation
def depth_estimation(img, pipe, device):
    pipe.model.to(device)
    pipe.device = torch.device(device) if not isinstance(device, torch.device) else device
    depth = pipe(img)["depth"]

    pipe.model.to('cpu')
    pipe.device = torch.device('cpu')
    torch.cuda.empty_cache()

    return depth

def normal_estimation(img, model, device):
    _move_to_device(model, device)
    img = img.convert("RGB")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, orig_H, orig_W = img.shape
    lrtb = utils.get_padding(orig_H, orig_W)
    img = F.pad(img, lrtb, mode="constant", value=0.0)
    img = normalize(img)

    intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)
    intrins[:, 0, 2] += lrtb[0]
    intrins[:, 1, 2] += lrtb[2]

    pred_norm = model(img, intrins=intrins)[-1]
    _move_to_device(model, 'cpu')
    torch.cuda.empty_cache()

    pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

    pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
    pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)
    im = Image.fromarray(pred_norm[0,...])

    return im

def normal_estimation_sn(img, predictor, device):
    _move_to_device(predictor, device)
    img = img.convert("RGB")
    normal = predictor(img)
    _move_to_device(predictor, 'cpu')
    return normal


## Lightmap Estimation
def _to_0_1(x: torch.Tensor) -> torch.Tensor:
    """
    將影像 tensor 大致轉到 [0,1]。
    - 若看起來是 [-1,1]，就 (x+1)/2
    - 若本來就在 [0,1] 左右，就不動
    """
    x_min = float(x.min())
    x_max = float(x.max())
    if x_min < -0.1 or x_max > 1.1:   # 粗略判斷是 [-1,1]
        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    return x


def calculate_pred_lightmap(
    I_relit: torch.Tensor,
    pseudo_gt: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        I_relit:   Bx3xHxW，VAE decode 出來的 relit 圖
        pseudo_gt: Bx3xHxW，pseudo GT（例如「燈關掉」或原圖）
    Returns:
        lightmap_pred: Bx1xHxW，灰階，值約在 [0,1]
        👉 只代表「這盞燈在該強度下讓哪裡變亮多少」，不含顏色
    """
    device = I_relit.device
    dtype = I_relit.dtype

    I_r = _to_0_1(I_relit)
    I_g = _to_0_1(pseudo_gt)

    # RGB -> luminance：Y = 0.299R + 0.587G + 0.114B
    w = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype).view(1, 3, 1, 1)
    L_r = (I_r * w).sum(dim=1, keepdim=True)  # Bx1xHxW
    L_g = (I_g * w).sum(dim=1, keepdim=True)  # Bx1xHxW

    # 只保留「變亮」的部分，變暗就當作別的光源被壓掉、先不管
    dL = torch.clamp(L_r - L_g, min=0.0)      # Bx1xHxW

    # 可選：clip 大 outlier，避免 loss 被幾個奇怪的點主導
    dL = dL.clamp(0.0, 1.0)

    return dL  # Bx1xHxW