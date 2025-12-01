import os
import sys
import glob
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from datasets import load_dataset, load_from_disk, Image as HfImage, Dataset
from transformers import pipeline

import sys
sys.path.append('../../')
import utils.utils as utils
import projects.dsine.config as config
from utils.projection import intrins_from_fov, intrins_from_txt

import json
from tqdm import tqdm

# gpu id | real id
#    0   |    2
#    1   |    4
#    2   |    6
#    3   |    0
#    4   |    1
#    5   |    3
#    6   |    5
#    7   |    7



def depth_estimation(example, pipe):
    img = example['image']
    depth = pipe(img)["depth"]

    return {"depth": depth}

def normal_estimation(example, model):
    img = example['image'].convert("RGB")
    
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
    pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

    pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
    pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)
    im = Image.fromarray(pred_norm[0,...])

    return {"normal": im}


if __name__ == '__main__':
    device = torch.device('cuda')
    args = config.get_args(test=True)
    assert os.path.exists(args.ckpt_path)

    if args.NNET_architecture == 'v00':
        from models.dsine.v00 import DSINE_v00 as DSINE
    elif args.NNET_architecture == 'v01':
        from models.dsine.v01 import DSINE_v01 as DSINE
    elif args.NNET_architecture == 'v02':
        from models.dsine.v02 import DSINE_v02 as DSINE
    elif args.NNET_architecture == 'v02_kappa':
        from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise Exception('invalid arch')

    model = DSINE(args).to(device)
    model = utils.load_checkpoint(args.ckpt_path, model)
    model.eval()

    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    ds: Dataset = load_from_disk('/mnt/HDD7/miayan/paper/relighting_datasets/lsun_train_jpg')
    # ds: Dataset = load_dataset('pcuenq/lsun-bedrooms', split='train', cache_dir='/mnt/HDD7/miayan/paper/relighting_datasets/lsun20')

    # img_root = '../../../gpt_relight/data/ori'
    # json_file = '../../../gpt_relight/data/mydata/prompt_v2.jsonl'
    # with open(json_file, "r", encoding="utf-8") as f:
    #     images = [json.loads(line) for line in f]

    # img_paths = glob.glob(f'{img_root}/*.png') + glob.glob(f'{img_root}/*.jpg')
    # img_paths.sort()
    os.makedirs('/mnt/HDD7/miayan/paper/relighting_datasets/lsun_train_normal_depth', exist_ok=True)

    # pbar = tqdm(enumerate(images), total=len(images), desc="Processing")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # ds = ds.select(range(min(2, len(ds))))

    ds = ds.map(lambda ex: depth_estimation(ex, pipe), desc="Adding depth")
    ds = ds.map(lambda ex: normal_estimation(ex, model), desc="Adding normal")

    ds = ds.cast_column("image", HfImage())
    ds = ds.cast_column("normal", HfImage())
    ds = ds.cast_column("depth", HfImage())
    ds.save_to_disk('/mnt/HDD7/miayan/paper/relighting_datasets/lsun_train_normal_depth')

    # test
    ds_new = load_from_disk('/mnt/HDD7/miayan/paper/relighting_datasets/lsun_train_normal_depth')
    image = ds_new[0]["image"]
    normal = ds_new[0]["normal"]
    depth = ds_new[0]["depth"]
    image.save('/mnt/HDD7/miayan/paper/relighting_datasets/test_image.png')
    normal.save('/mnt/HDD7/miayan/paper/relighting_datasets/test_normal.png')
    depth.save('/mnt/HDD7/miayan/paper/relighting_datasets/test_depth.png')

    # with torch.no_grad():
        # for img_path in img_paths:
        # for i, img_attr in pbar:
            # img_path = f"{img_root}/{img_attr['target']}"
            # target_path = f"{img_root}/{img_attr['normal']}"
            # print(img_path, target_path)
            # img_name = os.path.splitext(img_path)[0]
            # ext = os.path.splitext(img_path)[1]
            # target_path = f"{img_name.replace('ori', 'normal')}_normal.png"
            # img = Image.open(img_path).convert('RGB')
            # img = np.array(img).astype(np.float32) / 255.0
            # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

            # # pad input
            # _, _, orig_H, orig_W = img.shape
            # lrtb = utils.get_padding(orig_H, orig_W)
            # img = F.pad(img, lrtb, mode="constant", value=0.0)
            # img = normalize(img)

            # # get intrinsics
            # intrins_path = img_path.replace(ext, '.txt')
            # if os.path.exists(intrins_path):
            #     # NOTE: camera intrinsics should be given as a txt file
            #     # it should contain the values of fx, fy, cx, cy
            #     intrins = intrins_from_txt(intrins_path, device=device).unsqueeze(0)
            # else:
            #     # NOTE: if intrins is not given, we just assume that the principal point is at the center
            #     # and that the field-of-view is 60 degrees (feel free to modify this assumption)
            #     intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)
            # intrins[:, 0, 2] += lrtb[0]
            # intrins[:, 1, 2] += lrtb[2]

            # pred_norm = model(img, intrins=intrins)[-1]
            # pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

            # # save to output folder
            # # NOTE: by saving the prediction as uint8 png format, you lose a lot of precision
            # # if you want to use the predicted normals for downstream tasks, we recommend saving them as float32 NPY files
            # # target_path = img_path.replace('/img/', '/output/').replace(ext, '.png')

            # pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
            # pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)
            # im = Image.fromarray(pred_norm[0,...])
            # im.save(target_path)
