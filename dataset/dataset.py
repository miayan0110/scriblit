import torch
from tqdm import tqdm
import os
import pyiqa
import random

# import some helper functions from chrislib (will be installed by the intrinsic repo)
from chrislib.general import show, view, invert
from chrislib.data_util import load_image

# import model loading and running the pipeline
from intrinsic.pipeline import load_models, run_pipeline

import glob
from PIL import Image
import json
from transformers import (
    Blip2Processor,
    Blip2Model,
    Blip2ForConditionalGeneration,
)


def append_to_jsonl(path, data_dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")

def save_image(image, path):
    image = (image * 255).clip(0, 255).astype('uint8')
    image = Image.fromarray(image)
    image.save(path)

def gen_prompt(processor, gen_model, img_path, device):
    question   = "Describe the scene in this image"
    image = Image.open(img_path).convert("RGB")

    gen_inputs = processor(
        images=image,
        text=question,
        return_tensors="pt",
        padding=True, truncation=True, max_length=32
    ).to(device)

    with torch.no_grad():
        gen_ids = gen_model.generate(**gen_inputs, max_new_tokens=32)[0]
    answer = processor.decode(gen_ids, skip_special_tokens=True)
    return answer


def main(img_root, save_root='', in_json_file='', o_json_file='', device='cuda:5'):
    # clipscore = pyiqa.create_metric('clipscore', device=device)
    intrinsic_model = load_models('v2', device=device)

    processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", cache_dir="./blip2"
    )
    
    gen_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", cache_dir="./blip2"
    ).to(device)
    gen_model.eval()

    save_num = 0
    with open(in_json_file, "r", encoding="utf-8") as f:
        images = [json.loads(line) for line in f]
    # img_list = sorted(glob.glob(f'{img_root}/*.jpg'))
    pbar = tqdm(enumerate(images), total=len(images), desc="Processing")
    for i, img_attr in pbar:
        img_path = f"{img_root}/{img_attr['filename']}"
        # filter_prompts = ['indoor scene', 'photo-realistic', 'good lighting', 'no people', 'no words']
        # score = clipscore(img_path, caption_list=filter_prompts).mean().item()

        # append_to_jsonl(o_json_file, {
        #     "filename": os.path.basename(img_path),
        #     "clip_score": score
        # })
        # pbar.set_postfix({"clip_score": score, "save file": os.path.basename(img_path)})

        img = load_image(img_path)
        result = run_pipeline(
            intrinsic_model,
            img,
            device=device
        )
        
        img = result['image']
        alb = view(result['hr_alb']) # gamma correct the estimated albedo
        # dif = 1 - invert(result['dif_shd']) # tonemap the diffuse shading
        # res = result['residual']
        gry_shd = result['gry_shd']

        prompt = gen_prompt(processor, gen_model, img_path, device)

        save_image(img, f'{save_root}/image/{save_num}.png')
        save_image(alb, f'{save_root}/albedo/{save_num}.png')
        # save_image(dif, f'{save_root}/normal/{save_num}.png')
        save_image(gry_shd, f'{save_root}/shading/{save_num}.png')
        pbar.set_postfix({"num": save_num, "save file": os.path.basename(img_path)})

        append_to_jsonl(o_json_file, {
            "normal": f'normal/{save_num}.png',
            "shading": f'shading/{save_num}.png',
            "albedo": f'albedo/{save_num}.png',
            "target": f'image/{save_num}.png',
            "prompt": prompt,
            "ori_img_name": os.path.basename(img_path),
        })
        save_num += 1

def change_filename(in_json_file='./mydata/prompt.jsonl', o_json_file='./mydata/prompt_v2.jsonl'):
    with open(in_json_file, "r", encoding="utf-8") as f:
        images = [json.loads(line) for line in f]

    pbar = tqdm(enumerate(images), total=len(images), desc="Processing")
    for i, img_attr in pbar:
        append_to_jsonl(o_json_file, {
            "normal": f'normal/{i}.png',
            "shading": f'shading/{i}.png',
            "albedo": f'albedo/{i}.png',
            "target": f'image/{i}.png',
            "prompt": img_attr['prompt'],
            "ori_img_name": img_attr['ori_img_name'],
        })
        pbar.set_postfix({"new": i, "old": img_attr['target'], "ori": img_attr['ori_img_name']})


# def gen_control_prompt():



if __name__ == '__main__':
    change_filename()
    # main('../../Project/datasets/lsun_bedroom/lsun_train', './mydata', './mydata/filtered.jsonl', './mydata/prompt.jsonl')


# gpu id | real id
#    0   |    2
#    1   |    4
#    2   |    6
#    3   |    0
#    4   |    1
#    5   |    3
#    6   |    5
#    7   |    7