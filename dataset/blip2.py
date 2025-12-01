import argparse
import os
from platform import processor
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from datasets import load_from_disk, Dataset, Image, load_dataset
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def gen_caption(ex, processor, model, device):
    out = {}
    inputs = processor(ex['image'].convert("RGB"), return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    out['prompt'] = generated_text
    return out

def main(args):
    device = args.device
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir="./blip2")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir="./blip2")
    model.to(device)
    
    ds = load_from_disk(args.src_image)
    print(ds)
    ds = ds.cast_column('image', Image(decode=True))
            
    # print(gen_caption(ds[0], processor=processor, model=model, device=device))

    # generate captions
    ds = ds.map(
        lambda ex: gen_caption(ex, processor=processor, model=model, device=device),
        num_proc=args.num_proc,
        batched=False,
        writer_batch_size=1000,
        desc="Generating captions..."
    )
    
    os.makedirs(args.out, exist_ok=True)
    ds.save_to_disk(args.out)
    print(f"[DONE] saved to {args.out}")
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_image", default='/mnt/HDD7/miayan/paper/relighting_datasets/lsun_train', help="原dataset路徑")
    ap.add_argument("--out", default='/mnt/HDD7/miayan/paper/relighting_datasets/lsun_train_cap', help="輸出 dataset 路徑")
    ap.add_argument("--num_proc", type=int, default=None)
    ap.add_argument("--device", default='cuda:3', help="gpu")
    ap.add_argument("--quality", type=int, default=90)  # 補上，給 encode_all 用
    args = ap.parse_args()

    main(args)

    ds = load_from_disk(args.out)
    print(ds)
    for col in ds.column_names:
        if col not in ('color', 'intensity', 'prompt'):
            ds = ds.cast_column(col, Image(decode=True))
    
    ds[0]['image'].save('image.jpg')
    print(ds[0]['prompt'])
