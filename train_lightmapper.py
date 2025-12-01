# train_lightmapper.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from lightmapper import Lightmapper
from dataloader import Lightmap_dataset


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, writer,
    pbar, global_step=0, log_interval=200):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        x = batch["control_source"].to(device)  # (B,9,H,W)
        y = batch["control_target"].to(device)  # (B,3,H,W)

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_so_far = running_loss / (batch_idx + 1)

        # step 級別 scalar
        writer.add_scalar("Loss/train_step", loss.item(), global_step)

        # 「單一」進度條更新
        pbar.update(1)
        pbar.set_postfix(ep=epoch, step=batch_idx + 1, loss=f"{loss.item():.4f}", avg=f"{avg_so_far:.4f}")

        # 每 log_interval steps 記一次 Pred/GT 圖
        if (global_step % log_interval) == 0:
            with torch.no_grad():
                pred_grid = make_grid(y_pred[:4], nrow=4, normalize=True, scale_each=True)
                gt_grid   = make_grid(y[:4],     nrow=4, normalize=True, scale_each=True)
                writer.add_image("Pred/Lightmap_step", pred_grid, global_step)
                writer.add_image("GT/Lightmap_step",   gt_grid,   global_step)

        global_step += 1

    avg_loss = running_loss / len(dataloader)
    return avg_loss, global_step


def load_checkpoint(model, optimizer, ckpt_path, device):
    print(f"Resuming from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch


def main():
    parser = argparse.ArgumentParser(description="Train Lightmapper")
    parser.add_argument("--ds_path", type=str, default="/mnt/HDD7/miayan/paper/relighting_datasets/lsun_train", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--exp_dir", type=str, default="/mnt/HDD7/miayan/paper/scriblit/lightmapper_ckpt",
                        help="Root directory (contains checkpoints + logs)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--save_interval", type=int, default=10, help="Save ckpt every N epochs")
    parser.add_argument("--log_interval", type=int, default=200, help="Log images every N steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 統一實驗資料夾
    ckpt_dir = os.path.join(args.exp_dir, "checkpoints")
    log_dir  = os.path.join(args.exp_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Dataset & DataLoader
    dataset = Lightmap_dataset(tokenizer=None, ds_path=args.ds_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model / Loss / Optimizer
    # 注意：如果你的 control_source 是 mask(3)+normal(3)+depth(3) → 9 通道
    model = Lightmapper(in_ch=9, out_ch=3, base_ch=64).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)

    writer = SummaryWriter(log_dir=log_dir)

    # 單一 tqdm 進度條（最大值 = 總步數）
    steps_per_epoch = len(dataloader)
    total_steps = args.epochs * steps_per_epoch
    initial_steps = start_epoch * steps_per_epoch
    pbar = tqdm(
        total=total_steps,
        initial=initial_steps,
        dynamic_ncols=True,
        leave=True,
        desc="Training",
    )

    # global step 對齊 resume
    global_step = initial_steps

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        avg_loss, global_step = train_one_epoch(model, dataloader, optimizer, criterion,
            device, epoch, writer, pbar=pbar, global_step=global_step, log_interval=args.log_interval)

        # epoch 平均 loss
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)

        # 週期性儲存 checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{epoch}.pt")
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                ckpt_path,
            )

    # final save
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{args.epochs}.pt")
    torch.save({"epoch": args.epochs, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
    writer.close()
    pbar.close()


if __name__ == "__main__":
    main()



"""
CUDA_VISIBLE_DEVICES=7 python train_lightmapper.py
"""