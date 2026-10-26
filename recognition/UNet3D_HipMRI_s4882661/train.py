"""
train.py - train, validate, test and save a U-Net on HipMRI.

One script trains either model:
    python train.py --model 2d   
    python train.py --model 3d

Each epoch: train on the training split, compute validation loss, then score
whole validation
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import (DATA_ROOT, HipMRISlices, HipMRIVolumes, get_splits, save_splits)
from modules import DiceCELoss, UNet, count_parameters
from dataset import NUM_CLASSES
from utils import (evaluate_volumes, peak_memory_gb, plot_history, predict_2d,
                   predict_3d, reset_peak_memory, set_seed, summarise)


def parse_args():
    p = argparse.ArgumentParser(description="Train 2D baseline or 3D U-Net on HipMRI")
    p.add_argument("--model", choices=["2d", "3d"], required=True)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=None,
                   help="default: 16 slices (2d) or 2 volumes (3d)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-volumes", type=int, default=None,
                   help="cap volumes per split (quick debugging runs)")
    args = p.parse_args()
    if args.batch_size is None:
        args.batch_size = 16 if args.model == "2d" else 2
    return args


def mean_loss(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * images.size(0)
            count += images.size(0)
    return total / max(count, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device: {device} | model: {args.model}")

    # data
    splits = get_splits(args.data_root, seed=args.seed)
    save_splits(splits, os.path.join(args.out_dir, "splits.json"))
    train_vols = HipMRIVolumes(splits["train"], max_volumes=args.max_volumes)
    val_vols = HipMRIVolumes(splits["val"], max_volumes=args.max_volumes)

    if args.model == "3d":
        train_vols.augment = True
        train_set, val_set = train_vols, val_vols
        predict_fn = lambda vol: predict_3d(model, vol, device)
    else:
        train_set = HipMRISlices(train_vols, augment=True)
        val_set = HipMRISlices(val_vols, augment=False)
        predict_fn = lambda vol: predict_2d(model, vol, device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # model
    model = UNet(dim=3 if args.model == "3d" else 2, num_classes=NUM_CLASSES,
                 base_channels=args.base_channels, depth=args.depth,
                 dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"trainable parameters: {n_params:,}")

    criterion = DiceCELoss(NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # training loop 
    history = {"train_loss": [], "val_loss": [],
               "val_prostate_dice": [], "val_foreground_dice": []}
    best_dice, best_epoch, peak_train_vram = -1.0, 0, 0.0
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        reset_peak_memory(device)
        train_loss = mean_loss(model, train_loader, criterion, device,
                               optimizer, scaler, use_amp)
        peak_train_vram = max(peak_train_vram, peak_memory_gb(device))  # for training
        scheduler.step()

        val_loss = mean_loss(model, val_loader, criterion, device, amp=False)
        summary = summarise(evaluate_volumes(predict_fn, val_vols, device))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_prostate_dice"].append(summary["prostate_dice"])
        history["val_foreground_dice"].append(summary["foreground_dice"])
        plot_history(history, os.path.join(args.out_dir, f"{args.model}_curves.png"))

        if summary["prostate_dice"] > best_dice:
            best_dice, best_epoch = summary["prostate_dice"], epoch
            torch.save({"model_state": model.state_dict(), "args": vars(args),
                        "epoch": epoch}, ckpt_path)
        print(f"epoch {epoch:3d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| val prostate Dice {summary['prostate_dice']:.4f} | {time.time() - t0:.1f}s")

    train_time = time.time() - train_start
    with open(os.path.join(args.out_dir, f"{args.model}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # final test 
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    test_vols = HipMRIVolumes(splits["test"], max_volumes=args.max_volumes)
    reset_peak_memory(device)
    test_summary = summarise(evaluate_volumes(predict_fn, test_vols, device))
    peak_infer_vram = peak_memory_gb(device)

    results = {
        "model": args.model, "best_epoch": best_epoch, "best_val_prostate_dice": best_dice,
        "test": test_summary,
        "resources": {
            "trainable_parameters": n_params,
            "peak_train_vram_gb": peak_train_vram,
            "peak_inference_vram_gb": peak_infer_vram,
            "inference_time_per_volume_s": test_summary["time_s"],
            "total_train_time_s": train_time,
            "seconds_per_epoch": train_time / args.epochs,
        },
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, f"{args.model}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nbest epoch {best_epoch} (val prostate Dice {best_dice:.4f})")
    print(f"TEST prostate Dice {test_summary['prostate_dice']:.4f} | "
          f"IoU {test_summary['prostate_iou']:.4f} | "
          f"low/mid/high thirds {test_summary['dice_low']:.3f}/"
          f"{test_summary['dice_mid']:.3f}/{test_summary['dice_high']:.3f}")
    print(f"peak train VRAM {peak_train_vram:.2f} GB | params {n_params:,} | "
          f"inference {test_summary['time_s']:.3f}s/volume")


if __name__ == "__main__":
    main()