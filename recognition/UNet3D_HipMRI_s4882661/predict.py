"""
predict.py - example usage of the trained models: inference, comparison, figures.
    python predict.py --weights-2d outputs/2d_best.pt --weights-3d outputs/3d_best.pt

    * prints a baseline-vs-3D comparison table (Dice/IoU, Dice per position
      along the slice axis, leakage, inference time),
    * saves failure-case figures for the worst test cases:
      input, ground truth, and false-positive / false-negative 
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import HipMRIVolumes, NUM_CLASSES, PROSTATE_LABEL, get_splits
from modules import UNet
from utils import (POSITIONS, evaluate_volumes, predict_2d, predict_3d, summarise)


def parse_args():
    p = argparse.ArgumentParser(description="Run trained HipMRI models on the test split")
    p.add_argument("--weights-2d", default=None)
    p.add_argument("--weights-3d", default=None)
    p.add_argument("--data-root", default=None, help="default: root stored in checkpoint")
    p.add_argument("--out-dir", default="predict_outputs")
    p.add_argument("--num-cases", type=int, default=3,
                   help="number of worst test cases to draw as failure figures")
    args = p.parse_args()
    if not (args.weights_2d or args.weights_3d):
        p.error("give at least one of --weights-2d / --weights-3d")
    return args


def load_model(path, device):
    """Rebuild a UNet from a checkpoint"""
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    model = UNet(dim=3 if a["model"] == "3d" else 2, num_classes=NUM_CLASSES,
                 base_channels=a["base_channels"], depth=a["depth"],
                 dropout=a.get("dropout", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, a


def error_overlay(image, pred, gt):
    img = (image - image.min()) / (image.max() - image.min() + 1e-8)
    rgb = np.stack([img] * 3, axis=-1)
    for mask, colour in ((pred & gt, (0, 1, 0)),       # true positive
                         (pred & ~gt, (1, 0, 0)),      # false positive
                         (~pred & gt, (0, 0, 1))):     # false negative
        rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.array(colour)
    return rgb


def plot_failure_case(name, volume, gt, preds, path):
    """
    Columns: slices near the low / middle / high end of the true prostate.
    Rows   : input, ground truth.
    """
    g = gt == PROSTATE_LABEL
    z = torch.where(g.flatten(0, 1).any(0))[0]
    lo, hi = int(z.min()), int(z.max())
    span = hi - lo
    slice_ids = [lo + span // 6, (lo + hi) // 2, hi - span // 6]
    titles = ["low-slice end", "middle", "high-slice end"]

    rows = 2 + len(preds)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    for col, (zi, title) in enumerate(zip(slice_ids, titles)):
        img = volume[:, :, zi].numpy()
        gt_mask = (gt[:, :, zi] == PROSTATE_LABEL).numpy()
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"{title} (slice {zi})")
        axes[1, col].imshow(error_overlay(img, gt_mask, gt_mask))
        for r, (model_name, pred) in enumerate(preds.items(), start=2):
            pred_mask = (pred[:, :, zi] == PROSTATE_LABEL).numpy()
            axes[r, col].imshow(error_overlay(img, pred_mask, gt_mask))
            dice = 2 * (pred_mask & gt_mask).sum() / max(pred_mask.sum() + gt_mask.sum(), 1)
            axes[r, col].set_title(f"{model_name} (Dice {dice:.2f})", fontsize=9)
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("ground truth")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name}\ngreen = correct, red = false positive, blue = false negative",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_position_chart(summaries, path):
    """Bar chart: mean prostate Dice"""
    labels = ["low-slice end", "middle", "high-slice end"]
    width = 0.8 / len(summaries)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (model_name, s) in enumerate(summaries.items()):
        vals = [s[k] for k in POSITIONS]
        ax.bar(np.arange(3) + i * width, vals, width, label=model_name)
    ax.set_xticks(np.arange(3) + width * (len(summaries) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set(ylabel="mean prostate Dice", ylim=(0, 1), title="Prostate Dice by position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # load models
    models, predictors, seeds, roots = {}, {}, set(), set()
    if args.weights_2d:
        m, a = load_model(args.weights_2d, device)
        models["2D U-Net (baseline)"] = m
        predictors["2D U-Net (baseline)"] = lambda v, m=m: predict_2d(m, v, device)
        seeds.add(a["seed"]); roots.add(a["data_root"])
    if args.weights_3d:
        m, a = load_model(args.weights_3d, device)
        models["3D U-Net"] = m
        predictors["3D U-Net"] = lambda v, m=m: predict_3d(m, v, device)
        seeds.add(a["seed"]); roots.add(a["data_root"])
    if len(seeds) != 1:
        raise RuntimeError("Checkpoints were trained with different seeds -> different splits")

    data_root = args.data_root or roots.pop()
    splits = get_splits(data_root, seed=seeds.pop())
    test_vols = HipMRIVolumes(splits["test"])
    print(f"test volumes: {len(test_vols)}")

    # evaluate 
    records, summaries = {}, {}
    for model_name, fn in predictors.items():
        records[model_name] = evaluate_volumes(fn, test_vols, device)
        summaries[model_name] = summarise(records[model_name])

    # comparison table 
    cols = [("prostate_dice", "Dice"), ("prostate_iou", "IoU"),
            ("dice_low", "Dice low"), ("dice_mid", "Dice mid"), ("dice_high", "Dice high"),
            ("outside_extent", "outside"), ("leak_bladder", "->bladder"),
            ("leak_rectum", "->rectum"), ("time_s", "s/vol")]
    print("\n" + f"{'model':24s}" + "".join(f"{c[1]:>11s}" for c in cols))
    for model_name, s in summaries.items():
        print(f"{model_name:24s}" + "".join(f"{s[k]:11.3f}" for k, _ in cols))
    print("\nper-class Dice (class index order):")
    for model_name, s in summaries.items():
        print(f"  {model_name}: {[round(d, 3) for d in s['dice_per_class']]}")

    #  per-case CSV 
    with open(os.path.join(args.out_dir, "per_case_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "case"] + [k for k, _ in cols])
        for model_name, recs in records.items():
            for r in recs:
                writer.writerow([model_name, r["name"]] + [r[k] for k, _ in cols])
    plot_position_chart(summaries, os.path.join(args.out_dir, "dice_by_position.png"))

    # failure-case figures
    reference = "3D U-Net" if "3D U-Net" in records else next(iter(records))
    scored = [(r["prostate_dice"], i) for i, r in enumerate(records[reference])
              if r["has_prostate"]]
    worst = [i for _, i in sorted(scored)[:args.num_cases]]
    print(f"\nworst {len(worst)} cases by {reference} prostate Dice:")
    for i in worst:
        name = test_vols.names[i]
        row = "  ".join(f"{m}: {records[m][i]['prostate_dice']:.3f}" for m in records)
        print(f"  {name}  ->  {row}")
        preds = {m: fn(test_vols.images[i]) for m, fn in predictors.items()}
        plot_failure_case(name, test_vols.images[i], test_vols.labels[i].long(), preds,
                          os.path.join(args.out_dir, f"failure_{i:03d}.png"))
    print(f"\nfigures and CSV saved to {args.out_dir}/")


if __name__ == "__main__":
    main()