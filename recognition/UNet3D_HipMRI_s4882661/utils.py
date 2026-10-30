"""
utils.py - metrics, inference, evaluation.

The evaluation is built around the open research question (does 3D context fix
boundary / slice-transition errors that a 2D baseline makes?), so besides
    * prostate Dice in the low / middle / high third of the ground-truth
    * leak_bladder / leak_rectum: fraction 
Both the 2D baseline and the 3D model are scored on whole volumes, so the
numbers are directly comparable.
"""
import random
import time
import warnings

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import BLADDER_LABEL, NUM_CLASSES, PROSTATE_LABEL, RECTUM_LABEL

POSITIONS = ("dice_low", "dice_mid", "dice_high")


# reproducibility 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# GPU profiling 
def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device):
    """Peak allocated GPU memory """
    return torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# metrics
def _nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _dice(p, g):
    denom = int(p.sum()) + int(g.sum())
    if denom == 0:
        return float("nan")
    return 2.0 * int((p & g).sum()) / denom


def per_class_scores(pred, gt, num_classes=NUM_CLASSES):
    """Per-class Dice"""
    dice = torch.full((num_classes,), float("nan"))
    iou = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        p, g = pred == c, gt == c
        union = int((p | g).sum())
        if union == 0:
            continue
        inter = int((p & g).sum())
        dice[c] = 2.0 * inter / (int(p.sum()) + int(g.sum()))
        iou[c] = inter / union
    return dice, iou


def position_scores(pred, gt, label=PROSTATE_LABEL):
    """
    Prostate Dice
    """
    out = {k: float("nan") for k in POSITIONS}
    out["outside_extent"] = float("nan")
    g, q = gt == label, pred == label
    if not g.any():
        return out
    slices_with_gt = torch.where(g.flatten(0, 1).any(0))[0]
    lo, hi = int(slices_with_gt.min()), int(slices_with_gt.max()) + 1
    span = hi - lo
    for k, key in enumerate(POSITIONS):
        a, b = lo + span * k // 3, lo + span * (k + 1) // 3
        if b > a:
            out[key] = _dice(q[:, :, a:b], g[:, :, a:b])
    outside = int(q[:, :, :lo].sum()) + int(q[:, :, hi:].sum())
    out["outside_extent"] = outside / max(int(q.sum()), 1)
    return out


def boundary_leakage(pred, gt):
    q = pred == PROSTATE_LABEL
    n = max(int(q.sum()), 1)
    return {
        "leak_bladder": int((q & (gt == BLADDER_LABEL)).sum()) / n,
        "leak_rectum": int((q & (gt == RECTUM_LABEL)).sum()) / n,
    }


# inference
@torch.no_grad()
def predict_3d(model, volume, device):
    """volume: (H, W, D) -> predicted label"""
    model.eval()
    logits = model(volume[None, None].to(device))
    return logits.argmax(1)[0].cpu()


@torch.no_grad()
def predict_2d(model, volume, device, batch_size=32):
    """Slice-wise prediction of a (H, W, D)."""
    model.eval()
    slices = volume.permute(2, 0, 1).unsqueeze(1)  # (D, 1, H, W)
    preds = [model(slices[i:i + batch_size].to(device)).argmax(1).cpu()
             for i in range(0, slices.shape[0], batch_size)]
    return torch.cat(preds).permute(1, 2, 0)


# evaluation
def evaluate_volumes(predict_fn, volumes, device):
    records = []
    for i in range(len(volumes)):
        volume, gt = volumes.images[i], volumes.labels[i].long()
        _sync(device)
        t0 = time.time()
        pred = predict_fn(volume)
        _sync(device)
        elapsed = time.time() - t0

        dice, iou = per_class_scores(pred, gt)
        record = {
            "name": volumes.names[i],
            "time_s": elapsed,
            "has_prostate": bool((gt == PROSTATE_LABEL).any()),
            "dice": dice.tolist(),
            "iou": iou.tolist(),
            "prostate_dice": float(dice[PROSTATE_LABEL]),
            "prostate_iou": float(iou[PROSTATE_LABEL]),
        }
        record.update(position_scores(pred, gt))
        record.update(boundary_leakage(pred, gt))
        records.append(record)
    return records


def summarise(records):
    summary = {"n_cases": len(records)}
    for key in ("prostate_dice", "prostate_iou", "outside_extent",
                "leak_bladder", "leak_rectum", "time_s", *POSITIONS):
        summary[key] = _nanmean([r[key] for r in records])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dice = np.nanmean(np.array([r["dice"] for r in records], dtype=float), axis=0)
        iou = np.nanmean(np.array([r["iou"] for r in records], dtype=float), axis=0)
    summary["dice_per_class"] = dice.tolist()
    summary["iou_per_class"] = iou.tolist()
    summary["foreground_dice"] = _nanmean(dice[1:])
    return summary


# plotting
def plot_history(history, path):
    """Save training curves: losses (left) and validation Dice (right)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(xlabel="epoch", ylabel="Dice + CE loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_prostate_dice"], label="prostate Dice")
    axes[1].plot(epochs, history["val_foreground_dice"], label="mean foreground Dice")
    axes[1].set(xlabel="epoch", ylabel="Dice", ylim=(0, 1), title="Validation Dice")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)