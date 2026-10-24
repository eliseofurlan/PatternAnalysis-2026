"""
dataset.py - data loading, splitting and augmentation for HipMRI.
    * the study has 38 patients but 211 volumes
    * The 3D model and the 2D baseline are built from the SAME downsampled
      volumes and the SAME split, so they are compared on identical data.
"""
import glob
import json
import os
import random
import re
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# dataset constants
DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open"
IMAGE_DIR = "semantic_MRs"
LABEL_DIR = "semantic_labels_only"
PATIENT_REGEX = r"^([A-Z]\d+)_Week" 
NUM_CLASSES = 6                 # background, body, bone, bladder, rectum, prostate
BACKGROUND_LABEL = 0
BLADDER_LABEL = 3
RECTUM_LABEL = 4
PROSTATE_LABEL = 5
TARGET_SHAPE = (128, 128, 64)   # downsampled 
LEFT_RIGHT_AXIS = 0            


def patient_id(path):
    """Parse the patient ID from a filename"""
    match = re.search(PATIENT_REGEX, os.path.basename(path))
    if match is None:
        raise ValueError(f"Could not parse patient ID from {path}")
    return match.group(1)


PAIR_REGEX = r"^([A-Z]\d+_Week\d+)" 


def pair_key(path):
    """Key identifying one scan"""
    match = re.search(PAIR_REGEX, os.path.basename(path))
    if match is None:
        raise ValueError(f"Could not parse patient/week key from {path}")
    return match.group(1)


def find_pairs(data_root=DATA_ROOT):
    """
    Return (image_path, label_path) 
    """
    images = sorted(glob.glob(os.path.join(data_root, IMAGE_DIR, "*.nii*")))
    labels = sorted(glob.glob(os.path.join(data_root, LABEL_DIR, "*.nii*")))
    if not images or not labels:
        raise RuntimeError(f"Found {len(images)} images and {len(labels)} labels")

    img_by_key = {pair_key(p): p for p in images}
    lab_by_key = {pair_key(p): p for p in labels}
    if len(img_by_key) != len(images) or len(lab_by_key) != len(labels):
        raise RuntimeError("Duplicate patient/week keys found in image or label folder")

    unmatched = sorted(set(img_by_key) ^ set(lab_by_key))
    if unmatched:
        raise RuntimeError(f"{len(unmatched)} scans lack an image or a label, "
                           f"e.g. {unmatched[:5]}")
    return [(img_by_key[k], lab_by_key[k]) for k in sorted(img_by_key)]


def get_splits(data_root=DATA_ROOT, seed=0, fractions=(0.7, 0.15, 0.15)):
    """
    patient-level train/val/test split.
    Returns "train","val","test".
    """
    by_patient = defaultdict(list)
    for pair in find_pairs(data_root):
        by_patient[patient_id(pair[0])].append(pair)

    pids = sorted(by_patient)
    random.Random(seed).shuffle(pids)
    n = len(pids)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    groups = {
        "train": pids[:n_train],
        "val": pids[n_train:n_train + n_val],
        "test": pids[n_train + n_val:],
    }

    seen = {}
    for split, plist in groups.items():
        for p in plist:
            if p in seen:
                raise RuntimeError(f"Patient {p} in both {seen[p]} and {split}")
            seen[p] = split

    splits = {s: [pair for p in plist for pair in by_patient[p]]
              for s, plist in groups.items()}
    print("patients (train/val/test):", [len(groups[s]) for s in groups],
          "| volumes:", [len(splits[s]) for s in splits])
    return splits


def save_splits(splits, path):
    out = {s: [os.path.basename(img) for img, _ in pairs] for s, pairs in splits.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load_volume(path, is_label, target_shape=TARGET_SHAPE):
    """
    Load a NIfTI volume, downsample to target_shape.
    """
    data = nib.load(path).get_fdata(caching="unchanged")
    if data.ndim == 4:                       # occasional extra dimension in HipMRI
        data = data[..., 0]
    tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))[None, None]

    if is_label:
        resized = F.interpolate(tensor, size=target_shape, mode="nearest")
        return resized[0, 0].round().to(torch.uint8)

    resized = F.interpolate(tensor, size=target_shape, mode="trilinear",
                            align_corners=False)[0, 0]
    return (resized - resized.mean()) / (resized.std() + 1e-8)


def augment_pair(image, mask, flip_axes=(LEFT_RIGHT_AXIS,), noise_std=0.05):
    for axis in flip_axes:
        if torch.rand(1).item() < 0.5:
            image = image.flip(axis + 1)     # +1 skips the channel dimension
            mask = mask.flip(axis)
    scale = 1.0 + 0.1 * (2 * torch.rand(1) - 1)
    shift = 0.1 * (2 * torch.rand(1) - 1)
    image = image * scale + shift + noise_std * torch.randn_like(image)
    return image, mask


class HipMRIVolumes(Dataset):
    """
    Whole downsampled volumes.
    """

    def __init__(self, pairs, target_shape=TARGET_SHAPE, augment=False,
                 flip_axes=(LEFT_RIGHT_AXIS,), max_volumes=None):
        if max_volumes is not None:
            pairs = pairs[:max_volumes]
        self.augment = augment
        self.flip_axes = flip_axes
        self.names = [os.path.basename(img) for img, _ in pairs]
        self.images = [load_volume(img, False, target_shape) for img, _ in pairs]
        self.labels = [load_volume(lab, True, target_shape) for _, lab in pairs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].unsqueeze(0)
        mask = self.labels[idx].long()
        if self.augment:
            image, mask = augment_pair(image, mask, self.flip_axes)
        return image, mask


class HipMRISlices(Dataset):
    """
    2D slices
    Sharing the volume store with the 3D dataset guarantees the baseline sees
    exactly the same patients
    """

    def __init__(self, volumes, augment=False, flip_axes=(LEFT_RIGHT_AXIS,)):
        self.volumes = volumes
        self.augment = augment
        self.flip_axes = flip_axes
        depth = volumes.images[0].shape[-1]
        self.index = [(v, z) for v in range(len(volumes)) for z in range(depth)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        v, z = self.index[idx]
        image = self.volumes.images[v][:, :, z].unsqueeze(0)
        mask = self.volumes.labels[v][:, :, z].long()
        if self.augment:
            image, mask = augment_pair(image, mask, self.flip_axes)
        return image, mask