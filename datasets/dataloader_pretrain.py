# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import time

import numpy as np
import torch
from monai.data import PersistentDataset
from monai.transforms import (
    Compose,
    MapTransform,
    EnsureChannelFirstd,
    SpatialPadd,
    RandSpatialCropd,
    RandAffined,
    RandFlipd,
    EnsureTyped,
)

from src.models.masks import generate_block_mask

from datasets.util import FILES


def find_subjects(root: str) -> List[Dict[str, str]]:
    """
    Return subject file paths for O/N/F modalities.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    anchor = list(root.rglob(FILES["O"]))
    subs: List[Dict[str, str]] = []
    for p in anchor:
        d = p.parent
        paths = {k: str(d / v) for k, v in FILES.items()}
        if all(Path(pp).exists() for pp in paths.values()):
            subs.append({"id": d.name, **paths})
    subs.sort(key=lambda x: x["id"])
    return subs


class LoadNPYd(MapTransform):
    """
    Load .npy volumes and sanitize NaN/Inf values.
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            vol = np.load(d[key])
            vol = vol.astype(np.float32, copy=False)
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
            d[key] = vol
        return d


class NormalizeAndClipd(MapTransform):
    """
    Apply percentile clipping and optional z-score normalization.
    """

    def __init__(self, keys, clip_percentile=(1.0, 99.0), zscore=True):
        super().__init__(keys)
        self.clip_percentile = clip_percentile
        self.zscore = zscore

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            v = d[key]
            if self.clip_percentile is not None:
                lo, hi = np.percentile(v, self.clip_percentile)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    v = np.clip(v, lo, hi)
            if self.zscore:
                mu = float(v.mean())
                sd = float(v.std() + 1e-6)
                v = (v - mu) / sd
            d[key] = v
        return d


def build_geo_aug(
    view_size=(80, 96, 80),
    prob_affine=1.0,
    prob_flip=0.2,
    use_crop=True,
    mode="bilinear",
    device="cpu",
):
    """
    Build MONAI geometric augmentations that run on the target device.
    """
    transforms = [EnsureTyped(keys=["O", "N", "F"], device=device, track_meta=False)]

    if use_crop:
        transforms.append(
            RandSpatialCropd(keys=["O", "N", "F"], roi_size=view_size, random_size=False)
        )

    transforms.extend([
        RandAffined(
            keys=["O", "N", "F"],
            prob=prob_affine,
            rotate_range=(np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)),
            scale_range=(0.10, 0.10, 0.10),
            translate_range=(3, 3, 3),
            mode=(mode, mode, mode),
            padding_mode="border",
        ),
        RandFlipd(keys=["O", "N", "F"], prob=prob_flip, spatial_axis=0),
    ])

    return Compose(transforms)


class TwoViewGPUAugmentor:
    """
    Perform two-view augmentation on GPU.
    """

    def __init__(
        self,
        view_size=(80, 96, 80),
        img_size=(96, 112, 96),
        patch_size=(16, 16, 16),
        prob_affine=1.0,
        prob_flip=0.2,
        use_crop=True,
        device="cuda",
    ):
        self.device = device

        # two GPU augmentation pipelines (view A / view B)
        self.geo_a = build_geo_aug(view_size, prob_affine, prob_flip, use_crop, device=device)
        self.geo_b = build_geo_aug(view_size, prob_affine, prob_flip, use_crop, device=device)

        self.img_size = img_size
        self.view_size = view_size
        self.patch_size = patch_size

        # patch grid sizes for block-mask generation
        self.grid_3Dsize = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )
        self.block_3Dsize = (
            self.view_size[0] // self.patch_size[0],
            self.view_size[1] // self.patch_size[1],
            self.view_size[2] // self.patch_size[2],
        )

    def __call__(self, batch):
        
        Os_batch = batch["O"].to(self.device, non_blocking=True)
        Ns_batch = batch["N"].to(self.device, non_blocking=True)
        Fs_batch = batch["F"].to(self.device, non_blocking=True)

        Os = Os_batch.unbind(dim=0)
        Ns = Ns_batch.unbind(dim=0)
        Fs = Fs_batch.unbind(dim=0)

        # generate one seed per view for the whole batch
        seed_a = int(np.random.randint(0, 2**31 - 1))
        seed_b = int(np.random.randint(0, 2**31 - 1))

        Oa_list, Na_list, Fa_list = [], [], []
        Ob_list, Nb_list, Fb_list = [], [], []
        mask_a_list, indices_a_list = [], []
        mask_b_list, indices_b_list = [], []

        # augment on GPU and build block masks
        for O, N, F in zip(Os, Ns, Fs):
            data = {"O": O, "N": N, "F": F}

            # view A
            self.geo_a.set_random_state(seed=seed_a)
            va = self.geo_a(data)

            # view B
            self.geo_b.set_random_state(seed=seed_b)
            vb = self.geo_b(data)

            Oa_list.append(va["O"]); Na_list.append(va["N"]); Fa_list.append(va["F"])
            Ob_list.append(vb["O"]); Nb_list.append(vb["N"]); Fb_list.append(vb["F"])

            # generate mask (on CPU)
            mask_a, keep_idx_a, _ = generate_block_mask(
                grid_size=self.grid_3Dsize, view_size=self.block_3Dsize, seed=seed_a
            )
            mask_b, keep_idx_b, _ = generate_block_mask(
                grid_size=self.grid_3Dsize, view_size=self.block_3Dsize, seed=seed_b
            )
            mask_a_list.append(torch.as_tensor(mask_a, device=self.device))
            mask_b_list.append(torch.as_tensor(mask_b, device=self.device))
            indices_a_list.append(torch.as_tensor(keep_idx_a, device=self.device))
            indices_b_list.append(torch.as_tensor(keep_idx_b, device=self.device))

        # stack into batch tensors (on GPU)
        Oa = torch.stack(Oa_list, dim=0)
        Na = torch.stack(Na_list, dim=0)
        Fa = torch.stack(Fa_list, dim=0)
        Ob = torch.stack(Ob_list, dim=0)
        Nb = torch.stack(Nb_list, dim=0)
        Fb = torch.stack(Fb_list, dim=0)
        Ma = torch.stack(mask_a_list, dim=0)
        Mb = torch.stack(mask_b_list, dim=0)
        Ia = torch.stack(indices_a_list, dim=0)
        Ib = torch.stack(indices_b_list, dim=0)

        return Oa, Na, Fa, Ob, Nb, Fb, (Ma, Mb), (Ia, Ib), batch["id"]


def prepare_dataset(
    root: str,
    batch_size: int,
    num_workers: int = 4,
    split_file: str = '/scratch/Projects/CFP-03/CFP03-CF-031/DTI_workspace/DTI_meta/UKB/pretrain_downstream_split.pkl',
    state: str = 'pretrain',
    use_crop: bool = True,
    img_size=(96, 112, 96),
    view_size=(80, 96, 80),
    patch_size=(16, 16, 16),
    prob_affine=1.0,
    prob_flip=0.2,
    device="cuda",
    cache_rate=1.0,
    cache_dir: str = "./monai_cache",
):
    """
    Build a cached MONAI dataset for pretraining.
    """

    all_subjects = find_subjects(root)

    # filter subjects by split (e.g. pretrain / train / val / test)
    if split_file and Path(split_file).exists():
        with open(split_file, 'rb') as f:
            split_ids = pickle.load(f)
        if state in split_ids:
            ids = split_ids[state]
            subjects = [s for s in all_subjects if s.get('id') in ids]
        else:
            subjects = all_subjects
    else:
        subjects = all_subjects

    if len(subjects) == 0:
        raise RuntimeError(f"No subjects found in {root}")

    # cached deterministic transforms (load, normalize, pad) to disk
    pre_transforms = Compose([
        LoadNPYd(keys=["O", "N", "F"]),
        NormalizeAndClipd(keys=["O", "N", "F"], zscore=True),
        EnsureChannelFirstd(keys=["O", "N", "F"], channel_dim="no_channel"),
        SpatialPadd(keys=["O", "N", "F"], spatial_size=img_size),
    ])

    # initialize dataset
    print(f"Initializing PersistentDataset with {len(subjects)} subjects. This may take a while...")
    ds = PersistentDataset(
        data=subjects,
        transform=pre_transforms,
        cache_dir=cache_dir,
    )

    return ds
