# dataloader.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

import os
import time
import math
import pandas as pd

from src.models.masks import generate_block_mask

from monai.transforms import (
    Compose,
    RandSpatialCropd,
    RandAffined,
    RandFlipd,
    ToTensord,
    ToDeviced,
    EnsureTyped,
)
import warnings
from pandas.errors import DtypeWarning
warnings.filterwarnings("ignore", category=DtypeWarning)

from datasets.util import FILES


UKB_TASKS = {
    'age_when_attended_assessment_centre_f.21003.2.0': 'age',
    'sex_f.31.0.0': 'sex',
    'delta_avg_hipp_vol': 'hipp_delta_vol',
    'processing_speed_z': 'processing_speed_total',
    'has_29000_professional_diagnosis': 'mental_health_diag',
}

# Diagnosis task columns routed to the mental-health label file
_DIAGNOSIS_TASKS = {k for k, v in UKB_TASKS.items() if 'diag' in v}


def load_nii(path: str) -> np.ndarray:
    """Load a NIfTI volume as float32, replacing NaN/Inf with 0."""
    img = nib.load(path)
    vol = img.get_fdata(dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return vol

def load_npy(path: str) -> np.ndarray:
    """Load a NPY volume as float32, replacing NaN/Inf with 0."""
    vol = np.load(path)
    vol = vol.astype(np.float32, copy=False)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return vol


class NODDIRawDataset(Dataset):
    """
    UKB NODDI dataset for downstream fine-tuning.

    Returns (O, N, F, meta) with tensors shaped [1, D, H, W].
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = 'UKB',
        state: str = 'pretrain',
        meta_dir: str = '/scratch/Projects/CFP-03/CFP03-CF-031/DTI_workspace/DTI_meta/UKB/',
        splits: int = 0,
        label_file: str = '/scratch/Projects/CFP-03/CFP03-CF-031/DTI_workspace/DTI/master_UKB.csv',
        task: str = 'age_when_attended_assessment_centre_f.21003.2.0',
        task_type: str = 'regression',
        label_normalize: bool = True,
        zscore: bool = True,
        clip_percentile: Optional[Tuple[float, float]] = (1.0, 99.0),
        target_size: Tuple[int, int, int] = (96, 112, 96),
        batch_size: int = 300,
    ):
        # split / task config
        self.batch_size = batch_size
        self.epoch = 0

        self.state = state
        self.meta_dir = meta_dir
        self.target_size = target_size

        self.task = task
        self.task_type = task_type
        self.splits = splits
        self.label_file = label_file
        self.label_normalize = label_normalize

        task_name = UKB_TASKS[task]
        self.label_norm_file = os.path.join(meta_dir, 'label_stats', f'{dataset_name}_{task_name}_seed{splits}_stats.npy')
        self.label_stats = {}

        # load subject IDs for this split from pickle
        if state == 'pretrain':
            with open(os.path.join(meta_dir, 'pretrain_downstream_split.pkl'), 'rb') as f:
                self.split_ids = pickle.load(f)
            ids = self.split_ids[state]
        else:
            with open(os.path.join(meta_dir, 'splits_v1_only', f'downstream_split_0{splits}.pkl'), 'rb') as f:
                self.split_ids = pickle.load(f)
            ids = self.split_ids[state]
        
        # load subjects files
        root_path = Path(root)
        self.subjects = []
        for sid in ids:
            d = root_path / str(sid)
            paths = {k: str(d / v) for k, v in FILES.items()}
            if all(Path(pp).exists() for pp in paths.values()):
                self.subjects.append({"id": d.name, **paths})
        self.subjects.sort(key=lambda x: x["id"])
        if len(self.subjects) == 0:
            raise RuntimeError(
                f"No subjects found under {root} (each folder must contain {list(FILES.values())})"
            )
        else:
            print(f"Found {len(self.subjects)} subjects.")

        # load labels
        self.labels = {}
        if state != 'pretrain':
            df_all = pd.read_csv(label_file)
            df_task = df_all[['eid', task]].replace(["NA", "nan", ""], np.nan)
            df_task = df_task.dropna(subset=[task])

            subject_ids = [int(s["id"]) for s in self.subjects]
            df_task = df_task[df_task['eid'].isin(subject_ids)]
            self.labels = dict(zip(df_task['eid'].astype(str), df_task[task]))   

            missing = [s for s in self.subjects if str(s["id"]) not in self.labels]
            if len(missing) > 0:
                print(f"[Label] Missing/NA samples: {len(missing)} / {len(self.subjects)}")
            self.subjects = [s for s in self.subjects if str(s["id"]) in self.labels]

            if self.label_normalize:
                mean, std = self.get_label_norm_stats()
                self.label_stats = {'mean': mean, 'std': std}

        self.zscore = zscore
        self.clip_percentile = clip_percentile
    

    def get_label_norm_stats(self):
        """Compute mean/std on train; load cached stats for val and test."""
 
        if self.state == "train":
            if not os.path.exists(self.label_norm_file):
                print(f"Computing label norm stats and saving to {self.label_norm_file}")
                label_vals = np.asarray(list(self.labels.values()), dtype=np.float32)
                mean = float(label_vals.mean())
                std = float(label_vals.std())
                np.save(self.label_norm_file, np.array([mean, std], dtype=np.float32))
            else:
                print(f"Loading label norm stats from {self.label_norm_file}")
                mean, std = np.load(self.label_norm_file)
                mean, std = float(mean), float(std)

        if self.state == "val" or self.state == "test":
            if not os.path.exists(self.label_norm_file):
                raise FileNotFoundError(
                    f"Missing train label stats file: {self.label_norm_file}. "
                    "Generate it from the train split first."
                )
            print(f"Loading label norm stats from {self.label_norm_file}")
            mean, std = np.load(self.label_norm_file)

        return float(mean), float(std)

    def __len__(self):
        return len(self.subjects)
    
    def set_epoch(self, epoch):
        """Track epoch for reproducible shuffling in custom samplers."""
        self.epoch = epoch

    @staticmethod
    def _norm(vol: np.ndarray, clip_percentile, zscore: bool) -> np.ndarray:
        """Percentile clip followed by optional z-score normalization."""
        v = vol.astype(np.float32)
        if clip_percentile is not None:
            lo, hi = np.percentile(v, clip_percentile)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                v = np.clip(v, lo, hi)
        if zscore:
            mu = float(v.mean())
            sd = float(v.std() + 1e-6)
            v = (v - mu) / sd
        return v

    def __getitem__(self, idx: int):
        rec = self.subjects[idx]

        # load stacked NODDI_ALL.npy and split into O/N/F channels
        npy_file = load_npy(rec["O"].replace("ICVF", "ALL"))

        O = self._norm(npy_file[0], self.clip_percentile, self.zscore)
        N = self._norm(npy_file[1], self.clip_percentile, self.zscore)
        F = self._norm(npy_file[2], self.clip_percentile, self.zscore)

        # zero-pad to target_size and convert to [1, D, H, W] tensors        H, W, D = O.shape
        pH, pW, pD = self.target_size

        O_padded = np.zeros(self.target_size, dtype=O.dtype)
        N_padded = np.zeros(self.target_size, dtype=N.dtype)
        F_padded = np.zeros(self.target_size, dtype=F.dtype)
        O_padded[:min(pH, H), :min(pW, W), :min(pD, D)] = O[:min(pH, H), :min(pW, W), :min(pD, D)]
        N_padded[:min(pH, H), :min(pW, W), :min(pD, D)] = N[:min(pH, H), :min(pW, W), :min(pD, D)]
        F_padded[:min(pH, H), :min(pW, W), :min(pD, D)] = F[:min(pH, H), :min(pW, W), :min(pD, D)]

        O = torch.from_numpy(O_padded[None, ...]).to(torch.float32)
        N = torch.from_numpy(N_padded[None, ...]).to(torch.float32)
        F = torch.from_numpy(F_padded[None, ...]).to(torch.float32)
        meta = {"id": rec["id"]}

        # encode label for downstream tasks: regression or classification tasks
        id_str = rec["id"]
        label = self.labels[id_str]
        meta['raw_label'] = label

        if self.task_type == 'regression':
            label = torch.tensor(label, dtype=torch.float32)
            
            if self.label_normalize:
                mean = self.label_stats['mean']
                std = self.label_stats['std']
                label = ((label - mean) / std).to(torch.float32)
    
        elif self.task_type == 'classification':
            if label == 'Male':
                label = torch.tensor(0, dtype=torch.int64)
            elif label == 'Female':
                label = torch.tensor(1, dtype=torch.int64)
            elif isinstance(label, (bool, np.bool_)):
                label = torch.tensor(int(label), dtype=torch.int64)
            elif isinstance(label, str) and label.strip().lower() in ('true', 'false'):
                label = torch.tensor(1 if label.strip().lower() == 'true' else 0, dtype=torch.int64)
            else:
                raise Exception(f'Unsupported classification label value: {label!r}')
        else:
            raise Exception('error')
        meta['label'] = label

        return O, N, F, meta


def make_finetune_dataset(
    root: str = "./DTI_sample",
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    use_crop: bool = True,
    img_size=(91, 109, 91),
    view_size=(80, 96, 80),
    patch_size=(16, 16, 16),
    prob_affine=1.0,
    prob_flip=0.2,
    splits=0,
    task='age_when_attended_assessment_centre_f.21003.2.0',
    label_file = '/scratch/Projects/CFP-03/CFP03-CF-031/DTI_workspace/DTI_meta/UKB/master_UKB.csv',
    task_type='regression',
):
    """
    Create train, val, and test UKB datasets for downstream fine-tuning.
    """

    if task not in UKB_TASKS:
        raise ValueError(f"Unsupported UKB task '{task}'.")
    if not Path(label_file).is_file():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    label_normalize = True if task_type == 'regression' else False

    # build train / val / test splits with shared task config
    train_ds = NODDIRawDataset(root=root, target_size=img_size, batch_size=batch_size,
                                state='train', splits=splits, task=task, task_type=task_type,
                                label_file=label_file, label_normalize=label_normalize)

    val_ds = NODDIRawDataset(root=root, target_size=img_size, batch_size=batch_size, 
                                state='val', splits=splits, task=task, task_type=task_type,
                                label_file=label_file, label_normalize=label_normalize)
    
    test_ds = NODDIRawDataset(root=root, target_size=img_size, batch_size=batch_size, 
                            state='test', splits=splits, task=task, task_type=task_type,
                            label_file=label_file, label_normalize=label_normalize)
    
    return train_ds, val_ds, test_ds


# for sanity check only
if __name__ == "__main__":
    root = "./DTI"
    task = "age_when_attended_assessment_centre_f.21003.2.0"
    task_type = "regression"
    splits = 0

    train_ds, val_ds, test_ds = make_finetune_dataset(
        root=root, splits=splits, task=task, task_type=task_type,
    )

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        print(f"{name}: n={len(ds)}")
        if len(ds) == 0:
            continue
        O, N, F, meta = ds[0]
        print(f"  id={meta['id']}  shapes O/N/F={tuple(O.shape)}  label={meta.get('label')}")