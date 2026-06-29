import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple
import nibabel as nib

from datasets.util import FILES

SINGER_TASK = {
    "Demo": ["SC_age",
            "BL_Z_TMTPartBSec",  # Processing Speed / Executive Function
            ], 
    "Cth": ["mean_thickness"],
    "WMH": ["WMH_V1"],
}

HCP_TASK = {
    "HCPA_phe_tab_age_sex": ["age","sex", "Flanker", "CardSort"],
}

# format Excel IDs
SUBJECT_ID = "Subject ID"
SUBJECT_ID_ALIASES = {"Subject_ID": SUBJECT_ID}
SUBJECT_ID_FORMATTERS = {
    ("SINGER", "Cth"): lambda s: s.astype(str).str.removesuffix("_fs"),
    ("HCP", "*"): lambda s: s.astype(str).str.strip() + "_V1_MR",
}

def find_sheet_for_task(dataset_name: str, task: str) -> str:
    if dataset_name == "SINGER":
        task_map = SINGER_TASK
    elif dataset_name == "HCP":
        task_map = HCP_TASK    
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    for sheet, tasks in task_map.items():
        if task in tasks:
            return sheet
    
    raise ValueError(
        f"Task '{task}' not found in {dataset_name}_TASK. "
        f"Available tasks: {task_map}"
    )

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


class NODDIRawDataset_External(Dataset):
    """Fine-tuning dataset for PICMAN, SINGER, HCP, and similar cohorts."""

    def __init__(
        self,
        data_root: str,
        excel_path: str,
        split_dir: str,
        splits: int = 0,
        state: str = 'train',
        dataset_name: str = 'PICMAN',
        sheet_name: str = 'Demo',
        target_col: str = 'Calculated Age',
        task_type: str = 'regression',
        label_normalize: bool = True,
        stats_dir: str = './label_stats',
        target_size: Tuple[int, int, int] = (96, 112, 96),
        clip_percentile: Tuple[float, float] = (1.0, 99.0),
        zscore: bool = True,
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.state = state
        self.target_size = target_size
        self.task_type = task_type
        self.label_normalize = label_normalize
        self.stats_dir = stats_dir
        self.clip_percentile = clip_percentile
        self.zscore = zscore
        
        split_file = os.path.join(split_dir, f'split_seed_{splits}.json')

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            subject_splits = json.load(f)
            
        if state not in subject_splits:
            raise ValueError(f"State '{state}' not found in split file. Available: {list(subject_splits.keys())}")
            
        target_ids = {str(sid) for sid in subject_splits[state]}
        print(f"[{state.upper()}] Loaded {len(target_ids)} subjects from split file (Seed {splits}).")

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        # load label file for current cohort
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        # unify ID column name (e.g. Subject_ID -> Subject ID)
        for alias, col in SUBJECT_ID_ALIASES.items():
            if col not in df.columns and alias in df.columns:
                df = df.rename(columns={alias: col})
        if SUBJECT_ID not in df.columns:
            raise ValueError(f"Missing '{SUBJECT_ID}' column. Available: {list(df.columns)}")

        # format Excel IDs to match image file names
        fmt = SUBJECT_ID_FORMATTERS.get((dataset_name, sheet_name)) or SUBJECT_ID_FORMATTERS.get((dataset_name, "*"))
        df[SUBJECT_ID] = fmt(df[SUBJECT_ID]) if fmt else df[SUBJECT_ID].astype(str)

        # keep only subjects assigned to this train/val/test split
        df = df[df[SUBJECT_ID].isin(target_ids)]

        # drop rows with missing target labels
        df[target_col] = df[target_col].replace(["NA", "nan", "", " "], np.nan)
        df = df.dropna(subset=[target_col])

        self.valid_ids = df[SUBJECT_ID].tolist()
        self.labels = dict(zip(self.valid_ids, df[target_col]))
        print(f"[{state.upper()}] After filtering missing labels: {len(self.valid_ids)} subjects remain.")
        
        # keep subjects with complete O/N/F NIfTI files
        self.subjects = []
        for sid in self.valid_ids:
            subject_dir = self.data_root / sid
            O_path = subject_dir / FILES["O"]
            N_path = subject_dir / FILES["N"]
            F_path = subject_dir / FILES["F"]
            
            files_exist = O_path.exists() and N_path.exists() and F_path.exists()
            if files_exist:
                self.subjects.append({
                    "id": sid,
                    "label": self.labels[sid]
                })
            else:
                print(f"[WARNING] Missing files for subject {sid}, skipping...")
        
        if len(self.subjects) == 0:
            raise RuntimeError(f"No valid subjects found with complete image files in {data_root}")
    
        print(f"[{state.upper()}] After checking image files: {len(self.subjects)} subjects with complete data.")

        # compute or load label mean/std for regression
        if self.task_type == 'regression' and self.label_normalize:
            os.makedirs(self.stats_dir, exist_ok=True)
            self.stat_file = os.path.join(self.stats_dir, f'{dataset_name}_{target_col.replace(" ", "_")}_seed{splits}_stats.npy')
            self._handle_label_normalization()


    def _handle_label_normalization(self):
        """Compute or load label mean/std for regression normalization."""
        if self.state == 'train':
            all_labels = np.array([s['label'] for s in self.subjects], dtype=np.float32)
            mean = float(all_labels.mean())
            std = float(all_labels.std())
            
            print(f"Computing label stats on TRAIN: Mean={mean:.4f}, Std={std:.4f}")
            np.save(self.stat_file, np.array([mean, std]))
            self.label_mean, self.label_std = mean, std
            
        else:
            if not os.path.exists(self.stat_file):
                raise RuntimeError(f"Label stats file not found: {self.stat_file}. Please run 'train' state first to generate stats.")
            
            stats = np.load(self.stat_file)
            self.label_mean, self.label_std = float(stats[0]), float(stats[1])
            print(f"Loaded label stats from file: Mean={self.label_mean:.4f}, Std={self.label_std:.4f}")

    def __len__(self):
        return len(self.subjects)

    def _load_image(self, subject_id):
        """Load OD, ICVF, and ISOVF NIfTI volumes and return [3, H, W, D]."""
        subject_dir = self.data_root / subject_id
        
        O_path = subject_dir / FILES["O"]
        N_path = subject_dir / FILES["N"]
        F_path = subject_dir / FILES["F"]
        
        if not O_path.exists():
            raise FileNotFoundError(f"File not found: {O_path}")
        if not N_path.exists():
            raise FileNotFoundError(f"File not found: {N_path}")
        if not F_path.exists():
            raise FileNotFoundError(f"File not found: {F_path}")
        
        O = load_nii(str(O_path))
        N = load_nii(str(N_path))
        F = load_nii(str(F_path))
        
        O = self._norm_volume(O)
        N = self._norm_volume(N)
        F = self._norm_volume(F)
        
        # pad each modality to target_size
        H, W, D = O.shape
        pH, pW, pD = self.target_size
        
        O_padded = np.zeros(self.target_size, dtype=O.dtype)
        N_padded = np.zeros(self.target_size, dtype=N.dtype)
        F_padded = np.zeros(self.target_size, dtype=F.dtype)
        
        O_padded[:min(pH, H), :min(pW, W), :min(pD, D)] = O[:min(pH, H), :min(pW, W), :min(pD, D)]
        N_padded[:min(pH, H), :min(pW, W), :min(pD, D)] = N[:min(pH, H), :min(pW, W), :min(pD, D)]
        F_padded[:min(pH, H), :min(pW, W), :min(pD, D)] = F[:min(pH, H), :min(pW, W), :min(pD, D)]
        
        O_tensor = O_padded[None, ...]
        N_tensor = N_padded[None, ...]
        F_tensor = F_padded[None, ...]
        
        img_data = np.concatenate([O_tensor, N_tensor, F_tensor], axis=0)
        
        return img_data

    def _norm_volume(self, vol: np.ndarray) -> np.ndarray:
        """Apply percentile clipping and optional z-score to a single volume."""
        v = vol.astype(np.float32)
        
        if self.clip_percentile is not None:
            lo, hi = np.percentile(v, self.clip_percentile)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                v = np.clip(v, lo, hi)
        
        if self.zscore:
            mu = float(v.mean())
            sd = float(v.std() + 1e-6)
            v = (v - mu) / sd
        
        return v

    def __getitem__(self, idx):
        subject_info = self.subjects[idx]
        sid = subject_info['id']
        raw_label = subject_info['label']
        
        # load, normalize, pad O/N/F volumes
        img_data = self._load_image(sid)
        img_tensor = torch.from_numpy(img_data).float()
        
        O = img_tensor[0:1]
        N = img_tensor[1:2]
        F = img_tensor[2:3]

        meta = {"id": sid, "raw_label": raw_label}
        
        # encode label for regression or classification
        if self.task_type == 'regression':
            label_val = float(raw_label)
            if self.label_normalize:
                label_val = (label_val - self.label_mean) / self.label_std
            
            label_tensor = torch.tensor(label_val, dtype=torch.float32)
            
        elif self.task_type == 'classification':
            if isinstance(raw_label, str):
                if raw_label.lower() in ['male', 'm']:
                    label_tensor = torch.tensor(0, dtype=torch.long)
                elif raw_label.lower() in ['female', 'f']:
                    label_tensor = torch.tensor(1, dtype=torch.long)
                else:
                    label_tensor = torch.tensor(-1, dtype=torch.long) 
            else:
                label_tensor = torch.tensor(int(raw_label), dtype=torch.long)
                
        meta['label'] = label_tensor

        return O, N, F, meta


def make_internal_finetune_dataset(
    data_root: str = '/scratch/Projects/CFP-03/CFP03-CF-031/DTI_workspace',
    excel_path: str = '/scratch/Projects/CFP-03/CFP03-CF-031/DTI_workspace/DTI_meta/',
    split_dir: str = '/scratch/Projects/CFP-03/CFP03-CF-031/DTI_workspace/DTI_meta/',
    splits: int = 0,
    dataset_name: str = 'PICMAN',
    target_col: str = 'Calculated_Age',
    task_type: str = 'regression',
    stats_dir: str = '/scratch/Projects/CFP-03/CFP03-CF-031/DTI_workspace/DTI_meta/',
    target_size: Tuple[int, int, int] = (96, 112, 96),
    clip_percentile: Tuple[float, float] = (0.5, 99.5),
    zscore: bool = True,
):
    """Create train, val, and test datasets for external cohort fine-tuning."""

    # resolve cohort-specific paths and Excel sheet
    data_root = os.path.join(data_root, f'DTI_{dataset_name}')

    excel_path = os.path.join(excel_path, dataset_name, f'DTI_{dataset_name}_testing.xlsx')
    split_dir = os.path.join(split_dir, dataset_name)
    stats_dir = os.path.join(stats_dir, dataset_name, 'label_stats')
    sheet_name = find_sheet_for_task(dataset_name, target_col)
    print(f"[DATASET] Loading {dataset_name} dataset with sheet {sheet_name} and target column {target_col}.")

    label_normalize = True if task_type == 'regression' else False  

    # build train / val / test splits
    train_ds = NODDIRawDataset_External(
        data_root=data_root,
        excel_path=excel_path,
        split_dir=split_dir,
        splits=splits,
        state='train',
        dataset_name=dataset_name,
        sheet_name=sheet_name,
        target_col=target_col,
        task_type=task_type,
        label_normalize=label_normalize,
        stats_dir=stats_dir,
        target_size=target_size,
        clip_percentile=clip_percentile,
        zscore=zscore,
    )
    
    val_ds = NODDIRawDataset_External(
        data_root=data_root,
        excel_path=excel_path,
        split_dir=split_dir,
        splits=splits,
        state='val',
        dataset_name=dataset_name,
        sheet_name=sheet_name,
        target_col=target_col,
        task_type=task_type,
        label_normalize=label_normalize,
        stats_dir=stats_dir,
        target_size=target_size,
        clip_percentile=clip_percentile,
        zscore=zscore,
    )
    
    test_ds = NODDIRawDataset_External(
        data_root=data_root,
        excel_path=excel_path,
        split_dir=split_dir,
        splits=splits,
        state='test',
        dataset_name=dataset_name,
        sheet_name=sheet_name,
        target_col=target_col,
        task_type=task_type,
        label_normalize=label_normalize,
        stats_dir=stats_dir,
        target_size=target_size,
        clip_percentile=clip_percentile,
        zscore=zscore,
    )
    
    return train_ds, val_ds, test_ds


# for sanity check only
if __name__ == "__main__":
    data_root = "./"
    dataset_name = "PICMAN"
    target_col = "Calculated_Age"
    task_type = "regression"
    splits = 0

    train_ds, val_ds, test_ds = make_internal_finetune_dataset(
        data_root=data_root,
        splits=splits,
        dataset_name=dataset_name,
        target_col=target_col,
        task_type=task_type,
    )

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        print(f"{name}: n={len(ds)}")
        if len(ds) == 0:
            continue
        O, N, F, meta = ds[0]
        print(f"  id={meta['id']}  shapes O/N/F={tuple(O.shape)}  label={meta.get('label')}")