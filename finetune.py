# -*- coding: utf-8 -*-
import argparse
import builtins
import datetime
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.data import list_data_collate

from datasets.dataloader_external import make_internal_finetune_dataset
from datasets.dataloader_internal import UKB_TASKS, make_finetune_dataset
import src.utils.misc as misc
from src.models.vit_model import FullViTModel_Predict

# =============================================================================
# 1. DISTRIBUTED DATA PARALLEL (DDP) UTILITIES
# =============================================================================
def ddp_is_on() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def ddp_rank() -> int:
    return torch.distributed.get_rank() if ddp_is_on() else 0

def ddp_world() -> int:
    return torch.distributed.get_world_size() if ddp_is_on() else 1

def ddp_barrier():
    if ddp_is_on():
        torch.distributed.barrier()

def setup_for_distributed(is_master):
    """Disables printing when not in the master process."""
    builtin_print = builtins.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')
            builtin_print(*args, **kwargs)
    builtins.print = print

def ddp_all_gather_cat(x: torch.Tensor) -> torch.Tensor:
    """Gather tensor from all GPUs and concatenate (used for global metrics)."""
    if not ddp_is_on():
        return x
    xs = [torch.empty_like(x) for _ in range(ddp_world())]
    torch.distributed.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


# =============================================================================
# 2. EVALUATION
# =============================================================================
def compute_corr(labels: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    labels_centered = labels - torch.mean(labels, dim=0)
    preds_centered = preds - torch.mean(preds, dim=0)
    numerator = torch.sum(labels_centered * preds_centered)
    denominator = torch.sqrt(torch.sum(labels_centered ** 2)) * torch.sqrt(torch.sum(preds_centered ** 2))
    if denominator.item() == 0:
        return torch.tensor(0.0, device=labels.device)
    return numerator / denominator

# =============================================================================
# 3. EXPERT ABLATION CONFIGURATION
# =============================================================================
EXPERT_NAMES: List[str] = ["uO", "uN", "uF", "syn", "red"]

def build_expert_mask(ablation: str, expert_names: List[str] = EXPERT_NAMES) -> torch.Tensor:
    """
    Constructs a mask vector based on the specified ablation mode.

    Modes:
      "full"    - Retain all 5 experts
      "wo_syn"  - Remove syn, retain {uO, uN, uF, red}
      "wo_red"  - Remove red, retain {uO, uN, uF, syn}
      "wo_both" - Remove syn and red, retain {uO, uN, uF}

    Returns: FloatTensor [num_experts], 1.0 = retain, 0.0 = exclude
    """
    ablation_exclude = {
        "full":    [],
        "wo_syn":  ["syn"],
        "wo_red":  ["red"],
        "wo_both": ["syn", "red"],
    }
    if ablation not in ablation_exclude:
        raise ValueError(
            f"Unknown expert_ablation='{ablation}'. "
            f"Valid options: {list(ablation_exclude.keys())}"
        )
    excluded = set(ablation_exclude[ablation])
    mask = torch.tensor(
        [0.0 if name in excluded else 1.0 for name in expert_names],
        dtype=torch.float32,
    )
    return mask


# =============================================================================
# 4. CONFIGURATION (CFG)
# =============================================================================
@dataclass
class CFG:
    # Model Params
    project_heads: bool = True       
    single_pred_head: bool = True    
    freeze_encoders: bool = False     

    # Paths
    data_meta_root : str = "/path/to/dataset/meta/folder"
    data_root: str = "/path/to/dataset/folder"
    run_dir: str = ""
    resume_path: str = ""

    # --- task settings ---
    dataset_name: str = "HCP"   
    pretrained_path: str = ""     
    task_type: str = "regression"  
    task: str = "age"
    num_outputs: int = 1           
    splits: int = 0
    accum_iter: int = 1
    eval_mode: bool = False
    
    class_weights: str = "auto"  # Options: None, "auto", "w0,w1,..."
    
    # --- training params ---
    epochs: int = 100
    warmup_epochs: int = 10
    batch_size: int = 64 
    num_workers: int = 4
    patch_size: Tuple[int, int, int] = (16, 16, 16)

    lr: float = 1.7e-5
    weight_decay: float = 1e-4
    amp: bool = True
    grad_clip: float = 1.0

    params_json: str = ""

    # ----- input params -----
    use_crop: bool = False                            # False: full volume input; True: partial view input
    img_size: Tuple[int, int, int] = (96, 112, 96)    # used if input is full volume
    view_size: Tuple[int, int, int] = (64, 80, 64)    # used if input is partial view

    use_partial: bool = False           # UNUSED               
    drop_token: bool = False            # UNUSED                 
    use_brainmask: bool = False         # UNUSED  
 
    # ----- model params -----
    emb_dim: int = 384
    proj_dim: int = 128
    pred_dim: int = 64
    
    # ----- other params -----
    seed: int = 42
    log_every: int = 10

    expert_ablation: str = "full"  # Options: "full", "wo_syn", "wo_red", "wo_both"


# =============================================================================
# 5. GENERAL UTILITIES
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_ckpt(path: str, model, opt, sched, scaler, epoch: int, step: int, best_val):
    obj = {
        "epoch": epoch,
        "step": step,
        "best": best_val,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(obj, path)

def load_ckpt(path: str, model, opt, sched, scaler, device=None):
    """
    Load a checkpoint and restore model, optimizer, scheduler, and scaler states.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    map_location = device if device is not None else torch.device('cpu')
    checkpoint = torch.load(path, map_location=map_location)
    
    state_dict = checkpoint['model']
    
    # Handle DDP 'module.' prefix mismatches
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.module.load_state_dict(new_state_dict)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    
    if opt is not None and checkpoint.get('opt'):
        opt.load_state_dict(checkpoint['opt'])
    
    if sched is not None and checkpoint.get('sched'):
        sched.load_state_dict(checkpoint['sched'])
    
    if scaler is not None and checkpoint.get('scaler'):
        scaler.load_state_dict(checkpoint['scaler'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    best_val = checkpoint.get('best', None)
    
    print(f"[Resume] Loaded checkpoint from {path}")
    print(f"[Resume] Epoch={epoch}, Step={step}, Best={best_val}")
    
    return epoch, step, best_val

def open_csv(path: str, header: List[str]):
    exists = os.path.exists(path)
    f = open(path, "a", encoding="utf-8")
    if not exists:
        f.write(",".join(header) + "\n")
    else:
        if os.path.getsize(path) == 0:
            f.write(",".join(header) + "\n")
    f.flush()
    def write_row(row: List):
        f.write(",".join(map(str, row)) + "\n")
        f.flush()
    return f, write_row

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1"):
        return True
    if v in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

def freeze_vit_encoders(model):
    """
    Freezes the three ViT encoders (enc_O, enc_N, enc_F) within FullViTModel_Predict.
    """
    actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    frozen_params = 0
    encoder_names = ['enc_O', 'enc_N', 'enc_F']
    
    print("\n" + "="*50)
    print("Freezing ViT Encoders...")
    print("="*50)
    
    for encoder_name in encoder_names:
        if hasattr(actual_model, encoder_name):
            encoder = getattr(actual_model, encoder_name)
            encoder_params = 0
            
            for param in encoder.parameters():
                param.requires_grad = False
                encoder_params += param.numel()
            
            frozen_params += encoder_params
            print(f"✓ {encoder_name}: {encoder_params:,} parameters frozen")
        else:
            print(f"✗ Warning: {encoder_name} not found in model")
    
    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in actual_model.parameters())
    
    print("-"*50)
    print(f"Total params:      {total_params:,}")
    print(f"Frozen params:     {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"Trainable params:  {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print("="*50 + "\n")
    
    return frozen_params

def load_params(path: str) -> Dict:
    """Loads parameter configuration directly from a JSON file."""
    with open(path, "r") as f:
        params = json.load(f)
    
    if not isinstance(params, dict) or len(params) == 0:
        raise SystemExit(f"Invalid params_json format or empty file: {path}. Expected a flat dictionary.")
    
    return params


def load_cfg_from_run_dir(run_dir: str) -> CFG:
    """Restore full training CFG from a saved run_dir/config.json (for eval_mode)."""
    config_path = os.path.join(os.path.abspath(run_dir.strip()), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[EVAL] Saved config not found: {config_path}. "
            "Run eval on an existing training output directory that contains config.json."
        )

    with open(config_path, "r") as f:
        saved = json.load(f)

    cfg_fields = CFG.__dataclass_fields__.keys()
    kwargs = {field: getattr(CFG, field) for field in cfg_fields}
    for key, value in saved.items():
        if key in cfg_fields:
            kwargs[key] = value

    for key in ("patch_size", "img_size", "view_size"):
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(kwargs[key])

    return CFG(**kwargs)


def _load_pretrained_weights(model, pretrained_path: str, device):
    """Load pretrained backbone weights into model (training & eval)."""
    print(f"[Init] Loading pretrained weights from {pretrained_path} ...")
    checkpoint = torch.load(pretrained_path, map_location=device)
    state_dict = checkpoint["model"]

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    for k in ["mask_O", "mask_N", "mask_F"]:
        if k in new_state_dict:
            del new_state_dict[k]
            print(f"Removed mask token: {k}")

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"[Init] Pretrained weights loaded. {len(missing)} missing keys, {len(unexpected)} unexpected keys.")

# =============================================================================
# 6. TRAINING & EVALUATION RUNNERS
# =============================================================================
def epoch_train(cfg, model, loader, criterion, optimizer, scheduler, device, epoch, w_step,
                expert_mask: torch.Tensor = None):
    """
    Args:
        expert_mask: FloatTensor [num_experts], generated by build_expert_mask().
                     None indicates full mode (preserves original behavior).
    """
    model.train()
    optimizer.zero_grad()

    # Uniform weights for warmup: evenly distributed only among active experts
    if expert_mask is not None:
        active_mask = expert_mask.to(device)
        uniform_w = active_mask / active_mask.sum()
    else:
        uniform_w = torch.ones(len(EXPERT_NAMES), device=device) / len(EXPERT_NAMES)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    for iter_step, (batch) in enumerate(metric_logger.log_every(loader, cfg.log_every, f'Epoch: [{epoch}]')):
        if len(batch) >= 7:
            O, N, F, meta, masks, indices = batch[:7]
            masks, indices = masks.to(device), indices.to(device)
        else:
            O, N, F, meta = batch[:5]
            masks, indices = None, None

        labels = meta["label"]
        O, N, F, labels = O.to(device), N.to(device), F.to(device), labels.to(device)
        
        with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == "cuda")):
            if epoch <= cfg.warmup_epochs:
                # Warmup phase
                w_used = uniform_w.unsqueeze(0).expand(O.shape[0], -1)  # [B, E]
                outputs = model(O, N, F, w_used=w_used)
            else:
                outputs = model(O, N, F, expert_mask=expert_mask)
            
            preds = outputs["logits"]

            if cfg.task_type == "classification":
                loss = criterion(preds, labels.long())
            else:
                loss = criterion(preds.squeeze(), labels.float())
                
        loss /= cfg.accum_iter
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if (iter_step + 1) % cfg.accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()

        # Update metrics
        if cfg.task_type == "classification":
            loss_dict = {'loss': loss.item()}
        else:
            mae = torch.mean(torch.abs(labels - preds.squeeze()))
            mse = torch.mean((labels - preds.squeeze()) ** 2)
            corr = compute_corr(labels, preds.squeeze())
            loss_dict = {'loss': loss.item(), 'mae': mae.item(), 'corr': corr.item()}
        metric_logger.update(**loss_dict)

        min_lr, max_lr = 10.0, 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)  

        row = [epoch, iter_step, f"{max_lr:.6f}", f"{loss.item():.6f}"]
        w_step(row)
    
    scheduler.step()

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def epoch_eval(cfg, model, loader, criterion, log_writer, device, epoch=0, mode='val',
               save_pred_gt=False, expert_mask: torch.Tensor = None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    loss_sum = 0.0
    all_preds, all_labels, all_ids, all_raw_labels = [], [], [], []
    rw_weights_list = []

    for batch in metric_logger.log_every(loader, cfg.log_every, f"{mode} at Epoch: [{epoch}]"):
        if len(batch) >= 7:
            O, N, F, meta, masks, indices = batch[:7]
            masks, indices = masks.to(device), indices.to(device)
        else:
            O, N, F, meta = batch[:5]
            masks, indices = None, None

        raw_labels = meta["raw_label"]
        labels = meta["label"]
        subject_ids = meta["id"]
        O, N, F, labels = O.to(device), N.to(device), F.to(device), labels.to(device)

        if isinstance(subject_ids, str):
            subject_ids = [subject_ids]
        all_ids += subject_ids

        with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == "cuda")):
            outputs = model(O, N, F, expert_mask=expert_mask)
            pred_logits = outputs["logits"]

            if ddp_rank() == 0 and save_pred_gt:
                rw_weights_list.append(outputs["rw_weights"])

            if cfg.task_type == "classification":
                loss = criterion(pred_logits, labels.long())
            else:
                loss = criterion(pred_logits.squeeze(), labels.float())
        loss_sum += loss.item()

        # Monitor metrics
        batch_gt = labels.detach().cpu().numpy()
        if cfg.task_type == "classification":
            pred_probs = nn.functional.softmax(pred_logits, dim=1)
            batch_predict = torch.argmax(pred_probs, dim=1).detach().cpu().numpy()
            f1 = f1_score(batch_predict, batch_gt, average="macro")
            acc1 = accuracy_score(batch_gt, batch_predict) 
            loss_dict = {'val_loss': loss.item(), 'val_acc': acc1, 'val_f1': float(f1)}
        else:
            pred_logits = pred_logits.detach().cpu().numpy()
            batch_predict = pred_logits.squeeze()
            mae = mean_absolute_error(batch_predict, batch_gt)
            corr = compute_corr(torch.from_numpy(batch_predict), torch.from_numpy(batch_gt))
            loss_dict = {'val_loss': loss.item(), 'val_corr': corr.item(), 'val_mae': mae}

        all_preds.append(batch_predict)
        all_labels.append(batch_gt)
        all_raw_labels.append(raw_labels)
        metric_logger.update(**loss_dict)

    metric_logger.synchronize_between_processes()
    global_metrics = _compute_epoch_metrics(cfg, loader, loss_sum, all_preds, all_labels)

    if ddp_rank() == 0 and save_pred_gt:
        preds = np.concatenate(all_preds).squeeze()
        gts = np.concatenate(all_labels).squeeze()
        raw_gts = np.concatenate(all_raw_labels).squeeze()

        if cfg.task_type == "regression":
            task_name = UKB_TASKS[cfg.task] if cfg.dataset_name.upper() == "UKB" else cfg.task
            norm_stats_npy = f'/{cfg.data_meta_root}/{cfg.dataset_name.upper()}/label_stats/{cfg.dataset_name.upper()}_{task_name}_seed{cfg.splits}_stats.npy'

            if os.path.exists(norm_stats_npy):
                stats = np.load(norm_stats_npy)
                mean, std = stats[0], stats[1]
                preds = preds * std + mean
                gts = raw_gts  # Use raw test set labels
            else:
                print(f"[Warning] Stats file not found: {norm_stats_npy}")

        diff = preds - gts
        df = pd.DataFrame({
            "id": all_ids,
            "gt": gts,
            "pred": preds,
            "pred_minus_gt": diff,
        })
        df.to_csv(os.path.join(cfg.run_dir, f"{mode}_pred_gt.csv"), index=False)

        if cfg.task_type == "regression":
            global_metrics["val_mae_raw"] = np.mean(np.abs(gts - preds))
            global_metrics["val_corr_raw"] = float(pearsonr(gts, preds)[0])
        else:
            global_metrics[f'val_acc_raw'] = global_metrics['val_acc']
            global_metrics[f'val_f1_raw'] = global_metrics['val_f1']

    return global_metrics

def _compute_epoch_metrics(cfg, loader, loss_sum, all_preds, all_labels):
    metrics = {}
    if ddp_rank() == 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        avg_loss = loss_sum / len(loader)

        if cfg.task_type == "classification":
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro")
            metrics = {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1}
        else:
            mae = mean_absolute_error(all_labels, all_preds)
            corr = compute_corr(torch.from_numpy(all_labels), torch.from_numpy(all_preds))
            metrics = {'val_loss': avg_loss, 'val_corr': corr.item(), 'val_mae': mae}
    return metrics


# =============================================================================
# 7. MAIN SCRIPT
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    
    # Paths & General Params
    parser.add_argument("--data_root", type=str, default=CFG.data_root)
    parser.add_argument("--run_dir", type=str, required=True, help="Output directory for this run (required)")
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)
    parser.add_argument("--resume_path", type=str, default=CFG.resume_path, help="Path to checkpoint for resuming training")
    parser.add_argument("--use_crop", type=bool, default=CFG.use_crop)
    parser.add_argument("--view_size", type=int, default=CFG.view_size)
    parser.add_argument("--use_partial", type=str2bool, default=CFG.use_partial)
    parser.add_argument("--drop_token", type=str2bool, default=CFG.drop_token)

    # Grid Search Params
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--warmup_epochs", type=int, default=CFG.warmup_epochs)
    parser.add_argument("--params_json", type=str, default="")

    # Finetune Params
    parser.add_argument("--dataset_name", type=str, default=CFG.dataset_name, help="Dataset name: UKB, PICMAN, SINGER, or HCP")
    parser.add_argument("--pretrained_path", type=str, default=CFG.pretrained_path)
    parser.add_argument("--task_type", type=str, default=CFG.task_type)
    parser.add_argument("--task", type=str, default=CFG.task)
    parser.add_argument("--splits", type=int, default=CFG.splits)
    parser.add_argument("--num_outputs", type=int, default=CFG.num_outputs)
    parser.add_argument("--eval_mode", type=str2bool, default=CFG.eval_mode)
    parser.add_argument("--class_weights", type=str, default=CFG.class_weights,
                        help="Classification loss weights. Options: '' (none), 'auto', or comma-separated floats.")
    
    parser.add_argument("--freeze_encoders", type=str2bool, default=CFG.freeze_encoders, help="Whether to freeze ViT encoders")
    parser.add_argument("--expert_ablation", type=str, default=CFG.expert_ablation,
                        help="Expert ablation mode: 'full'| 'wo_syn' | 'wo_red' | 'wo_both'")

    args = parser.parse_args()

    # -----------------------------------------
    # Load config parameters
    # -----------------------------------------
    if args.params_json and not args.eval_mode:
        if not os.path.exists(args.params_json):
            raise FileNotFoundError(f"Params file not found: {args.params_json}")
        
        params = load_params(args.params_json)
        print(f"\n{'='*60}")
        print(f"[Config] Loading params from: {args.params_json}")
        print(f"{'='*60}")
        
        updated_params = []
        skipped_params = []
        
        for key, value in params.items():
            if hasattr(args, key):
                original_value = getattr(args, key)
                original_type = type(original_value)
                
                try:
                    if original_type == bool:
                        converted_value = value.lower() in ("true", "1", "yes") if isinstance(value, str) else bool(value)
                    elif original_type in (int, float, str):
                        converted_value = original_type(value)
                    else:
                        converted_value = value
                    
                    setattr(args, key, converted_value)
                    updated_params.append((key, converted_value))
                except (ValueError, TypeError) as e:
                    print(f"[Warning] Failed to convert '{key}': {value} to {original_type.__name__}: {e}")
                    skipped_params.append((key, value))
            else:
                skipped_params.append((key, value))
        
        if updated_params:
            print(f"\n[Config] Updated {len(updated_params)} parameter(s):")
            for key, value in updated_params:
                print(f"  - {key}: {value}")
        
        if skipped_params:
            print(f"\n[Config] Skipped {len(skipped_params)} parameter(s) (not in args):")
            for key, value in skipped_params:
                print(f"  - {key}: {value}")
        print(f"{'='*60}\n")

    # -----------------------------------------
    # Initialize CFG
    # -----------------------------------------
    if args.eval_mode:
        cfg = load_cfg_from_run_dir(args.run_dir)
        cfg.eval_mode = True
        cfg.run_dir = os.path.abspath(args.run_dir.strip())
        if args.resume_path:
            cfg.resume_path = args.resume_path
        if args.pretrained_path:
            cfg.pretrained_path = args.pretrained_path
        if args.data_root and args.data_root != CFG.data_root:
            cfg.data_root = args.data_root
        print(f"[EVAL] Restored config from {os.path.join(cfg.run_dir, 'config.json')}")
        print(f"[EVAL] resume_path={cfg.resume_path or os.path.join(cfg.run_dir, 'ckpt_best.pt')}")
        print(f"[EVAL] pretrained_path={cfg.pretrained_path}")
    else:
        cfg = CFG(
            data_root=args.data_root, run_dir=args.run_dir,
            epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
            resume_path=args.resume_path, use_partial=args.use_partial, drop_token=args.drop_token,
            lr=args.lr, weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
            params_json=args.params_json,
            dataset_name=args.dataset_name, pretrained_path=args.pretrained_path,
            task_type=args.task_type, task=args.task, num_outputs=args.num_outputs,
            splits=args.splits, eval_mode=args.eval_mode,
            freeze_encoders=args.freeze_encoders, expert_ablation=args.expert_ablation,
        )

    print("-"*20)
    if cfg.eval_mode:
        print(f"Evaluation mode: dataset={cfg.dataset_name}, task={cfg.task}, splits={cfg.splits}")
    else:
        print(f"Finetuning mode: use_partial={args.use_partial}, drop_token={args.drop_token}")
        print(f"Dataset: {args.dataset_name}")
        print(f"Downstream task: task={args.task}, splits={args.splits}")
    print("-"*20)

    # -----------------------------------------
    # System Setup 
    # -----------------------------------------
    if not cfg.eval_mode and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    
    rank = 0 if cfg.eval_mode else ddp_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        run_dir = os.path.abspath(cfg.run_dir.strip())
        os.makedirs(run_dir, exist_ok=True)
        cfg.run_dir = run_dir

        if not cfg.eval_mode:
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(asdict(cfg), f, indent=2)

        metric1_name = "acc" if cfg.task_type == "classification" else "mae"
        metric2_name = "f1" if cfg.task_type == "classification" else "corr"
        if cfg.eval_mode:
            f_step = w_step = f_ep = w_ep = None
        else:
            step_log_header = ["epoch", "step", "lr", "train_loss"] 
            epoch_log_header = ["epoch", "lr", "train_loss", "val_loss", f"val_{metric1_name}", f"val_{metric2_name}", "test_loss", f"test_{metric1_name}", f"test_{metric2_name}"] 
            f_step, w_step = open_csv(os.path.join(run_dir, "step_log.csv"), step_log_header)
            f_ep, w_ep = open_csv(os.path.join(run_dir, "epoch_log.csv"), epoch_log_header)
        f_test, w_test = open_csv(
            os.path.join(run_dir, "test_results.csv"),
            ['test_loss', f'test_{metric1_name}', f'test_{metric2_name}', f'test_{metric1_name}_raw', f'test_{metric2_name}_raw'],
        )
    else:
        run_dir = ""
        f_step = w_step = f_ep = w_ep = f_test = w_test = None
        
    if not cfg.eval_mode:
        ddp_barrier()
    setup_for_distributed(rank == 0)
    
    if not cfg.eval_mode and ddp_is_on():
        obj_list = [run_dir]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        run_dir = obj_list[0]

    set_seed(cfg.seed + rank)

    # -----------------------------------------
    # Data Preparation
    # -----------------------------------------
    if cfg.eval_mode:
        if cfg.dataset_name.lower() == "ukb":
            _, _, test_dataset = make_finetune_dataset(
                root=cfg.data_root, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                use_crop=cfg.use_crop, img_size=cfg.img_size, view_size=cfg.view_size,
                patch_size=cfg.patch_size, splits=cfg.splits, task=cfg.task, task_type=cfg.task_type,
                eval_only=True,
            )
        elif cfg.dataset_name.lower() in ["picman", "singer", "hcp"]:
            _, _, test_dataset = make_internal_finetune_dataset(
                splits=cfg.splits, dataset_name=cfg.dataset_name.upper(),
                target_col=cfg.task, task_type=cfg.task_type, target_size=cfg.img_size,
                clip_percentile=(1.0, 99.0), eval_only=True,
            )
        else:
            raise ValueError(f"Unknown dataset name: {cfg.dataset_name}")
        train_dataset = val_dataset = None
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=list_data_collate, pin_memory=True,
        )
        train_loader = val_loader = None
        num_batches_per_epoch = 0
    else:
        if cfg.dataset_name.lower() == "ukb":
            train_dataset, val_dataset, test_dataset = make_finetune_dataset(
                root=cfg.data_root, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                use_crop=cfg.use_crop, img_size=cfg.img_size, view_size=cfg.view_size,
                patch_size=cfg.patch_size, splits=cfg.splits, task=cfg.task, task_type=cfg.task_type,
            )
        elif cfg.dataset_name.lower() in ["picman", "singer", "hcp"]:
            train_dataset, val_dataset, test_dataset = make_internal_finetune_dataset(
                splits=cfg.splits, dataset_name=cfg.dataset_name.upper(),
                target_col=cfg.task, task_type=cfg.task_type, target_size=cfg.img_size,
                clip_percentile=(1.0, 99.0),
            )
        else:
            raise ValueError(f"Unknown dataset name: {cfg.dataset_name}")

        if ddp_is_on():
            sampler_train = torch.utils.data.DistributedSampler(train_dataset, num_replicas=ddp_world(), rank=ddp_rank(), shuffle=True)
            print("Sampler_train = %s" % str(sampler_train))

            if len(val_dataset) % ddp_world() != 0:
                print('Warning: Eval dataset size is not divisible by process number. '
                      'This will slightly alter validation results due to duplicate entries.')
            sampler_val = torch.utils.data.DistributedSampler(val_dataset, num_replicas=ddp_world(), rank=ddp_rank(), shuffle=True)
            sampler_test = torch.utils.data.DistributedSampler(test_dataset, num_replicas=ddp_world(), rank=ddp_rank(), shuffle=True)
        else:
            sampler_train = torch.utils.data.RandomSampler(train_dataset)    
            sampler_val = torch.utils.data.RandomSampler(val_dataset)
            sampler_test = torch.utils.data.RandomSampler(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler_train, num_workers=cfg.num_workers, collate_fn=list_data_collate, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=sampler_val, num_workers=cfg.num_workers, collate_fn=list_data_collate, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, sampler=sampler_test, num_workers=cfg.num_workers, collate_fn=list_data_collate, pin_memory=True)
        num_batches_per_epoch = len(train_loader)

    # -----------------------------------------
    # Model Initialization
    # -----------------------------------------
    model_args = {
        "emb_dim": cfg.emb_dim, "out_dim": cfg.num_outputs, "pred_dim": cfg.pred_dim,
        "patch_size": cfg.patch_size, "img_size": cfg.img_size,
        "project_heads": cfg.project_heads, "single_pred_head": cfg.single_pred_head,
        "use_brainmask": cfg.use_brainmask
    }
    if cfg.use_partial:
        model_args["view_size"] = cfg.view_size

    model = FullViTModel_Predict(**model_args).to(device)
    print(model)

    # Load Pretrained Weights (training only; eval loads finetuned ckpt below)
    if not cfg.eval_mode:
        if cfg.pretrained_path and os.path.exists(cfg.pretrained_path):
            _load_pretrained_weights(model, cfg.pretrained_path, device)
        else:
            if cfg.pretrained_path:
                print(f"[Warning] Pretrained path specified but not found: {cfg.pretrained_path}")
            print("[Init] Training from SCRATCH.")

    # Freeze ViT Encoders if configured
    if cfg.freeze_encoders:
        freeze_vit_encoders(model)

    # Wrap DDP (training only; eval uses single-process inference)
    if ddp_is_on() and not cfg.eval_mode:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))], find_unused_parameters=True
        )
    
    if rank == 0:
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"\n[Model Statistics]\nTrainable params: {num_trainable:,}\nTotal params:     {num_total:,}\nTraining ratio:   {num_trainable/num_total*100:.2f}%\n")

    # Experts Setup
    expert_mask = build_expert_mask(cfg.expert_ablation, EXPERT_NAMES)
    expert_mask_device = expert_mask.to(device)
    if rank == 0:
        active_experts = [n for n, m in zip(EXPERT_NAMES, expert_mask.tolist()) if m > 0]
        print(f"[Expert Ablation] mode='{cfg.expert_ablation}', active experts={active_experts}")
    _mask_for_model = None if cfg.expert_ablation == "full" else expert_mask_device

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    
    if cfg.task_type == "classification":
        weight_tensor = None
        if cfg.class_weights and not cfg.eval_mode:
            if cfg.class_weights.strip().lower() == "auto":
                train_labels = np.array([int(v) for v in train_dataset.labels.values()], dtype=np.int64)
                classes, counts = np.unique(train_labels, return_counts=True)
                n_total, n_classes = len(train_labels), len(classes)
                weights = np.zeros(cfg.num_outputs, dtype=np.float32)
                for cls, cnt in zip(classes, counts):
                    weights[cls] = n_total / (n_classes * cnt)
                weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
                print(f"[Loss] Auto class weights (train only): {dict(zip(classes.tolist(), weights.tolist()))}")
            else:
                weights = [float(w) for w in cfg.class_weights.split(",")]
                weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
                print(f"[Loss] Manual class weights: {weights}")

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        best_metric_name, best_mode = "val_acc", "max"
    else:
        criterion = nn.MSELoss()
        best_metric_name, best_mode = "val_mae", "min"
    
    # State tracking
    start_epoch, global_step, best_val = 1, 0, None

    def _load_model_weights(ckpt_path: str, strict: bool = False):
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model"]
        target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        missing, unexpected = target.load_state_dict(new_state_dict, strict=strict)
        if missing:
            print(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"[Checkpoint] Loaded model weights from {ckpt_path}")

    if cfg.resume_path and os.path.exists(cfg.resume_path) and not cfg.eval_mode:
        print(f"[Resume] Resuming training from checkpoint: {cfg.resume_path}")
        e0, s0, best0 = load_ckpt(cfg.resume_path, model, optimizer, lr_scheduler, scaler, device=device)
        start_epoch, global_step, best_val = e0 + 1, s0, best0
        print(f"[Resume] Will resume from epoch {start_epoch}")
    elif cfg.resume_path and not cfg.eval_mode:
        print(f"[Warning] Resume path specified but not found: {cfg.resume_path}\n[Info] Starting training from scratch")

    # -----------------------------------------
    # Evaluation Mode: load finetuned checkpoint and run final testing
    # -----------------------------------------
    if cfg.eval_mode:
        print("[EVAL] Starting evaluation mode.")
        eval_ckpt = cfg.resume_path or os.path.join(run_dir, "ckpt_best.pt")
        if not os.path.exists(eval_ckpt):
            raise FileNotFoundError(
                f"[EVAL] Finetuned checkpoint not found: {eval_ckpt}. "
                "Pass --resume_path or place ckpt_best.pt under --run_dir."
            )

        _load_model_weights(eval_ckpt, strict=True)
        test_metrics = epoch_eval(
            cfg, model, test_loader, criterion, None, device, epoch=0,
            mode="test", save_pred_gt=True, expert_mask=_mask_for_model,
        )

        if rank == 0:
            row = [
                test_metrics["val_loss"],
                test_metrics[f"val_{metric1_name}"],
                test_metrics[f"val_{metric2_name}"],
                test_metrics.get(f"val_{metric1_name}_raw", ""),
                test_metrics.get(f"val_{metric2_name}_raw", ""),
            ]
            w_test(row)
            print("=" * 30)
            print("FINAL TEST RESULTS (EVAL MODE):")
            print(f"Test Loss: {test_metrics['val_loss']:.4f}")
            print(f"Metric ({metric1_name}): {test_metrics[f'val_{metric1_name}']:.4f}, "
                  f"Raw: {test_metrics.get(f'val_{metric1_name}_raw', float('nan')):.4f}")
            print(f"Metric ({metric2_name}): {test_metrics[f'val_{metric2_name}']:.4f}, "
                  f"Raw: {test_metrics.get(f'val_{metric2_name}_raw', float('nan')):.4f}")
            print(f"Predictions saved to: {os.path.join(run_dir, 'test_pred_gt.csv')}")
            print(f"Results appended to: {os.path.join(run_dir, 'test_results.csv')}")
            print("=" * 30)
            f_test.close()
        return

    # -----------------------------------------
    # Main Training Loop
    # -----------------------------------------
    if best_val is not None:
        best_val_score = best_val
        print(f"[Resume] Initialized best_val_score from checkpoint: {best_val_score:.4f}")
    else:
        best_val_score = -float('inf') if best_mode == "max" else float('inf')

    print(f"Start training from epoch {start_epoch} to {cfg.epochs}. Task: {cfg.task_type}")
    
    for epoch in range(start_epoch, cfg.epochs + 1):
        if ddp_is_on() and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_metrics = epoch_train(cfg, model, train_loader, criterion, optimizer, lr_scheduler, device, epoch, w_step, expert_mask=_mask_for_model)
        val_metrics = epoch_eval(cfg, model, val_loader, criterion, None, device, epoch, mode='val', expert_mask=_mask_for_model)
        test_metrics = epoch_eval(cfg, model, test_loader, criterion, None, device, epoch, mode='test', expert_mask=_mask_for_model)

        if ddp_is_on():
            torch.cuda.empty_cache()

        row = [epoch, f"{optimizer.param_groups[0]['lr']:.2e}", f"{train_metrics['loss']:.6f}", 
               f"{val_metrics['val_loss']:.6f}", f"{val_metrics[f'val_{metric1_name}']:.6f}", f"{val_metrics[f'val_{metric2_name}']:.6f}", 
               f"{test_metrics['val_loss']:.6f}", f"{test_metrics[f'val_{metric1_name}']:.6f}", f"{test_metrics[f'val_{metric2_name}']:.6f}"]
        w_ep(row)

        if rank == 0:
            save_ckpt(os.path.join(run_dir, "ckpt_latest.pt"), model, optimizer, lr_scheduler, scaler, epoch, global_step, val_metrics[best_metric_name])
            
            improved = (val_metrics[best_metric_name] > best_val_score) if cfg.task_type == "classification" else (val_metrics[best_metric_name] < best_val_score)
            if improved:
                best_val_score = val_metrics[best_metric_name]
                save_ckpt(os.path.join(run_dir, "ckpt_best.pt"), model, optimizer, lr_scheduler, scaler, epoch, global_step, best_val_score)
                print(f"[Best] update best model at epoch={epoch} ({best_metric_name}: {best_val_score:.4f}), {os.path.join(run_dir, 'ckpt_best.pt')}")

        ddp_barrier()

    # -----------------------------------------
    # Final Testing
    # -----------------------------------------
    if rank == 0:
        print("\nTraining Finished. Loading Best Model for Testing...")
    ddp_barrier()
    
    best_path = os.path.join(run_dir, "ckpt_best.pt") if rank == 0 else ""
    
    if os.path.exists(best_path):
        _load_model_weights(best_path)
    
    test_metrics = epoch_eval(cfg, model, test_loader, criterion, None, device, epoch, mode='test', save_pred_gt=True, expert_mask=_mask_for_model)
    
    if rank == 0:
        row = [test_metrics['val_loss'], test_metrics[f'val_{metric1_name}'], test_metrics[f'val_{metric2_name}'], 
               test_metrics[f'val_{metric1_name}_raw'], test_metrics[f'val_{metric2_name}_raw']]
        w_test(row)

        print("="*30)
        print("FINAL TEST RESULTS:")
        print(f"Test Loss: {test_metrics['val_loss']:.4f}")
        print(f"Metric ({metric1_name}): {test_metrics[f'val_{metric1_name}']:.4f}, Raw: {test_metrics[f'val_{metric1_name}_raw']:.4f}")
        print(f"Metric ({metric2_name}): {test_metrics[f'val_{metric2_name}']:.4f}, Raw: {test_metrics[f'val_{metric2_name}_raw']:.4f}")
        print("="*30)

        f_step.close()
        f_ep.close()
        f_test.close()
    ddp_barrier()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()