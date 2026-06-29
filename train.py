# train.py
# -*- coding: utf-8 -*-

import argparse
import builtins
import datetime
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from monai.data import DataLoader, list_data_collate

from datasets.dataloader_pretrain import prepare_dataset, TwoViewGPUAugmentor
from src.models.vit_model import FullViTModel

# =============================================================================
# 1. DISTRIBUTED DATA PARALLEL (DDP) UTILS
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
    """
    Disables printing when not in the master process.
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (ddp_world() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')
            builtin_print(*args, **kwargs)

    builtins.print = print

def ddp_all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_on():
        return x
    y = x.clone()
    torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
    y /= ddp_world()
    return y

@torch.no_grad()
def ddp_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    """
    All_gather without gradients (used for cross-GPU negatives).
    """
    if not ddp_is_on():
        return x
    xs = [torch.empty_like(x) for _ in range(ddp_world())]
    torch.distributed.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)

def set_reweighter_trainable(model, trainable: bool):
    m = model.module if hasattr(model, "module") else model
    for p in m.reweighter.parameters():
        p.requires_grad = trainable


# =============================================================================
# 2. CONFIGURATION
# =============================================================================
@dataclass
class CFG:
    data_root: str = "./DTI_sample"
    run_root: str = "./output"
    run_dir: str = ""
    
    # ----- training params -----
    epochs: int = 200
    warmup_epochs: int = 5     
    batch_size: int = 280
    num_workers: int = 4                              
    patch_size: Tuple[int, int, int] = (16, 16, 16)   

    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    amp: bool = True
    grad_clip: float = 1.0

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

    # ----- loss function params -----
    # temperature
    tau_task: float = 0.2    # Fused SimCLR
    tau_inter: float = 0.12  # Interaction multi-pos InfoNCE
    # weights
    task_weight: float = 1.0     
    alpha_inter: float = 1.0     
    beta_balance: float = 0.05   # KL(mean(w)||uniform)
    gamma_entropy: float = 0.01  # -Entropy(w)


    # ----- other params -----
    seed: int = 42
    log_every: int = 6
    save_every_epochs: int = 20
    resume_path: str = ""

    best_metric: str = "loss"    # "loss" or "syn_sim_b0"
    best_mode: str = "min"       # loss -> "min", sim -> "max"

    cross_gpu_task_neg: bool = False   # Set to True for cross-GPU negatives (requires sync)
    cross_gpu_inter_neg: bool = False


# =============================================================================
# 3. INTERACTION RULES & CANDIDATE GENERATION
# =============================================================================
CAND_ORDER = ["b0", "drop_O", "drop_N", "drop_F", "swap_O", "swap_N", "swap_F"]

def pos_rules():
    """
    Interation rules defining which candidates each expert treats as positives.
    """
    return {
        # uO relies on O. Positive if N/F is dropped or swapped. Negative if O is altered.
        "uO": ["b0", "drop_N", "drop_F", "swap_N", "swap_F"],
        "uN": ["b0", "drop_O", "drop_F", "swap_O", "swap_F"],
        "uF": ["b0", "drop_O", "drop_N", "swap_O", "swap_N"],
        # Redundancy handles missing modalities robustly (drop is okay, swap is strictly wrong)
        "red": ["b0", "drop_O", "drop_N", "drop_F"],
        # Synergy requires all modalities to be perfectly intact and aligned
        "syn": ["b0"],
    }

def derangement_perm(B: int, device) -> torch.Tensor:
    """
    Returns a permutation where perm[i] != i.
    """
    if B <= 1:
        return torch.arange(B, device=device)
    while True:
        perm = torch.randperm(B, device=device)
        if torch.all(perm != torch.arange(B, device=device)):
            return perm

def rand_like_stats(x: torch.Tensor) -> torch.Tensor:
    """
    Random masking using batch mean and std for scale matching.
    """
    mu = x.mean()
    sd = x.std() + 1e-6
    return torch.randn_like(x) * sd + mu

def build_candidates_viewb(
        hOb: torch.Tensor, hNb: torch.Tensor, hFb: torch.Tensor,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Constructs drop/swap triplet candidates at the embedding level.
    """
    B, _ = hOb.shape
    device = hOb.device
    perm = derangement_perm(B, device=device)

    rO = rand_like_stats(hOb)
    rN = rand_like_stats(hNb)
    rF = rand_like_stats(hFb)

    return {
        "b0": (hOb, hNb, hFb),
        "drop_O": (rO, hNb, hFb),
        "drop_N": (hOb, rN, hFb),
        "drop_F": (hOb, hNb, rF),
        "swap_O": (hOb[perm], hNb, hFb),
        "swap_N": (hOb, hNb[perm], hFb),
        "swap_F": (hOb, hNb, hFb[perm]),
    }

def build_pos_mask(B: int, device, pos_conds: List[str]) -> torch.Tensor:
    """
    Builds a positive sample mask matrix of shape [B, B*7].
    """
    pos_mask = torch.zeros((B, B * len(CAND_ORDER)), dtype=torch.bool, device=device)
    idx_i = torch.arange(B, device=device)
    for c in pos_conds:
        blk = CAND_ORDER.index(c)
        idx_j = idx_i + blk * B
        pos_mask[idx_i, idx_j] = True
    return pos_mask

def multipos_infonce_per_sample(
        A: torch.Tensor,         # anchor, view-a expert feature k: [B,d]
        C: torch.Tensor,         # candidate pool, view-b expert features (7 perturbations): [B*7,d]
        pos_mask: torch.Tensor,  # positive sample mask, indicating which samples are positive: [B,B*7]
        tau: float
) -> torch.Tensor:

    """
    Multi-positive InfoNCE loss computation per sample: log p(pos) = logsumexp(pos logits) - logsumexp(all logits)
    """
    S = (A @ C.t()) / tau
    S_max = S.max(dim=1, keepdim=True).values
    expS = torch.exp(S - S_max)

    denom = expS.sum(dim=1) + 1e-12
    pos = (expS * pos_mask).sum(dim=1) + 1e-12

    loss = -torch.log(pos / denom)
    return loss


# =============================================================================
# 4. TRAINING UTILS (Losses & Fusion)
# =============================================================================
def diag_metrics_from_S(S: torch.Tensor, B: int) -> Dict[str, torch.Tensor]:
    """
    Extracts diagonal similarity metrics between anchor and candidates.
    -
    S: [B, B*7] (anchor vs candidates)
    -
    Returns: dict(cond -> [B])
    """
    out = {}
    for bi, cond in enumerate(CAND_ORDER):
        blk = S[:, bi * B:(bi + 1) * B]
        out[cond] = blk.diag()
    return out

def fuse_q_by_w(q_dict: Dict[str, torch.Tensor], w: torch.Tensor, names: List[str]) -> torch.Tensor:
    """
    Weighted fusion of expert representations followed by L2 normalization.
    -
    q_dict: name -> [B,D]
    w: [B,E]
    -
    return q_fused: [B,D] (L2 norm)
    """
    q_fused = 0.0
    for i, n in enumerate(names):
        q_fused = q_fused + w[:, i:i + 1] * q_dict[n]
    return F.normalize(q_fused, p=2, dim=-1)

def simclr_infonce(qa: torch.Tensor, qb: torch.Tensor, tau: float, cross_gpu: bool) -> torch.Tensor:
    """
    Standard SimCLR InfoNCE with optional cross-GPU negative aggregation.
    -
    qa: [B,D]
    qb: [B,D]
    tau: temperature parameter
    cross_gpu: bool, whether to use cross-GPU negative aggregation
    -
    return loss: [B]
    """
    B = qa.shape[0]
    if cross_gpu and ddp_is_on():
        qb_all = ddp_gather_no_grad(qb)
        qa_all = ddp_gather_no_grad(qa)
        rank = ddp_rank()
        labels = torch.arange(B, device=qa.device) + rank * B

        logits_a2b = (qa @ qb_all.t()) / tau
        loss_a2b = F.cross_entropy(logits_a2b, labels)

        logits_b2a = (qb @ qa_all.t()) / tau
        loss_b2a = F.cross_entropy(logits_b2a, labels)
        return 0.5 * (loss_a2b + loss_b2a)

    labels = torch.arange(B, device=qa.device)
    logits_a2b = (qa @ qb.t()) / tau
    logits_b2a = (qb @ qa.t()) / tau
    return 0.5 * (F.cross_entropy(logits_a2b, labels) + F.cross_entropy(logits_b2a, labels))


# =============================================================================
# 5. IO & LOGGING HELPERS
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_run_dir(root: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = os.path.join(root, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

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

def load_ckpt(path: str, model, opt=None, sched=None, scaler=None, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    model.load_state_dict(obj["model"], strict=True)
    if opt is not None and obj.get("opt") is not None:
        opt.load_state_dict(obj["opt"])
    if sched is not None and obj.get("sched") is not None:
        sched.load_state_dict(obj["sched"])
    if scaler is not None and obj.get("scaler") is not None:
        scaler.load_state_dict(obj["scaler"])
    return obj.get("epoch", 0), obj.get("step", 0), obj.get("best", None)

def open_csv(path: str, header: List[str]):
    exists = os.path.exists(path)
    f = open(path, "a", encoding="utf-8")
    if not exists:
        f.write(",".join(header) + "\n")
        f.flush()

    def write_row(row: List):
        f.write(",".join(map(str, row)) + "\n")
        f.flush()

    return f, write_row

def compute_diag_cache(A: torch.Tensor, C: torch.Tensor, B: int, tau: float) -> Dict[str, torch.Tensor]:
    S = (A @ C.t()) / tau
    return diag_metrics_from_S(S, B)

def log_memory_usuage(tag=''):
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 1024**2
    memory_res = torch.cuda.max_memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"{tag} Memory: {memory:.2f} MB, Res: {memory_res:.2f} MB, Total: {total:.2f} MB, Usage: {memory/total*100:.2f}%")

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1"):
        return True
    if v in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


# =============================================================================
# 6. MAIN TRAINING SCRIPT
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=CFG.data_root)
    parser.add_argument("--run_root", type=str, default=CFG.run_root)
    parser.add_argument("--run_dir", type=str, default=CFG.run_dir)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--use_crop", type=bool, default=CFG.use_crop)
    parser.add_argument("--view_size", type=int, default=CFG.view_size)
    parser.add_argument("--use_partial", type=str2bool, default=CFG.use_partial)
    parser.add_argument("--drop_token", type=str2bool, default=CFG.drop_token)
    args = parser.parse_args()

    # Initialize CFG
    cfg = CFG(
        data_root=args.data_root,
        run_root=args.run_root,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume_path=args.resume,
        use_partial=args.use_partial,
        drop_token=args.drop_token
    )

    # -----------------------------------------
    # System Setup (DDP & Device Info)
    # -----------------------------------------
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl") 
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        print(f"Initialized DDP with rank={os.environ.get('RANK')}, world_size={os.environ.get('WORLD_SIZE')}, local_rank={os.environ.get('LOCAL_RANK')}")

    rank = ddp_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        print(f"Rank {rank} using device {local_rank}, UUID = {torch.cuda.get_device_properties(local_rank).uuid}")

    # Synchronize Directory Logging
    if rank == 0:
        run_dir = cfg.run_dir.strip()
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
        else:
            run_dir = make_run_dir(cfg.run_root)
        cfg.run_dir = run_dir
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    else:
        run_dir = None
    ddp_barrier()
    setup_for_distributed(rank == 0)

    if ddp_is_on():
        obj_list = [run_dir]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        run_dir = obj_list[0]

    set_seed(cfg.seed + rank)

    # -----------------------------------------
    # Data Preparation
    # -----------------------------------------
    data_loader_start = time.time()
    ds = prepare_dataset(
        cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_crop=cfg.use_crop,
        img_size=cfg.img_size,
        view_size=cfg.view_size,
        patch_size=cfg.patch_size,
        device='cpu',
    )
    
    if ddp_is_on():
        sampler = DistributedSampler(
            dataset=ds,
            num_replicas=ddp_world(),
            rank=ddp_rank(),
            shuffle=True,
        )
    else:
        sampler = torch.utils.data.RandomSampler(ds)    

    # Caution: DataLoader defaults to fork. Subprocesses copy memory but not CUDA context.
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=2,
        collate_fn=list_data_collate,
        pin_memory=True, 
        persistent_workers=True
    )
    data_loader_end = time.time()
    print(f"DataLoader initialization time: {data_loader_end - data_loader_start:.4f}s")

    # -----------------------------------------
    # Model Initialization
    # -----------------------------------------
    model_kwargs = {
        "emb_dim": cfg.emb_dim,
        "proj_dim": cfg.proj_dim,
        "patch_size": cfg.patch_size,
        "img_size": cfg.img_size,
        "use_brainmask": cfg.use_brainmask
    }
    if cfg.use_partial:
        model_kwargs["view_size"] = cfg.view_size

    model = FullViTModel(**model_kwargs).to(device)
    if ddp_is_on():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))], find_unused_parameters=False)
        print("Initialized model with DDP")
    else:
        print("Initialized model without DDP")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {num_trainable:,}\nTotal params:     {num_total:,}")

    # Optimizer & Scheduler & Augmentor
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    gpu_augmentor = TwoViewGPUAugmentor(
        view_size=cfg.view_size,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        prob_affine=0.5,
        prob_flip=0.2,
        use_crop=cfg.use_crop,
        device=device
    )

    # Resume Checkpoint
    start_epoch = 1
    global_step = 0
    best_val = None
    if cfg.resume_path:
        e0, s0, best0 = load_ckpt(cfg.resume_path, model, opt, sched, scaler, map_location=device)
        start_epoch = e0 + 1
        global_step = s0
        best_val = best0
        if rank == 0:
            print(f"[Resume] epoch={e0} step={s0} best={best_val} from {cfg.resume_path}") 

    # Intialize Interaction Rules 
    pos_conds_dict = pos_rules()
    expert_names = getattr(model.module.experts if hasattr(model, "module") else model.experts, "names",
                           ["uO", "uN", "uF", "syn", "red"])
    E = len(expert_names)
    uniform_w = torch.ones(E, device=device) / E

    # Set up Logs
    if rank == 0:
        step_csv = os.path.join(run_dir, "step_log.csv")
        epoch_csv = os.path.join(run_dir, "epoch_log.csv")
        f_step, w_step = open_csv(step_csv, [
            "epoch", "step", "loss", "loss_task", "loss_inter", "loss_balance", "loss_entropy", "lr",
            "syn_sim_b0", "w_uO", "w_uN", "w_uF", "w_syn", "w_red"
        ])
        f_ep, w_ep = open_csv(epoch_csv, [
            "epoch", "epoch_loss", "epoch_loss_task", "epoch_loss_inter", "lr",
            "epoch_syn_sim_b0", "epoch_w_uO", "epoch_w_uN", "epoch_w_uF", "epoch_w_syn", "epoch_w_red",
            "metric_used"
        ])
    else:
        f_step = w_step = f_ep = w_ep = None

    # -----------------------------------------
    # Training Loop
    # -----------------------------------------
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            if ddp_is_on() and hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch)
            ddp_barrier()

            model.train()
            loss_sum, task_sum, inter_sum, syn_sim_sum, n_steps = 0.0, 0.0, 0.0, 0.0, 0
            w_sum = torch.zeros(E, device=device)
            
            pbar = tqdm(range(len(loader)), disable=(rank != 0), desc=f"epoch {epoch}/{cfg.epochs}", dynamic_ncols=True)
            data_iter = iter(loader)

            for it in pbar:
                batch = next(data_iter)
                global_step += 1
                with torch.no_grad():
                    Oa, Na, Fa, Ob, Nb, Fb, (Ma, Mb), (Ia, Ib), metas = gpu_augmentor(batch)

                with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                    # -- Encode embeddings for both views --
                    core = model.module if hasattr(model, "module") else model
                    hOa, hNa, hFa = core.encode(Oa, Na, Fa, masks=Ma, indices=Ia, drop_token=cfg.drop_token)
                    hOb, hNb, hFb = core.encode(Ob, Nb, Fb, masks=Mb, indices=Ib, drop_token=cfg.drop_token)

                    # -- Obtain expert weights via reweighter --
                    w_a, _ = core.reweighter(torch.cat([hOa, hNa, hFa], dim=-1))
                    w_b, _ = core.reweighter(torch.cat([hOb, hNb, hFb], dim=-1))
                    
                    w_mean = w_a.mean(dim=0)
                    w_sum += w_mean.detach()

                    # -- Extract expert embeddings & fuse --
                    qA = core.experts.forward_q(hOa, hNa, hFa)
                    qB0 = core.experts.forward_q(hOb, hNb, hFb)
                    q_fused_a = fuse_q_by_w(qA, w_a, expert_names)
                    q_fused_b = fuse_q_by_w(qB0, w_b, expert_names)

                    # ================= (1) TASK LOSS =================
                    loss_task = simclr_infonce(
                        q_fused_a, q_fused_b, tau=cfg.tau_task,
                        cross_gpu=(cfg.cross_gpu_task_neg)
                    )

                    # ================= (2) INTERACTION LOSS =================
                    cands = build_candidates_viewb(hOb, hNb, hFb)
                    qB_by_cond: Dict[str, Dict[str, torch.Tensor]] = {}  
                    for cond in CAND_ORDER:
                        cO, cN, cF = cands[cond]
                        qB_by_cond[cond] = core.experts.forward_q(cO, cN, cF)

                    cands_a = build_candidates_viewb(hOa, hNa, hFa)
                    qA_by_cond: Dict[str, Dict[str, torch.Tensor]] = {}
                    for cond in CAND_ORDER:
                        cO, cN, cF = cands_a[cond]
                        qA_by_cond[cond] = core.experts.forward_q(cO, cN, cF)

                    B = hOa.shape[0]
                    per_sample_losses_ab = []
                    per_sample_losses_ba = []
                    diag_cache = {}

                    for k in expert_names:
                        # A->B Perspective
                        A_ab = F.normalize(qA[k], p=2, dim=-1)
                        C_ab = F.normalize(
                            torch.cat([qB_by_cond[cond][k] for cond in CAND_ORDER], dim=0), p=2, dim=-1
                        )
                        pos_mask = build_pos_mask(B, device=A_ab.device, pos_conds=pos_conds_dict[k])
                        Li_ab = multipos_infonce_per_sample(A_ab, C_ab, pos_mask, tau=cfg.tau_inter)
                        per_sample_losses_ab.append(Li_ab)

                        # B->A Perspective
                        A_ba = F.normalize(qB0[k], p=2, dim=-1)
                        C_ba = F.normalize(
                            torch.cat([qA_by_cond[cond][k] for cond in CAND_ORDER], dim=0), p=2, dim=-1
                        )
                        Li_ba = multipos_infonce_per_sample(A_ba, C_ba, pos_mask, tau=cfg.tau_inter)
                        per_sample_losses_ba.append(Li_ba)

                        # Cache evaluation stats
                        if (rank == 0) and (global_step % cfg.log_every == 0) and (k in ["uO", "uN", "uF", "syn", "red"]):
                            diag_ab = compute_diag_cache(A_ab, C_ab, B, cfg.tau_inter)
                            diag_ba = compute_diag_cache(A_ba, C_ba, B, cfg.tau_inter)
                            diag_cache[k] = {cond: 0.5 * (diag_ab[cond] + diag_ba[cond]) for cond in CAND_ORDER}

                    L_ik_ab = torch.stack(per_sample_losses_ab, dim=1)
                    L_ik_ba = torch.stack(per_sample_losses_ba, dim=1)

                    # Warmup Phase / Interaction Computation
                    if epoch <= cfg.warmup_epochs:
                        w_used = uniform_w.unsqueeze(0).expand(B, -1)
                    else:
                        w_used = w_a.detach()

                    loss_inter_ab = (w_used * L_ik_ab).sum(dim=1).mean()
                    loss_inter_ba = (w_used * L_ik_ba).sum(dim=1).mean()
                    loss_inter = 0.5 * (loss_inter_ab + loss_inter_ba)

                    # ================= (3) ANTI-COLLAPSE REGULARIZATION =================
                    # Balance Loss: Encourage uniform assignment across experts globally
                    w_bar = w_a.mean(dim=0) + 1e-12
                    u = torch.ones_like(w_bar) / E
                    loss_balance = (w_bar * (w_bar.log() - u.log())).sum()

                    # Entropy Loss: Encourage softer assignments per individual sample
                    ent = -(w_a * (w_a + 1e-12).log()).sum(dim=1).mean()
                    loss_entropy = -ent 

                    # Final loss
                    loss = (cfg.task_weight * loss_task + 
                            cfg.alpha_inter * loss_inter + 
                            cfg.beta_balance * loss_balance + 
                            cfg.gamma_entropy * loss_entropy)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()

                # Accumulate metrics
                cur_loss = float(loss.item())
                loss_sum += cur_loss
                task_sum += float(loss_task.item())
                inter_sum += float(loss_inter.item())
                n_steps += 1

                if "syn" in diag_cache:
                    syn_sim_b0 = float(diag_cache["syn"]["b0"].mean().item())
                else:
                    syn_sim_b0 = float((qA["syn"] * qB0["syn"]).sum(dim=1).mean().item())
                syn_sim_sum += syn_sim_b0

                # -----------------------------------------
                # Step Logging
                # -----------------------------------------
                if rank == 0:
                    pbar.set_postfix(loss=f"{cur_loss:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}", step=str(global_step))

                    if global_step % cfg.log_every == 0:
                        lines = [f"[Diag] epoch={epoch} step={global_step} loss={cur_loss:.4f}"]
                        if diag_cache:
                            for k in ["uO", "uN", "uF", "red", "syn"]:
                                b0 = float(diag_cache[k]["b0"].mean().item())
                                deltas = {c: float((diag_cache[k]["b0"] - diag_cache[k][c]).mean().item()) for c in ["drop_O", "drop_N", "drop_F", "swap_O", "swap_N", "swap_F"]}
                                lines.append(
                                    f"{k}: sim_b0={b0:.3f} "
                                    f"Δdrop={deltas['drop_O']:.3f}/{deltas['drop_N']:.3f}/{deltas['drop_F']:.3f} "
                                    f"Δswap={deltas['swap_O']:.3f}/{deltas['swap_N']:.3f}/{deltas['swap_F']:.3f}"
                                )
                        lines.append("w_mean(uO/uN/uF/syn/red)=" + "/".join([f"{x:.3f}" for x in w_mean.detach().cpu().tolist()]))
                        print("\n" + "\n".join(lines))

                        row = [
                            epoch, global_step, f"{cur_loss:.6f}", f"{float(loss_task.item()):.6f}",
                            f"{float(loss_inter.item()):.6f}", f"{float(loss_balance.item()):.6f}",
                            f"{float(loss_entropy.item()):.6f}", f"{opt.param_groups[0]['lr']:.8e}", f"{syn_sim_b0:.6f}",
                        ] + [f"{v:.6f}" for v in w_mean.detach().cpu().tolist()]
                        w_step(row)
                del cands, cands_a, qB_by_cond, qA_by_cond

            # Epoch Finalization
            sched.step()
            loss_t = torch.tensor([loss_sum, task_sum, inter_sum, syn_sim_sum, n_steps], device=device, dtype=torch.float32)
            loss_t = ddp_all_reduce_mean(loss_t)
            w_epoch = ddp_all_reduce_mean(w_sum / max(1, n_steps))

            epoch_loss = float(loss_t[0].item() / max(1.0, loss_t[4].item()))
            epoch_task = float(loss_t[1].item() / max(1.0, loss_t[4].item()))
            epoch_inter = float(loss_t[2].item() / max(1.0, loss_t[4].item()))
            epoch_syn_sim = float(loss_t[3].item() / max(1.0, loss_t[4].item()))

            if ddp_is_on():
                torch.cuda.empty_cache()

            if rank == 0:
                print(f"[Epoch End] epoch={epoch} time={time.time() - t0:.1f}s lr={opt.param_groups[0]['lr']:.2e} "
                      f"epoch_loss={epoch_loss:.4f} task={epoch_task:.4f} inter={epoch_inter:.4f} syn_sim_b0={epoch_syn_sim:.4f}")

                metric_used = epoch_loss if cfg.best_metric == "loss" else epoch_syn_sim
                row = [
                    epoch, f"{epoch_loss:.6f}", f"{epoch_task:.6f}", f"{epoch_inter:.6f}",
                    f"{opt.param_groups[0]['lr']:.8e}", f"{epoch_syn_sim:.6f}",
                ] + [f"{float(x):.6f}" for x in w_epoch.detach().cpu().tolist()] + [f"{metric_used:.6f}"]
                w_ep(row)

                save_ckpt(os.path.join(run_dir, "ckpt_latest.pt"), model, opt, sched, scaler, epoch, global_step, best_val)
                if (epoch % cfg.save_every_epochs) == 0:
                    save_ckpt(os.path.join(run_dir, f"ckpt_epoch_{epoch:04d}.pt"), model, opt, sched, scaler, epoch, global_step, best_val)

                metric = epoch_loss if cfg.best_metric == "loss" else epoch_syn_sim
                improved = metric < best_val if best_val is not None and cfg.best_mode == "min" else (best_val is None or metric > best_val)

                if improved:
                    best_val = metric
                    save_ckpt(os.path.join(run_dir, "ckpt_best.pt"), model, opt, sched, scaler, epoch, global_step, best_val)
                    print(f"[Best] Updated best={best_val:.6f} at epoch={epoch}")

    finally:
        if rank == 0:
            f_step.close()
            f_ep.close()
        ddp_barrier()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()