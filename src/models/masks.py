import torch
from typing import Tuple, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import SpatialPad


def process_brain_mask(nii_path='/path/to/brain_mask.nii.gz', 
            target_img_size=(96, 112, 96), 
            patch_size=(16, 16, 16),
            downsample=True):
    """
    Process DTI mask file: binarization -> padding -> patch downsampling -> batching & flattening
    """
    
    # Load and Binarize
    img = nib.load(nii_path)
    data = img.get_fdata()
    binary_mask = (data > 0.5).astype(np.float32)
        
    # Convert to PyTorch Tensor (H, W, D) -> (Channel, H, W, D) -> (1, 91, 109, 91)
    mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)
    
    # Spatial Padding
    padder = SpatialPad(spatial_size=target_img_size)
    padded_mask = padder(mask_tensor) 
    
    # Patch Downsampling via MaxPool
    if downsample:
        downsampled_grid = F.max_pool3d(
            padded_mask, 
            kernel_size=patch_size, 
            stride=patch_size
        )  # (1, 96, 112, 96) -> (1, 6, 7, 6)
        grid_volume = downsampled_grid.squeeze(0) # (1, 6, 7, 6) -> (6, 7, 6)
    
    else:
        grid_volume = padded_mask.squeeze(0)

    # Batch Expansion and Flattening
    # batch_volume = grid_volume.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # [B, 6*7*6] -> [B, 252]
    # flattened_output = batch_volume.view(batch_size, -1)
    
    return grid_volume


def generate_block_mask(
        grid_size: Tuple[int, int, int],     
        view_size: Tuple[int, int, int],      
        seed: int = False,
        B: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回：
      - mask_flat: [B, N]，N = gd*gh*gw，binary mask，1=visible, 0=invisible
      - keep_idx:  [B, K]，visible token indices
      - start_idx: [B, 3]，start indices of each sample (sd, sh, sw)
    """
    gd, gh, gw = grid_size
    vd, vh, vw = view_size
    assert vd <= gd and vh <= gh and vw <= gw, "view_size must fit within grid_size"

    g = torch.Generator()
    g.manual_seed(seed)

    max_sd = gd - vd
    max_sh = gh - vh
    max_sw = gw - vw

    sd = torch.randint(0, max_sd + 1, (1,), generator=g).item()
    sh = torch.randint(0, max_sh + 1, (1,), generator=g).item()
    sw = torch.randint(0, max_sw + 1, (1,), generator=g).item()

    # Define a binary mask, mask the selected block region
    if B is not None:
        start_idx = torch.tensor([[sd, sh, sw]]).repeat(B, 1)  # [B,3]

        mask_3d = torch.zeros((B, gd, gh, gw), dtype=torch.float32)    # [B, gd, gh, gw]
        for b in range(B):
            sd_b, sh_b, sw_b = start_idx[b].tolist()
            mask_3d[b, sd_b:sd_b+vd, sh_b:sh_b+vh, sw_b:sw_b+vw] = 1.0
    else:
        start_idx = torch.tensor([[sd, sh, sw]])      # [1,3]
        mask_3d = torch.zeros((1, gd, gh, gw), dtype=torch.float32)    # [1, gd, gh, gw]
        mask_3d[:, sd:sd+vd, sh:sh+vh, sw:sw+vw] = 1.0

    mask_flat = mask_3d.flatten(1)  # [B, gd, gh, gw] -> [B, N]

    # Save token indices of the masked region
    keep_idx = mask_flat.nonzero(as_tuple=False)  # [B*K, 2] -> keep_idx[:,0] is batch id, keep_idx[:,1] is token id
    if B is not None:
        keep_idx = keep_idx[:, 1].view(B, -1).to(torch.long)  # [B, K]
    else:
        keep_idx = keep_idx[:, 1].view(1, -1).to(torch.long)
        
    return mask_flat, keep_idx, start_idx


def apply_block_mask(
    x: torch.Tensor,                     # [B, N, E] or [B, N+1, E]
    mask_flat: torch.Tensor = None,      # [B, N], 1=visible,0=invisible
    keep_idx: torch.Tensor = None,       # [B, K]，token indicies
    cls_token: bool = False,
    mode: str = "zero",           # "zero" or "mask_token"
) -> torch.Tensor:
    """
    Apply mask to patch embedding:
    - If mask_flat is provided: mask the unused tokens to zeros
    - If keep_idx is provided: directly remove the unused tokens
    """
    B, N, E = x.shape

    assert mask_flat == None or keep_idx == None, "pass either mask_flat or keep_idx"

    if mask_flat is not None:
        assert mask_flat.shape == (B, N), "mask_flat should be in shape of [B,N]"
    
    if keep_idx is not None:
        assert keep_idx.dim() == 2 and keep_idx.size(0) == B, "keep_idx should be in shape of [B,K]"

    ## approach 1: mask unused tokens to zeros
    if mask_flat is not None:
        if cls_token:
            cls_t, patches = x[:, :1, :], x[:, 1:, :]  
            assert mask_flat.shape[1] == patches.size(1)
            m = mask_flat.unsqueeze(-1)  
            return torch.cat([cls_t, patches], dim=1)
        else:
            assert mask_flat.shape[1] == x.size(1)
            m = mask_flat.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            return x * m  

    ## approach 2: drop unused tokens
    if keep_idx is not None:
        if cls_token:
            cls_t, patches = x[:, :1, :], x[:, 1:, :]            
            N = patches.size(1)
            assert keep_idx.max().item() < N, "keep_idx out of range for patch tokens"
            index = keep_idx.unsqueeze(-1).repeat(1, 1, E)                 
            kept = torch.gather(patches, dim=1, index=index)              
            return torch.cat([cls_t, kept], dim=1)                        
        else:
            N = x.size(1)
            assert keep_idx.max().item() < N, "keep_idx out of range"
            index = keep_idx.unsqueeze(-1).repeat(1, 1, E)                
            return torch.gather(x, dim=1, index=index)                     





    # if mode == "mask_token":
    #     assert mask_token is not None, "mask_token must be provided for mode='mask_token'"
    #     if mask_token.ndim == 1:
    #         mask_token = mask_token.view(1, 1, -1)
    #     mask_token = mask_token.to(x.device).to(x.dtype)
    #     return x * m + mask_token * (1.0 - m)

    # raise ValueError(f"Unknown mode: {mode}")



def masked_mean_pool(x_tokens: torch.Tensor, mask_flat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x_tokens: [B, N, E]  (patch tokens only)
    mask_flat: [B, N]    (1=keep, 0=drop)
    return: [B, E]
    """
    m = mask_flat.to(dtype=x_tokens.dtype, device=x_tokens.device).unsqueeze(-1)  # [B, N, 1]

    # breakpoint()
    x_sum = (x_tokens * m).sum(dim=1)           # [B, E]
    denom = m.sum(dim=1).clamp_min(eps)         # [B, 1] -> 该批次中每个样本中“可见”的token数量
    return x_sum / denom
