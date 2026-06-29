"""
This script defines a 3-modal (O / N / F) representation learning model (ViT version), which can be summarized as:

1) Three 3D-ViT encoders: extract unimodal embeddings from O/N/F input volumes;
2) Five "experts" branches: construct 5 different semantic representations from unimodal embeddings:
   - uO / uN / uF: unique information representations of each modality (directly use the corresponding modality embedding)
   - syn：use learnable syn_token as query, perform cross-attention with the unique information of three modalities
   - red：use learnable red_token as query, perform cross-attention with the unique information of three modalities
3) Projection head: map each expert's emb_dim to proj_dim and perform L2 normalization
4) ReWeighting network: output 5 expert importance weights w for each sample (sample-level gating)
5) Fusion vector q_fused: perform weighted sum in the contrastive space (proj_dim) to get the fused representation
"""

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attn_utils.fa2_utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

from einops import rearrange, repeat
from functools import partial
import math
from src.utils.util import trunc_normal_

from .masks import generate_block_mask, apply_block_mask, masked_mean_pool, process_brain_mask
from .pos_emb import SineCosine3D_PosEmbed

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.dropout = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None, output_attentions=False):
        if (attention_mask is not None) and (attention_mask.dim()==2):
            from src.attn_utils.expand_mask import _expand_mask
            # Expands attention_mask from [bsz, seq_len] to [bsz, 1, seq_len, seq_len]
            attention_mask = _expand_mask(attention_mask, x.dtype)
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        src_len = k.size(2)

        if attn.size() != (B, self.num_heads, N, src_len):
            raise ValueError(
                f"Attention weights should be of size {(B, self.num_heads, N, src_len)}, but is"
                f" {attn.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (B, 1, N, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(B, 1, N, src_len)}, but is {attention_mask.size()}"
                )
            attn = attn + attention_mask
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class CrossAttention(nn.Module):
    """
    Cross-attention:
      - q:  [B, Nq, C]
      - kv: [B, Nk, C]
    returns:
      - out:  [B, Nq, C]
      - attn: [B, heads, Nq, Nk]
    """
    def __init__(self, dim, num_heads=8, q_bias=False, kv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q  = nn.Linear(dim, dim, bias=q_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=kv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, attention_mask=None, output_attentions=True):
        """
        attention_mask: optional, shape should be [B, 1, Nq, Nk] (add to attn logits, masked positions use -inf)
        """
        B, Nq, C = q.shape
        B2, Nk, C2 = kv.shape
        assert B == B2 and C == C2, "q and kv must have same batch and dim"

        # Q: [B, heads, Nq, head_dim]
        q = self.to_q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # K,V: [B, heads, Nk, head_dim]
        kv = self.to_kv(kv).reshape(B, Nk, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # attn logits: [B, heads, Nq, Nk]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            if attention_mask.shape != (B, 1, Nq, Nk):
                raise ValueError(f"attention_mask should be {(B, 1, Nq, Nk)}, but got {attention_mask.shape}")
            attn = attn + attention_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if output_attentions:
            return out, attn
        return out, None


class FlashAttention2(Attention):
    """
    CLIPAttention flash attention module. This module inherits from `CLIPAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output_attentions = False

        # batch_size, q_len, _ = hidden_states.size()
        
        B, N, C = hidden_states.shape

        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        query_states, key_states, value_states = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        # query_states = query_states.view(B, N, self.num_heads, self.head_dim)
        # key_states = key_states.view(b, N, self.num_heads, self.head_dim)
        # value_states = value_states.view(B, N, self.num_heads, self.head_dim)

        dropout_rate = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # if (attention_mask == 1).all():
        #     attention_mask = None # save time for ukb data
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            N,
            dropout=dropout_rate,
            is_causal=causal_attention_mask is not None,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(B, N, C).contiguous()

        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


CLIP_ATTENTION_CLASSES = {
    "normal": Attention,
    "flash_attn": FlashAttention2,
}

class PatchEmbed3D(nn.Module):
    """
    3D patchify + linear projection
    """
    def __init__(self, in_chans: int = 1, embed_dim: int = 256, patch_size: Tuple[int, int, int] = (8, 10, 8)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W] -> [B, E, D', H', W'] -> [B, N, E]
        x = self.proj(x)
        B, E, Dp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, 
                p: float = 0.1, act_layer: nn.Module = nn.GELU):
        super().__init__()
        
        layers: List[nn.Module] = []
        layers += [nn.Linear(in_dim, hidden), act_layer(), nn.Dropout(p=p)]  
        layers += [nn.Linear(hidden, out_dim), nn.Dropout(p=p)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, 
                p: float = 0.1, num_layers: int = 2, 
                act_layer: nn.Module = nn.ReLU):
        super().__init__()

        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(max(1, num_layers - 1)):
            layers += [nn.Linear(d, hidden), act_layer(inplace=True), nn.Dropout(p=p)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer Block: MHA + MLP + residual + LayerNorm
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_mode='normal'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CLIP_ATTENTION_CLASSES[attn_mode](dim, num_heads=num_heads, 
                                                      qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                                      attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TransformerMLP(in_dim=dim, out_dim=dim, hidden=mlp_hidden_dim,
                                  p=drop, act_layer=act_layer)

    def forward(self, x, attention_mask=None, return_attention=False):

        y, attn = self.attn(self.norm1(x), attention_mask=attention_mask, output_attentions=return_attention)

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class ProjectionHead(nn.Module):
    """2-layer MLP for projection"""
    def __init__(self, in_dim: int = 256, proj_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=-1)


class ExpertCrossAttention(nn.Module):
    """
    Construct a learnable token as query, then compute synergy and redundant embeddings from the unique information of three modalities via cross-attention.
    """
    def __init__(self, emb_dim: int = 256, num_heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()

        self.query_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.attn = CrossAttention(emb_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm = nn.LayerNorm(emb_dim)

        nn.init.normal_(self.query_token, std=0.02)

    def forward(self, h_o: torch.Tensor = None,
                      h_n: torch.Tensor = None,
                      h_f: torch.Tensor = None,
                      x: torch.Tensor = None) -> torch.Tensor:
        
        if x is None:
            assert h_o is not None and h_n is not None and h_f is not None, "x is None, but h_o, h_n, h_f are all None"
            x = torch.stack([h_o, h_n, h_f], dim=1)  # [B, 3, emb_dim]
        else:
            assert x.dim() == 3 and x.size(1) == 3, "x must be [B, 3, emb_dim]."
        
        B, _, E = x.shape
        q = self.query_token.expand(B, 1, E)  # [B, 1, E]
        attn_out, attn_w = self.attn(q, x, output_attentions=True)
        out = self.norm(q + attn_out)
        return out.squeeze(1)


class Experts(nn.Module):
    """
    5 experts: uO,uN,uF,syn,red
      - uO/uN/uF: unimodal uniqueness expert
      - syn: learnable syn_token + cross-attention 
      - red: learnable red_token + cross-attention
    Output:
      z_dict: unprojected representations (emb_dim)
      q_dict: projected representations (proj_dim, L2 normalized)
    """
    def __init__(self, emb_dim: int = 256, proj_dim: int = 128, attn_heads: int = 16):
        super().__init__()
        self.emb_dim = emb_dim
        self.names = ["uO", "uN", "uF", "syn", "red"]

        self.syn_attn = ExpertCrossAttention(emb_dim, attn_heads)
        self.red_attn = ExpertCrossAttention(emb_dim, attn_heads)

        self.  proj = nn.ModuleDict({k: ProjectionHead(emb_dim, proj_dim) for k in self.names})

    def forward_z(self, hO: torch.Tensor, hN: torch.Tensor, hF: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = torch.stack([hO, hN, hF], dim=1)  # [B, 3, emb_dim]

        return {
            "uO": hO,
            "uN": hN,
            "uF": hF,
            "syn": self.syn_attn(x=tokens),
            "red": self.red_attn(x=tokens),
        }
    
    def forward_q(self, hO, hN, hF):
        """return dict[name] = projected & L2-normalized vector"""
        if hasattr(self, "forward_z"):
            z = self.forward_z(hO, hN, hF)
        else:
            z = {
                "uO": hO,
                "uN": hN,
                "uF": hF,
                "syn": self.syn_attn(hO, hN, hF),
                "red": self.red_attn(hO, hN, hF),
            }
    
        return {k: self.proj[k](z[k]) for k in z.keys()}

    def forward(self, hO: torch.Tensor, hN: torch.Tensor, hF: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        z = self.forward_z(hO, hN, hF)
        q = {k: self.proj[k](z[k]) for k in z.keys()}
        return z, q


class ReWeightingMLP(nn.Module):
    """
    Re-weighting model
    Output: sample-level importance weights for each expert
    """
    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        hidden: int = 256,
        num_layers: int = 2,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.mlp = MLP(in_dim, num_experts, hidden=hidden, p=dropout, num_layers=num_layers)

    def forward(self, x: torch.Tensor, detach_in: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, in_dim]
        if detach_in:
            x = x.detach()
        logits = self.mlp(x)                              # [B, E]
        w = F.softmax(logits / self.temperature, dim=-1)  # [B, E]
        return w, logits


class ViT3DEncoder(nn.Module):
    """
    3D ViT Encoder:
    - patchify + pos embed
    - Transformer Blocks 
    - CLS token or mean pooling
    """
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (64, 80, 64),
        in_chans: int = 1,
        patch_size: Tuple[int, int, int] = (8, 10, 8),
        embed_dim: int = 256,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_mode: str = 'normal',
        add_w: bool = False,
        gradient_checkpointing: bool = False,
        use_cls_token: bool = False,
        init_std: float = 0.02,
        dropout: float = 0.0,
        view_size: Optional[Tuple[int, int, int]] = None,
        use_brainmask: bool = False,
    ):
        super().__init__()

        if any(s % p != 0 for s, p in zip(img_size, patch_size)):
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")

        self.init_std = init_std
        self.patch_size = patch_size
        self.img_size = img_size
        self.view_size = view_size
        self.embed_dim = embed_dim
        self.use_cls_token = bool(use_cls_token)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        if view_size is not None:
            assert view_size[0]%patch_size[0]==0 and view_size[1]%patch_size[1]==0 and view_size[2]%patch_size[2]==0, "view_size must be divisible by patch_size"

        # patch embed
        self.patch_embed = PatchEmbed3D(in_chans, embed_dim, patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        grid_3Dsize = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1], self.img_size[2] // self.patch_size[2])

        # cls token, pos enc, pos drop
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.use_cls_token else None
        self.pos_embed = SineCosine3D_PosEmbed(grid_3Dsize, embed_dim, cls_token=self.use_cls_token)
        self.pos_drop = nn.Dropout(p=dropout)  
        self.norm = norm_layer(embed_dim)

        self.use_brainmask = use_brainmask
        self.brainmask = None

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                attn_mode=attn_mode  # 'normal' or 'flash_attn'
                )
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # init weights
        self.apply(self._init_weights)
        self.fix_init_weight()
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=self.init_std)

    def fix_init_weight(self): 
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if hasattr(layer.attn, "proj"):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            if hasattr(layer.mlp, "net"):
                for m in reversed(layer.mlp.net):
                    if isinstance(m, nn.Linear):
                        rescale(m.weight.data, layer_id + 1)
                        break

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, masks: torch.Tensor = None, indices: torch.Tensor = None, 
                        seed: int = None, return_attention: bool = False,
                        drop_token: bool = False) -> torch.Tensor:

        pos_embed = self.pos_embed()[0].to(x.device)
        if self.use_brainmask:
            brainmask = process_brain_mask(target_img_size=self.img_size, patch_size=self.patch_size)
            brainmask = brainmask.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
            self.brainmask = brainmask.flatten(1).to(x.device)  # [B, num_patches]

        # patchify
        x = self.patch_embed(x)  # [B, N, E] = [batch_size, num_patches, embed_dim]
        B, N, E = x.shape

        # add cls token
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = B)  # (1,1,E) -> (B,1,E)
            x = torch.cat([cls_tokens, x], dim=1) 
            x = x + pos_embed[:, :N+1, :]  # [B, 1+N, E]
        else:
            x = x + pos_embed[:, :N, :]    # [B, N, E]
        
        # partial view input
        if self.view_size is not None and indices is not None:
            if drop_token:
                x = apply_block_mask(x, keep_idx=indices.squeeze(1))  # [B, K, E], K is the num of kept patches
        
        x = self.pos_drop(x)

        # transformer blocks
        attn_set = []
        for blk in self.blocks:
            if return_attention:
                x, attn = blk(x, attention_mask=self.brainmask, return_attention=return_attention)
                attn_set.append(attn.detach().cpu())
            else:
                x = blk(x, attention_mask=self.brainmask)                         # [B, N+1, E] or [B, N, E]
        if self.norm is not None:
            x = self.norm(x)

        # output embedding
        if self.use_cls_token:
            if return_attention:
                return x[:, 0, :], attn         
            return x[:, 0, :]
        else:
            if (self.view_size is not None) and (masks is not None) and (not drop_token):
                x_pool = masked_mean_pool(x, masks.squeeze(1))   
            else:
                if self.use_brainmask:
                    x_pool = masked_mean_pool(x, self.brainmask)
                else:
                    x_pool = x.mean(dim=1)

            if return_attention:
                return x_pool, attn     
            return x_pool


class FullViTModel(nn.Module):
    """
    Output:
      - hO,hN,hF: unimodal embeddings
      - z_dict: unprojected representations (emb_dim) for 5 experts
      - q_dict: projected representations (proj_dim) for 5 experts
      - w: re-weighting weights (sample-level)
      - q_fused: Σ w_e * q_e (fused representation in the contrastive space)
    """
    def __init__(
        self,
        emb_dim: int = 256,
        proj_dim: int = 128,
        # reweighter params
        rw_hidden: int = 256,
        rw_layers: int = 2,
        rw_temperature: float = 1.0,
        rw_detach_in: bool = True,
        rw_heads: int = 8,  
        # vit params
        img_size: Tuple[int, int, int] = (96, 112, 96),
        patch_size: Tuple[int, int, int] = (8, 10, 8),
        view_size: Optional[Tuple[int, int, int]] = None,
        vit_depth: int = 12, vit_heads: int = 6, vit_mlp_ratio: float = 4.0,
        vit_qkv_bias: bool = True, vit_norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        vit_attn_mode: str = 'normal',  # 'normal' or 'flash_attn'
        use_brainmask: bool = False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.rw_detach_in = bool(rw_detach_in)

        # three 3D-ViT encoders
        self.enc_O = ViT3DEncoder(img_size=img_size, patch_size=patch_size, view_size=view_size, embed_dim=emb_dim, depth=vit_depth, num_heads=vit_heads, 
                                 mlp_ratio=vit_mlp_ratio, qkv_bias=vit_qkv_bias, norm_layer=vit_norm_layer,
                                 attn_mode=vit_attn_mode, use_brainmask=use_brainmask)
        self.enc_N = ViT3DEncoder(img_size=img_size, patch_size=patch_size, view_size=view_size, embed_dim=emb_dim, depth=vit_depth, num_heads=vit_heads, 
                                 mlp_ratio=vit_mlp_ratio, qkv_bias=vit_qkv_bias, norm_layer=vit_norm_layer,
                                 attn_mode=vit_attn_mode, use_brainmask=use_brainmask)
        self.enc_F = ViT3DEncoder(img_size=img_size, patch_size=patch_size, view_size=view_size, embed_dim=emb_dim, depth=vit_depth, num_heads=vit_heads, 
                                 mlp_ratio=vit_mlp_ratio, qkv_bias=vit_qkv_bias, norm_layer=vit_norm_layer,
                                 attn_mode=vit_attn_mode, use_brainmask=use_brainmask)

        # five experts branches
        self.experts = Experts(emb_dim, proj_dim, attn_heads=rw_heads)

        # mask tokens for drop_* conditions (replace random vectors)
        self.mask_O = nn.Parameter(torch.zeros(emb_dim))
        self.mask_N = nn.Parameter(torch.zeros(emb_dim))
        self.mask_F = nn.Parameter(torch.zeros(emb_dim))
        nn.init.normal_(self.mask_O, std=0.02)
        nn.init.normal_(self.mask_N, std=0.02)
        nn.init.normal_(self.mask_F, std=0.02)

        self.reweighter = ReWeightingMLP(
            in_dim=emb_dim * 3,
            num_experts=len(self.experts.names),
            hidden=rw_hidden,
            num_layers=rw_layers,
            temperature=rw_temperature
        )

    def get_mask_batch(self, which: str, B: int) -> torch.Tensor:
        """
        Output:
          - [B, emb_dim] mask embedding (for drop conditions)
        """
        if which == "O":
            m = self.mask_O
        elif which == "N":
            m = self.mask_N
        elif which == "F":
            m = self.mask_F
        else:
            raise ValueError(f"unknown mask type: {which}")
        return m.unsqueeze(0).expand(B, -1)

    def encode(self, O: torch.Tensor, N: torch.Tensor, F: torch.Tensor, 
                    masks: torch.Tensor = None, indices: torch.Tensor = None,
                    drop_token: bool = False
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.enc_O(O, masks=masks, indices=indices, drop_token=drop_token), self.enc_N(N, masks=masks, indices=indices, drop_token=drop_token), self.enc_F(F, masks=masks, indices=indices, drop_token=drop_token)

    def forward(self, O: torch.Tensor, N: torch.Tensor, F: torch.Tensor):

        # modality-specific encoder
        hO, hN, hF = self.encode(O, N, F)

        # experts
        z, q = self.experts.forward(hO, hN, hF)

        # reweighter
        w, logits = self.reweighter(torch.cat([hO, hN, hF], dim=-1), detach_in=self.rw_detach_in)  # [B,5]

        # fusion
        q_fused = 0.0
        for i, name in enumerate(self.experts.names):
            q_fused = q_fused + w[:, i:i+1] * q[name]  # [B,proj_dim]
        q_fused = F.normalize(q_fused, p=2, dim=-1)

        return (hO, hN, hF), z, q, w, q_fused, logits


class PredictionHead(nn.Module):
    """MLP-based prediction head for downstream tasks."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pred_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred_head(x)

class PredictionHead_LayerNorm(nn.Module):
    """
    Improved Prediction Head:
    1. Added LayerNorm at the beginning (Crucial for fused features).
    2. Kept 2-layer structure (Safe for medical data).
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pred_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred_head(x)


class FullViTModel_Predict(nn.Module):
    """
    Downstream prediction model with:
      - three 3D-ViT encoders (same as pretrain)
      - reweighter (same as pretrain)
      - per-encoder prediction heads -> logits
      - weighted fusion of logits by reweighter weights
    """
    def __init__(
        self,
        emb_dim: int = 256,
        proj_dim: int = 128,
        # prediction params
        project_heads: bool = True,        # whether to use project_heads
        single_pred_head: bool = False,    # whether to use single prediction head for fused features
        out_dim: int = 1,                
        pred_dim: int = 256,
        pred_dropout: float = 0.1,
        # reweighter params
        rw_hidden: int = 256,
        rw_layers: int = 2,
        rw_temperature: float = 1.0,
        rw_detach_in: bool = True,
        rw_heads: int = 8,
        # vit params
        img_size: Tuple[int, int, int] = (96, 112, 96),
        patch_size: Tuple[int, int, int] = (8, 10, 8),
        view_size: Optional[Tuple[int, int, int]] = None,
        vit_depth: int = 12,
        vit_heads: int = 6,
        vit_mlp_ratio: float = 4.0,
        vit_qkv_bias: bool = True,
        vit_norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        vit_attn_mode: str = "normal",
        use_brainmask: bool = False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.pred_dim = pred_dim
        self.out_dim = out_dim
        self.rw_detach_in = bool(rw_detach_in)
        self.project_heads = bool(project_heads)
        self.use_single_pred_head = bool(single_pred_head)

        # 3D ViT encoders (same as pretrain)
        self.enc_O = ViT3DEncoder(img_size=img_size, patch_size=patch_size, view_size=view_size, embed_dim=emb_dim, depth=vit_depth, num_heads=vit_heads, 
                                 mlp_ratio=vit_mlp_ratio, qkv_bias=vit_qkv_bias, norm_layer=vit_norm_layer,
                                 attn_mode=vit_attn_mode, use_brainmask=use_brainmask)
        self.enc_N = ViT3DEncoder(img_size=img_size, patch_size=patch_size, view_size=view_size, embed_dim=emb_dim, depth=vit_depth, num_heads=vit_heads, 
                                 mlp_ratio=vit_mlp_ratio, qkv_bias=vit_qkv_bias, norm_layer=vit_norm_layer,
                                 attn_mode=vit_attn_mode, use_brainmask=use_brainmask)
        self.enc_F = ViT3DEncoder(img_size=img_size, patch_size=patch_size, view_size=view_size, embed_dim=emb_dim, depth=vit_depth, num_heads=vit_heads, 
                                 mlp_ratio=vit_mlp_ratio, qkv_bias=vit_qkv_bias, norm_layer=vit_norm_layer,
                                 attn_mode=vit_attn_mode, use_brainmask=use_brainmask)

        # experts (same as pretrain)
        self.experts = Experts(emb_dim, proj_dim, attn_heads=rw_heads)

        # prediction heads
        pred_in_dim = proj_dim if self.project_heads else emb_dim

        if self.use_single_pred_head:
            # one prediction head for fused features
            self.single_pred_head = PredictionHead_LayerNorm(pred_in_dim, out_dim, hidden=pred_dim, dropout=pred_dropout)
        else:
            self.pred_heads = nn.ModuleDict({
                name: PredictionHead(pred_in_dim, out_dim, hidden=pred_dim, dropout=pred_dropout)
                for name in self.experts.names
            })   # one per expert

        # reweighter (same as pretrain)
        self.reweighter = ReWeightingMLP(
            in_dim=emb_dim * 3,
            num_experts=len(self.experts.names),  # only three modality heads to weight
            hidden=rw_hidden,
            num_layers=rw_layers,
            temperature=rw_temperature,
        )

    def encode(
        self,
        O: torch.Tensor,
        N: torch.Tensor,
        F: torch.Tensor,
        masks: torch.Tensor = None,
        indices: torch.Tensor = None,
        drop_token: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.enc_O(O, masks=masks, indices=indices, drop_token=drop_token),
            self.enc_N(N, masks=masks, indices=indices, drop_token=drop_token),
            self.enc_F(F, masks=masks, indices=indices, drop_token=drop_token),
        )

    def forward(self, O: torch.Tensor, N: torch.Tensor, F: torch.Tensor,
                w_used: torch.Tensor = None, expert_mask: torch.Tensor = None):
        """
        expert_mask: optional, shape [num_experts] float/bool tensor.
                     value 1 means keep this expert, 0 means exclude.
                     only effective when w_used is None (applied to the weights output by reweighter).
                     used for ablation experiments: exclude specified expert and re-normalize the remaining weights.
        """
        # Encoder -> Embeddings: [B, 1, H, W, D] -> [B, emb_dim]
        hO, hN, hF = self.encode(O, N, F)

        # Experts -> Projections
        emb, emb_proj = self.experts(hO, hN, hF)   # emb = expert(h), emb_proj = mlp(emb)

        # Prediction Heads -> Logits [B, out_dim]
        logits_by_expert = {}
        for name in self.experts.names:

            if self.use_single_pred_head:
                if self.project_heads:
                    logits_by_expert[name] = emb_proj[name]
                else:
                    logits_by_expert[name] = emb[name]
            else:
                if self.project_heads:
                    logits_by_expert[name] = self.pred_heads[name](emb_proj[name])
                else:
                    logits_by_expert[name] = self.pred_heads[name](emb[name])

        # Compute Weights (w) [B, 5]
        rw_weights, rw_logits = self.reweighter(
            torch.cat([hO, hN, hF], dim=-1),
            detach_in=self.rw_detach_in,
        )

        # Ablation: apply expert mask to the weights output by reweighter and re-normalize the remaining weights
        if expert_mask is not None:
            mask = expert_mask.to(rw_weights.device).float()   # [num_experts]
            rw_weights = rw_weights * mask.unsqueeze(0)        # [B, num_experts]
            rw_weights = rw_weights / (rw_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Fuse Predictions by weights (w)
        fused_logit = 0.0
        if w_used is None:
            for i, name in enumerate(self.experts.names):
                fused_logit = fused_logit + rw_weights[:, i:i+1] * logits_by_expert[name]
        else:
            for i, name in enumerate(self.experts.names):
                fused_logit = fused_logit + w_used[:, i:i+1] * logits_by_expert[name]
        
        if self.use_single_pred_head:
            fused_logit = self.single_pred_head(fused_logit)
        
        return {
            "logits": fused_logit,               # [B, out_dim]
            "logits_modal": logits_by_expert,    # {name: [B, out_dim]}
            "rw_weights": rw_weights,            # [B, 5]
            "rw_logits": rw_logits,              # [B, 5]
            "exp_embed": emb,                    # {name: [B, emb_dim]}
            "exp_embed_proj": emb_proj,          # {name: [B, proj_dim]}
            "embeddings": (hO, hN, hF),          # [B, emb_dim]
        }