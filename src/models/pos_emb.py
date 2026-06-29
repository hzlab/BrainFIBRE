import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import pdb


class PosEmbed_Base(nn.Module):
    def __init__(self, grid_size, embed_dim, device=None, 
                        use_pos_embed_decoder=False,
                        predictor_embed_dim=None):
        super().__init__()

        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.device = device
        self.use_pos_embed_decoder = use_pos_embed_decoder
        
        self.repeat_time = self.grid_size[1]
        
        self.grid = self.get_grid(grid_size)

        self.emb_h_encoder = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1], embed_dim // 2), requires_grad=False)

        pos_emb_h_encoder = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, self.grid[0])  # (H*W, E/2)

        self.emb_h_encoder.data.copy_(torch.from_numpy(pos_emb_h_encoder).float())
        
        if self.use_pos_embed_decoder and predictor_embed_dim is not None:
            self.emb_h_decoder = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1], predictor_embed_dim // 2), requires_grad=False)
            pos_emb_h_decoder = self.get_1d_sincos_pos_embed_from_grid(predictor_embed_dim // 2, self.grid[0])  # (H*W, E/2)
            self.emb_h_decoder.data.copy_(torch.from_numpy(pos_emb_h_decoder).float())
    
    def get_grid(self, grid_size):    
        grid_h = np.arange(grid_size[0], dtype=float)  
        grid_w = np.arange(grid_size[1], dtype=float) 
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)                            
        grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])  
        return grid        
    
    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega   

        pos = pos.reshape(-1)   
        out = np.einsum('m,d->md', pos, omega)   

        emb_sin = np.sin(out) 
        emb_cos = np.cos(out) 

        emb = np.concatenate([emb_sin, emb_cos], axis=1) 
        return emb


class SineCosine3D_PosEmbed(PosEmbed_Base):
    def __init__(self, grid_size, embed_dim, device=None, use_pos_embed_decoder=False, predictor_embed_dim=None, cls_token=None):
        super().__init__(grid_size, embed_dim, device, use_pos_embed_decoder, predictor_embed_dim)
        
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.device = device
        self.use_pos_embed_decoder = use_pos_embed_decoder
        self.cls_token = cls_token
        
        d = embed_dim // 3 // 2 * 2 # make sure divisible by 2
        self.d_pad = embed_dim - d * 3 if embed_dim % 3 != 0 else None # padding if embed_dim is not divisible by 3
        self.emb_h_encoder = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1]*grid_size[2], d), requires_grad=False)
        self.emb_w_encoder = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1]*grid_size[2], d), requires_grad=False)
        self.emb_d_encoder = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1]*grid_size[2], d), requires_grad=False)

        if self.d_pad is not None:
            self.register_buffer("pos_pad", torch.zeros(grid_size[0]*grid_size[1]*grid_size[2], self.d_pad), persistent=False) 
        
        grid = self.get_3d_grid(grid_size)  # [3, H*W*D]
        
        pos_emb_h_encoder = self.get_1d_sincos_pos_embed_from_grid(d, grid[0])  # (H*W*D, E/3)
        pos_emb_w_encoder = self.get_1d_sincos_pos_embed_from_grid(d, grid[1])  # (H*W*D, E/3)
        pos_emb_d_encoder = self.get_1d_sincos_pos_embed_from_grid(d, grid[2])  # (H*W*D, E/3)
        
        self.emb_h_encoder.data.copy_(torch.from_numpy(pos_emb_h_encoder).float())
        self.emb_w_encoder.data.copy_(torch.from_numpy(pos_emb_w_encoder).float())
        self.emb_d_encoder.data.copy_(torch.from_numpy(pos_emb_d_encoder).float())
        
        if self.use_pos_embed_decoder and predictor_embed_dim is not None:
            
            self.emb_h_decoder = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1]*grid_size[2], predictor_embed_dim // 3), requires_grad=False)
            self.emb_w_decoder = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1]*grid_size[2], predictor_embed_dim // 3), requires_grad=False)
            self.emb_d_decoder = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1]*grid_size[2], predictor_embed_dim // 3), requires_grad=False)
            
            pos_emb_h_decoder = self.get_1d_sincos_pos_embed_from_grid(predictor_embed_dim // 3, grid[0])
            pos_emb_w_decoder = self.get_1d_sincos_pos_embed_from_grid(predictor_embed_dim // 3, grid[1])
            pos_emb_d_decoder = self.get_1d_sincos_pos_embed_from_grid(predictor_embed_dim // 3, grid[2])
            
            self.emb_h_decoder.data.copy_(torch.from_numpy(pos_emb_h_decoder).float())
            self.emb_w_decoder.data.copy_(torch.from_numpy(pos_emb_w_decoder).float())
            self.emb_d_decoder.data.copy_(torch.from_numpy(pos_emb_d_decoder).float())
            
    def get_3d_grid(self, grid_size):
        """Generate a 3D grid.
            e.g. input grid_size: tuple (H, W, D) = (6, 7, 6)

            return grid: numpy array in shape [3, H*W*D], wherer H*W*D is the total number of tokens (positions)
        """
        grid_h = np.arange(grid_size[0], dtype=float)  
        grid_w = np.arange(grid_size[1], dtype=float)  
        grid_d = np.arange(grid_size[2], dtype=float)  
        
        grid = np.meshgrid(grid_w, grid_h, grid_d)  # w, h, d order
        grid = np.stack(grid, axis=0) 
        grid = grid.reshape([3, -1])  
        return grid
    
    def forward(self):
        # Concatenate embeddings for encoder
        if self.d_pad is not None:
            emb_encoder = torch.cat([self.emb_h_encoder, 
                                    self.emb_w_encoder, 
                                    self.emb_d_encoder,
                                    self.pos_pad
                                    ], dim=1).unsqueeze(0) 
        else: 
            emb_encoder = torch.cat([self.emb_h_encoder, self.emb_w_encoder, self.emb_d_encoder], dim=1).unsqueeze(0)
        
        if self.cls_token:
            pos_embed_encoder = torch.concat([torch.zeros([1, 1, emb_encoder.shape[2]]), emb_encoder], dim=1)
        else:
            pos_embed_encoder = emb_encoder
            
        if self.use_pos_embed_decoder:
            emb_decoder = torch.cat([self.emb_h_decoder, self.emb_w_decoder, self.emb_d_decoder], dim=1).unsqueeze(0)
            if self.cls_token:
                pos_embed_decoder = torch.concat([torch.zeros([1, 1, emb_decoder.shape[2]]), emb_decoder], dim=1)
            else:
                pos_embed_decoder = emb_decoder
            return pos_embed_encoder, pos_embed_decoder
            
        return pos_embed_encoder, None