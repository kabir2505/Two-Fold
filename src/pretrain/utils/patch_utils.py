# Copyright ...
from __future__ import annotations
import torch
from torch import nn

def rot_ops_3d():
    # Each op expects (B, C, d, h, w)
    def rot_x(x): return x.transpose(2, 4).flip(2)   # D<->W, flip D
    def rot_y(x): return x.transpose(2, 3).flip(2)   # D<->H, flip D
    def rot_z(x): return x.transpose(3, 4).flip(3)   # H<->W, flip H
    return [
        lambda x: x,                # 0 (identity)
        rot_x,                      # 1
        rot_y,                      # 2
        rot_z,                      # 3
        lambda x: rot_x(rot_y(x)),  # 4
        lambda x: rot_x(rot_z(x)),  # 5
        lambda x: rot_y(rot_z(x)),  # 6
        lambda x: rot_y(rot_x(x)),  # 7
        lambda x: rot_z(rot_x(x)),  # 8
    ]

def split_patches(x: torch.Tensor, n: int):
    # x: (B,C,D,H,W) -> (B,P,C,d,h,w)
    B,C,D,H,W = x.shape
    assert D% n==0 and H% n==0 and W% n==0, "Dims must be multiples of n."
    d,h,w = D//n, H//n, W//n
    xp = x.unfold(2,d,d).unfold(3,h,h).unfold(4,w,w)  # (B,C,n,n,n,d,h,w)
    xp = xp.contiguous().view(B, C, n*n*n, d,h,w).permute(0,2,1,3,4,5)
    return xp, (d,h,w)

def combine_patches(xt: torch.Tensor, n: int, d: int, h: int, w: int):
    # xt: (B,P,C,d,h,w) -> (B,C,D,H,W)
    B,P,C,_,_,_ = xt.shape
    pt = xt.view(B, n,n,n, C, d,h,w).permute(0,4,1,5,2,6,3,7).contiguous()
    return pt.view(B, C, n*d, n*h, n*w)

@torch.no_grad()
def apply_twofold(x: torch.Tensor, n: int, p_mask: float = 0.5, K: int = 9):
    """
    Two-fold corruption:
      1) Bernoulli mask M over P=n^3 patches.
      2) Discrete rotations for masked patches only.
    Returns: x_tilde (B,C,D,H,W), M (B,P) float, R (B,P) long in [0..K-1], where R=0 for unmasked.
    """
    assert x.dim() == 5, "Expected (B,C,D,H,W)"
    OPS = rot_ops_3d() if K==9 else rot_ops_3d()  # extend to 24 if you add them
    patches, (d,h,w) = split_patches(x, n)        # (B,P,C,d,h,w)
    B,P = patches.shape[0], patches.shape[1]

    M = (torch.rand(B, P, device=x.device) < float(p_mask)).float()
    R = torch.zeros(B, P, dtype=torch.long, device=x.device)
    if K>1:
        R_mask = torch.randint(1, min(K, len(OPS)), (B,P), device=x.device)
        R = torch.where(M>0, R_mask, R)

    # Vectorized apply per rotation id
    out = patches.clone()
    for k in range(1, min(K, len(OPS))):
        sel = (R == k) & (M > 0)
        if sel.any():
            # gather and apply
            idx_b, idx_p = sel.nonzero(as_tuple=True)
            psel = out[idx_b, idx_p]  # (N,C,d,h,w)
            psel = OPS[k](psel)
            out[idx_b, idx_p] = psel

    x_tilde = combine_patches(out, n, d,h,w)
    return x_tilde, M, R
