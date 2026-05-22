# losses_twofold.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(residual: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    # residual = x_hat - x
    return torch.sqrt(residual * residual + eps * eps)

def _safe_mean(x: torch.Tensor, denom: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x.sum() / (denom.sum().clamp_min(eps)))

def _voxelize_patch_mask(
    M: torch.Tensor,                 # (B, P)
    vol_shape: tuple[int,int,int],   # (D, H, W)
    grid_shape: tuple[int,int,int]   # (nx, ny, nz) with P = nx*ny*nz
) -> torch.Tensor:
    """
    Voxel-accurate mask: reshape (B,P) -> (B,1,nx,ny,nz), upsample NN -> (B,1,D,H,W).
    """
    B, P = M.shape
    nx, ny, nz = grid_shape
    D, H, W = vol_shape
    assert P == nx * ny * nz, f"Mask P={P} must equal nx*ny*nz={nx*ny*nz}"
    m_grid = M.float().view(B, 1, nx, ny, nz)  # (B,1,nx,ny,nz)
    m_vox = F.interpolate(m_grid, size=(D, H, W), mode="nearest")  # (B,1,D,H,W)
    return m_vox

def _maybe_voxel_mask_or_scalar(
    M: torch.Tensor | None,
    x_like: torch.Tensor,                     # (B,C,D,H,W)
    grid_shape: tuple[int,int,int] | None
) -> torch.Tensor:
    """
    If M and grid_shape are provided, return voxel-accurate (B,1,D,H,W).
    Else return a per-volume scalar broadcast (B,1,D,H,W) as a safe fallback.
    """
    B, C, D, H, W = x_like.shape
    if (M is not None) and (grid_shape is not None):
        return _voxelize_patch_mask(M, (D, H, W), grid_shape)
    # Fallback: broadcast mean(M) or zeros if M is None.
    if M is None:
        m_scalar = torch.zeros(B, 1, device=x_like.device, dtype=x_like.dtype)
    else:
        m_scalar = M.float().mean(dim=1, keepdim=True).to(x_like.dtype)  # (B,1)
    return m_scalar.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)

# -------------------------
# Reconstruction losses
# -------------------------

def recon_loss(
    x_hat: torch.Tensor,            # (B,C,D,H,W)
    x: torch.Tensor,                # (B,C,D,H,W)
    M: torch.Tensor | None = None,  # (B,P), optional
    grid_shape: tuple[int,int,int] | None = None,  # (nx,ny,nz) if using masked variants
    variant: str = "full_charbonnier",  # {"full_mae","full_charbonnier","masked_only_charbonnier","weighted_charbonnier"}
    mask_weight: float = 5.0,
    unmasked_weight: float = 1.0,
    eps_charb: float = 1e-3,
) -> torch.Tensor:
    assert x_hat.shape == x.shape and x.dim() == 5, "x_hat and x must be (B,C,D,H,W)"
    residual = x_hat - x

    if variant == "full_mae":
        return residual.abs().mean()

    if variant == "full_charbonnier":
        return charbonnier(residual, eps=eps_charb).mean()

    # Variants requiring voxel mask
    m_vox = _maybe_voxel_mask_or_scalar(M, x_like=x, grid_shape=grid_shape)  # (B,1,D,H,W)

    if variant == "masked_only_charbonnier":
        w = m_vox  # 1 on masked voxels, 0 elsewhere
        charb = charbonnier(residual, eps=eps_charb)
        return _safe_mean(charb * w, w)

    if variant == "weighted_charbonnier":
        #print("using weighted charbonnier as a loss function, with default masked", mask_weight)
        w = mask_weight * m_vox + unmasked_weight * (1.0 - m_vox)
        charb = charbonnier(residual, eps=eps_charb)
        return _safe_mean(charb * w, w)

    raise ValueError(f"Unknown recon variant: {variant}")

# -------------------------
# Mask loss (BCE [+ Dice])
# -------------------------

def mask_loss(
    m_logits: torch.Tensor,    # (B,P)
    M: torch.Tensor,           # (B,P) in {0,1}
    pos_weight: str | float = "auto",  # "auto" or scalar
    dice_gamma: float = 0.0,   # add gamma * (1 - Dice)
    eps: float = 1e-6,
) -> torch.Tensor:
    assert m_logits.shape == M.shape, "shape mismatch for mask loss"
    B, P = M.shape

    # BCE with class balancing
    if pos_weight == "auto":
        pos = M.float().sum().item()
        neg = P * B - pos
        if pos > 0:
            pw = torch.tensor(neg / max(pos, 1.0), device=m_logits.device, dtype=m_logits.dtype)
            bce = F.binary_cross_entropy_with_logits(m_logits, M.float(), pos_weight=pw)
        else:
            # No positives: fall back to unweighted BCE
            bce = F.binary_cross_entropy_with_logits(m_logits, M.float())
    else:
        bce = F.binary_cross_entropy_with_logits(m_logits, M.float(), pos_weight=torch.as_tensor(pos_weight, device=m_logits.device, dtype=m_logits.dtype))

    if dice_gamma <= 0:
        return bce

    # Soft Dice (probabilities), averaged over batch
    p = torch.sigmoid(m_logits)  # (B,P)
    intersection = (p * M).sum(dim=1)
    denom = (p.sum(dim=1) + M.sum(dim=1)).clamp_min(eps)
    dice = (2.0 * intersection + eps) / (denom + eps)
    dice_loss = 1.0 - dice.mean()

    return bce + dice_gamma * dice_loss


def rotation_loss(
    r_logits: torch.Tensor,  # (B,P,K)
    R: torch.Tensor,         # (B,P) long
    M: torch.Tensor | None = None,  # (B,P) float/bool, 1==masked
    masked_only: bool = True,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    assert r_logits.dim() == 3, "r_logits must be (B,P,K)"
    B, P, K = r_logits.shape
    R = R.long()
    if masked_only:
        assert M is not None, "M required for masked_only rotation loss"
        sel = (M > 0.5)
        if not torch.any(sel):
            # No masked patches in this batch: zero loss on correct device/dtype
            return r_logits.sum() * 0.0
        r_sel = r_logits[sel]             # (N_masked, K)
        y_sel = R[sel]                    # (N_masked,)
        return F.cross_entropy(r_sel, y_sel, reduction="mean", label_smoothing=label_smoothing)
    else:
        return F.cross_entropy(r_logits.view(B * P, K), R.view(B * P), reduction="mean", label_smoothing=label_smoothing)


class TwoFoldLoss(nn.Module):
    """
    Flexible loss combiner for two-fold SSL.

    Usage (manual lambdas):
        crit = TwoFoldLoss(
            recon_variant="full_charbonnier",
            lambda_rec=1.0, lambda_mask=0.05, lambda_rot=0.05,
            use_uncertainty=False
        )

    Usage (learned weighting):
        crit = TwoFoldLoss(recon_variant="masked_only_charbonnier", use_uncertainty=True)

    Forward:
        out = crit(
            x_hat, x, m_logits, M, r_logits, R,
            grid_shape=(n,n,n)   # required for masked-only/weighted recon variants
        )
        loss = out["loss_total"]
        # individual terms: out["loss_rec"], out["loss_mask"], out["loss_rot"]
    """
    def __init__(
        self,
        recon_variant: str = "full_charbonnier",
        mask_pos_weight: str | float = "auto",
        mask_dice_gamma: float = 0.0,
        rot_masked_only: bool = True,
        rot_label_smoothing: float = 0.05,
        # reconstruction knobs
        recon_mask_weight: float = 5.0,
        recon_unmasked_weight: float = 1.0,
        recon_charb_eps: float = 1e-3,
        # weighting
        lambda_rec: float = 1.0,
        lambda_mask: float = 0.1,
        lambda_rot: float = 0.1,
        use_uncertainty: bool = False,
    ):
        super().__init__()
        self.recon_variant = recon_variant
        self.mask_pos_weight = mask_pos_weight
        self.mask_dice_gamma = mask_dice_gamma
        self.rot_masked_only = rot_masked_only
        self.rot_label_smoothing = rot_label_smoothing
        self.recon_mask_weight = recon_mask_weight
        self.recon_unmasked_weight = recon_unmasked_weight
        self.recon_charb_eps = recon_charb_eps

        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            # log sigma parameters (start at 0 => sigma=1)
            self.log_sigma_rec = nn.Parameter(torch.zeros(1))
            #[ABLATON 1 : MASKING REMOVED, RECONSTRUCTION + ROTATION KEPT]
            self.log_sigma_mask = nn.Parameter(torch.zeros(1))

            #[ABLATON 2 : ROTATION REMOVED, RECONSTRUCTION + MASKING KEPT]
            self.log_sigma_rot = nn.Parameter(torch.zeros(1))
        else:
            self.lambda_rec = lambda_rec
            #[ABLATON 1 : MASKING REMOVED, RECONSTRUCTION + ROTATION KEPT]
            self.lambda_mask = lambda_mask
            #[ABLATON 2 : ROTATION REMOVED, RECONSTRUCTION + MASKING KEPT]
            self.lambda_rot = lambda_rot

    def forward(
        self,
        x_hat: torch.Tensor, x: torch.Tensor,
        m_logits: torch.Tensor, M: torch.Tensor,
        r_logits: torch.Tensor, R: torch.Tensor,
        grid_shape: tuple[int,int,int] | None = None,
    ):


        # Reconstruction
        L_rec = recon_loss(
            x_hat, x, M=M, grid_shape=grid_shape,
            variant=self.recon_variant,
            mask_weight=self.recon_mask_weight,
            unmasked_weight=self.recon_unmasked_weight,
            eps_charb=self.recon_charb_eps,
        )
        #[ABLATON 1 : MASKING REMOVED, RECONSTRUCTION + ROTATION KEPT]
        # Mask
        L_mask = mask_loss(
             m_logits, M,
             pos_weight=self.mask_pos_weight,
             dice_gamma=self.mask_dice_gamma,
         )
        # L_mask = torch.tensor(0.0,device=x_hat.device)

        #[ABLATON 2 : ROTATION REMOVED, RECONSTRUCTION + MASKING KEPT]
        # Rotation
        L_rot = rotation_loss(
             r_logits, R, M=M,
             masked_only=self.rot_masked_only,
             label_smoothing=self.rot_label_smoothing,
         )
        #L_rot = torch.tensor(0.0, device=x_hat.device)

        if self.use_uncertainty:
            # Kendall & Gal (2018): sum( L_t / (2 sigma_t^2) + log sigma_t )
            sigma_rec = torch.exp(self.log_sigma_rec)
            #[ABLATON 1 : MASKING REMOVED, RECONSTRUCTION + ROTATION KEPT]
            sigma_mask = torch.exp(self.log_sigma_mask)
            #[ABLATON 2 : ROTATION REMOVED, RECONSTRUCTION + MASKING KEPT]
            sigma_rot = torch.exp(self.log_sigma_rot)
            loss_total = (
                L_rec / (2 * sigma_rec * sigma_rec) + self.log_sigma_rec +
                #[ABLATON 1 : MASKING REMOVED, RECONSTRUCTION + ROTATION KEPT]
                L_mask / (2 * sigma_mask * sigma_mask) + self.log_sigma_mask +
                #[ABLATON 2 : ROTATION REMOVED, RECONSTRUCTION + MASKING KEPT]
                L_rot / (2 * sigma_rot * sigma_rot) + self.log_sigma_rot
            )
        else:
            loss_total = (
                self.lambda_rec * L_rec +
                #[ABLATON 1 : MASKING REMOVED, RECONSTRUCTION + ROTATION KEPT]
                self.lambda_mask * L_mask +
                #[ABLATON 2 : ROTATION REMOVED, RECONSsweTRUCTION + MASKING KEPT]
                self.lambda_rot * L_rot
            )
        #print(f"labda_rec", self.lambda_rec)
        #print(f"lambda_mask", self.lambda_mask)
        #print(f"lambda_rot", self.lambda_rot)
        return {
            "loss_total": loss_total,
            "loss_rec": L_rec.detach(),
            "loss_mask": L_mask.detach(), 
            "loss_rot": L_rot.detach(), #should return 0.0
        }
