from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.blocks import UnetrBasicBlock
from monai.utils import ensure_tuple_rep

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=1152, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # x: (B, in_dim)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class SwinUNETREncoder(nn.Module):
    def __init__(self, in_channels=1, feature_size=48, spatial_dims=3,
                 dropout_path_rate=0.0, use_checkpoint=False, img_size=(64,64,64)):
        super().__init__()
        self.spatial_dims = spatial_dims
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        self.net = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2,2,2,2],
            num_heads=[3,6,12,24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            use_v2=True,
        )
        norm_name = "instance"
      
        self.encoder1 = UnetrBasicBlock(spatial_dims, in_channels, feature_size,   3, 1, norm_name, res_block=True)
        self.encoder2 = UnetrBasicBlock(spatial_dims, feature_size, feature_size,  3, 1, norm_name, res_block=True)
        self.encoder3 = UnetrBasicBlock(spatial_dims, 2*feature_size, 2*feature_size, 3, 1, norm_name, res_block=True)
        self.encoder4 = UnetrBasicBlock(spatial_dims, 4*feature_size, 4*feature_size, 3, 1, norm_name, res_block=True)
        self.encoder10= UnetrBasicBlock(spatial_dims,16*feature_size,16*feature_size, 3, 1, norm_name, res_block=True)

        # Projection head input dim is sum of pooled channel dims: (1+1+2+4+16)*feature_size
        self.ms_proj_in_dim = (1+1+2+4+16) * feature_size
        self.proj_head = ProjectionHead(in_dim=self.ms_proj_in_dim, hidden_dim=2048, out_dim=2048)

    def _pool_cat(self, encs):
        b = encs[0].size(0)
        pooled = [F.adaptive_avg_pool3d(e, (1,1,1)).view(b, -1) for e in encs]
        return torch.cat(pooled, dim=1)  # (B, ms_proj_in_dim)

    def forward(self, x):
        """
        Returns:
          encs: [enc0, enc1, enc2, enc3, dec4]
          z_global: (B, ms_proj_in_dim) — pooled multi-scale vector
          z_proj: (B, 2048) — 3-layer MLP projection (useful if needed)
          z_spatial: dec4 spatial map (B, 16F, D',H',W') for reconstruction head
        """
        hs = self.net(x)  # list: [feat1, feat2, feat3, feat4, feat5]

        
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hs[0])
        enc2 = self.encoder3(hs[1])
        enc3 = self.encoder4(hs[2])
        dec4 = self.encoder10(hs[4])  # highest-level

        encs = [enc0, enc1, enc2, enc3, dec4]
        z_global = self._pool_cat(encs)              # (B, (1+1+2+4+16)*F)
        z_proj   = self.proj_head(z_global)          # (B, 2048)
        z_spatial= dec4                               # spatial map for recon
        return hs, encs, z_global, z_proj, z_spatial

    @property
    def embed_channels(self):
        # 16*feature_size is the channel count of dec4
        return self.encoder10.conv1.out_channels

    def extract_encoder_state(self):
        keep = {}
        for k,v in self.swinViT.state_dict().items():
            keep[f"swinViT.{k}"] = v
        return keep


class MaskHeadFromVec(nn.Module):
   
    def __init__(self, in_dim: int, num_patches: int, hid: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.GELU(),
            nn.Linear(hid, num_patches)
        )
    def forward(self, z_global):  # (B, in_dim)
        return self.mlp(z_global) # logits (B, P)

class RotHeadFromVec(nn.Module):
    #what transformation happened there (orientation)
    def __init__(self, in_dim: int, num_patches: int, K: int, hid: int = 512):
        super().__init__()
        self.num_patches, self.K = num_patches, K
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.GELU(),
            nn.Linear(hid, num_patches * K)
        )
    def forward(self, z_global):
        
        B = z_global.size(0)
        logits = self.mlp(z_global).view(B, self.num_patches, self.K)  # (B,P,K)
        return logits




class SwinUNETRDecoder(nn.Module):
    """
    Decoder path from MONAI's SwinUNETR.
    Used here for reconstruction pretraining, so at finetuning time
    both encoder and decoder weights can be reused.
    """
    def __init__(self, swinunetr_backbone, feature_size, out_ch: int = 1, spatial_dims=3):
            super().__init__()
            # reuse the exact decoder blocks defined in SwinUNETR
            self.decoder5 = swinunetr_backbone.decoder5
            self.decoder4 = swinunetr_backbone.decoder4
            self.decoder3 = swinunetr_backbone.decoder3
            self.decoder2 = swinunetr_backbone.decoder2
            self.decoder1 = swinunetr_backbone.decoder1
            # instead of segmentation head, use a 1-channel recon head
            from monai.networks.blocks import UnetOutBlock
            self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_ch)

    def forward(self, hidden_states_out, enc0, enc1, enc2, enc3, dec4):
   
        out = self.decoder5(dec4, hidden_states_out[3])
        out = self.decoder4(out, enc3)
        out = self.decoder3(out, enc2)
        out = self.decoder2(out, enc1)
        out = self.decoder1(out, enc0)
        return self.out(out)   



class TwoFoldSSL(nn.Module):
    """
    Mirrors the Swin-UNETR encoder wiring from the provided snippet,
    but keeps our 3-head objective and interface unchanged.
    """
    def __init__(self, in_ch=1, out_ch=1, feature_size=48,
                 img_size=(64,64,64), num_patches=8, K=9, spatial_dims = 3,
                 dropout_path_rate=0.0, use_checkpoint=False):
        super().__init__()

        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        self.encoder = SwinUNETREncoder(
            in_channels=in_ch, feature_size=feature_size, spatial_dims=3,
            dropout_path_rate=dropout_path_rate, use_checkpoint=use_checkpoint,
            img_size=img_size
        )

        self.swinunetr_backbone = SwinUNETR(in_channels=in_ch, out_channels=out_ch, feature_size = feature_size,dropout_path_rate = dropout_path_rate,
        patch_size = patch_size, window_size = window_size, spatial_dims = spatial_dims)

        # Heads
        self.recon = SwinUNETRDecoder(self.swinunetr_backbone, spatial_dims=spatial_dims, out_ch=out_ch, feature_size = feature_size)

        self.mask  = MaskHeadFromVec(in_dim=self.encoder.ms_proj_in_dim, num_patches=num_patches)

        self.rot   = RotHeadFromVec(in_dim=self.encoder.ms_proj_in_dim, num_patches=num_patches, K=K)

    def forward(self, x_tilde):
        hidden_states_out, encs, z_global, z_proj, z_spatial = self.encoder(x_tilde)
        enc0, enc1, enc2, enc3, dec4 = encs
        
        x_hat= self.recon(hidden_states_out, enc0, enc1, enc2, enc3,dec4) 
        m_logits= self.mask(z_global)     
        r_logits= self.rot(z_global)        
        return x_hat, m_logits, r_logits, z_spatial  
        

    def extract_encoder_state(self):
        return self.encoder.extract_encoder_state()
