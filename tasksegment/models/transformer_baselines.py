# tasksegment/models/transformer_baselines.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入基线适配器
from .baselines import BaseBaseline

# 如果没有安装，请 pip install monai
try:
    from monai.networks.nets import SwinUNETR
except ImportError:
    SwinUNETR = None

# ==========================================
# 1. Swin-Unet (使用 MONAI 标准实现)
# ==========================================
class SwinUnet(BaseBaseline):
    """Swin-Unet Baseline (依赖 monai 库的 SwinUNETR)"""
    def __init__(self, in_channels=1, num_classes=2, img_size=256):
        super().__init__(in_channels, num_classes)
        if SwinUNETR is None:
            raise ImportError("请安装 monai: pip install monai")
        
        self.net = SwinUNETR(
            img_size=(img_size, img_size),
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=48,
            spatial_dims=2,
            use_checkpoint=False
        )

# ==========================================
# 2. TransUNet (纯 PyTorch 标准实现)
# ==========================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, dim=512, depth=12, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class TransUNet(BaseBaseline):
    """TransUNet Baseline (CNN Encoder + ViT Bottleneck + CNN Decoder)"""
    def __init__(self, in_channels=1, num_classes=2, img_size=256):
        super().__init__(in_channels, num_classes)
        
        self.patch_size = 16
        self.hidden_dim = 512
        self.depth = 12
        self.num_heads = 8
        
        # --- CNN Encoder (3 stages) ---
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        
        # --- ViT Bottleneck ---
        self.patch_embed = nn.Conv2d(256, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // 4 // self.patch_size)**2, self.hidden_dim))
        self.transformer = Transformer(dim=self.hidden_dim, depth=self.depth, num_heads=self.num_heads)
        self.norm = nn.LayerNorm(self.hidden_dim)
        
        # --- Decoder ---
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256 + self.hidden_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        
        self.head = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)
        # 移除了 self.up，改用动态 F.interpolate
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, xq=None, **kwargs):
        x = xq
        
        # Encoder
        e1 = self.encoder1(x)        # [B, 64, H, W]
        e2 = self.encoder2(self.pool(e1)) # [B, 128, H/2, W/2]
        e3 = self.encoder3(self.pool(e2)) # [B, 256, H/4, W/4]
        
        # ViT Bottleneck
        x = self.patch_embed(e3)     # [B, 512, H/16, W/16] (对于512输入，这里是8x8)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, N, 512]
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x).transpose(1, 2).reshape(B, C, H, W) 
        
        # Decoder (使用动态插值替代固定 scale_factor 的 Upsample)
        d3 = F.interpolate(x, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1)) 
        
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1)) 
        
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1)) 
        
        logits = self.head(d1)
        return {"pred_masks": logits}

