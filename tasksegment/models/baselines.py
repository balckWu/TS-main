# tasksegment/models/baselines.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 导入依赖 ---
try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

try:
    from monai.networks.nets import DynUNet, SwinUNETR
except ImportError:
    DynUNet = None
    SwinUNETR = None


# ==========================================
# 0. 统一基类 (确保对比实验绝对公平的核心)
# ==========================================
class BaseBaseline(nn.Module):
    """基线模型统一接口适配器"""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    def forward(self, xs=None, ys=None, Ftext=None, xq=None):
        if xq is None:
            raise ValueError("xq (query image) must be provided for baselines")
        logits = self.net(xq)
        return {"pred_masks": logits}

    def segment_with_task(self, xq=None, task_tokens=None, query_feats=None, output_size=None):
        """适配 evaluation.py / inference.py 中的接口"""
        out = self(xq=xq) 
        logits = out["pred_masks"]
        
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
            
        return {"pred_masks": logits}

    def encode_query(self, xq):
        return xq  # 基线不需要特征金字塔


# ==========================================
# 1. 纯 CNN 基线模型
# ==========================================
class StandardUNet(BaseBaseline):
    """标准 2D UNet Baseline (使用 smp 库实现)"""
    def __init__(self, in_channels=1, num_classes=2, base_dim=64):
        super().__init__(in_channels, num_classes)
        if smp is None:
            raise ImportError("请安装 segmentation_models_pytorch")
        self.net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes
        )

class UNetPlusPlus(BaseBaseline):
    """UNet++ Baseline (使用 smp 库实现)"""
    def __init__(self, in_channels=1, num_classes=2, base_dim=64):
        super().__init__(in_channels, num_classes)
        if smp is None:
            raise ImportError("请安装 segmentation_models_pytorch")
        self.net = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes
        )

class NNUNetLike(BaseBaseline):
    """模拟 nnUNet 骨架 Baseline (使用 monai 库实现)"""
    def __init__(self, in_channels=1, num_classes=2, base_dim=32):
        super().__init__(in_channels, num_classes)
        if DynUNet is None:
            raise ImportError("请安装 monai")
        kernels, strides = [3, 3, 3, 3, 3], [1, 2, 2, 2, 2]
        self.net = DynUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=[base_dim, base_dim*2, base_dim*4, base_dim*8, base_dim*16],
            norm_name="batch",
            deep_supervision=False,
            res_block=True
        )


# ==========================================
# 2. Transformer 基线模型
# ==========================================
class SwinUnet(BaseBaseline):
    """Swin-Unet Baseline (依赖 monai 库的 SwinUNETR)"""
    def __init__(self, in_channels=1, num_classes=2, img_size=256):
        super().__init__(in_channels, num_classes)
        if SwinUNETR is None:
            raise ImportError("请安装 monai")
        
        # 🌟 修复点：已移除 img_size 参数以适配最新版 MONAI
        self.net = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=48,
            spatial_dims=2,
            use_checkpoint=False
        )

# --- TransUNet 依赖的底层模块 ---
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
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

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
        return self.proj_drop(self.proj(x))

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
        for blk in self.blocks: x = blk(x)
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
        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.encoder2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.encoder3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        # --- ViT Bottleneck ---
        self.patch_embed = nn.Conv2d(256, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // 4 // self.patch_size)**2, self.hidden_dim))
        self.transformer = Transformer(dim=self.hidden_dim, depth=self.depth, num_heads=self.num_heads)
        self.norm = nn.LayerNorm(self.hidden_dim)
        
        # --- Decoder ---
        self.decoder3 = nn.Sequential(nn.Conv2d(256 + self.hidden_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.decoder2 = nn.Sequential(nn.Conv2d(128 + 256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.decoder1 = nn.Sequential(nn.Conv2d(64 + 128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.head = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, xq=None, **kwargs):
        e1 = self.encoder1(xq)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        
        x = self.patch_embed(e3)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x).transpose(1, 2).reshape(B, C, H, W) 
        
        d3 = F.interpolate(x, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1)) 
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1)) 
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1)) 
        
        return {"pred_masks": self.head(d1)}