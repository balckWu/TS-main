# tasksegment/training/trainer.py
from __future__ import annotations
from pathlib import Path  # 新增：用于处理保存路径
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 新增：用于显示进度条

from ..configs import (
    DEFAULT_DENSE_GUIDANCE_LOSS_CFG, DEFAULT_DOMAIN_THRESHOLDS, DEFAULT_EMA_ALPHA_CFG,
    DEFAULT_MODEL_CONFIG, DEFAULT_POSTPROCESS_CFG, DEFAULT_TASK_IMPORTANCE_CFG,
    DEFAULT_TRAIN_LOSS_WEIGHT_CFG,
)
from ..data.datasets import MultiOrganDataset
from ..text.bank import make_Ftext_batch
from .augmentations import augment_episode_medical
from .evaluation import evaluate
from .losses import (
    boundary_weighted_ce_loss, compute_dense_guidance_supervision, compute_task_importance,
    dice_loss, get_boundary_weight,
)
from .memory import EMATaskMemoryBank, make_balanced_domain_schedule

def _resolve_float_cfg(user_cfg, default_cfg):
    cfg = dict(default_cfg)
    if user_cfg is not None:
        cfg.update({key: float(value) for key, value in user_cfg.items()})
    return cfg

def train(
    model: torch.nn.Module,
    train_dataset: MultiOrganDataset,
    val_dataset: MultiOrganDataset,
    device: torch.device,
    text_bank: Dict[str, torch.Tensor],
    is_baseline: bool = False,  # 新增：是否为基线模型
    num_epochs: int = 100,
    batch_size: int = 2,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    patience: int = 8,
    save_path: str = "best_model.pth",
    episodes_per_epoch: Optional[int] = None,
    grad_accum_steps: int = 4,
    ema_bank_momentum: float = 0.99,
    ema_alpha_cfg: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    boundary_loss_weight: float = 0.5,
    model_config: Optional[Dict[str, Any]] = None,
    warmup_epochs: int = 3,
    dense_guidance_loss_cfg: Optional[Dict[str, float]] = None,
    train_loss_weight_cfg: Optional[Mapping[str, float]] = None,
    task_importance_cfg: Optional[Mapping[str, float]] = None,
) -> None:
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    ce_loss = nn.CrossEntropyLoss()
    
    domains = list(train_dataset.organ_to_positive_indices.keys())
    best_val_dice, patience_counter = -1.0, 0
    ema_memory_bank = EMATaskMemoryBank(momentum=ema_bank_momentum) if not is_baseline else None
    
    modality_map = {"thyroid": "thyroid", "TN3K": "thyroid", "BUSI_WHU": "BUSI", "BUS-BRA": "BUSI", "OTU": "generic", "prostate": "generic"}
    train_loss_weight_cfg = _resolve_float_cfg(train_loss_weight_cfg, DEFAULT_TRAIN_LOSS_WEIGHT_CFG)
    task_importance_cfg = _resolve_float_cfg(task_importance_cfg, DEFAULT_TASK_IMPORTANCE_CFG)
    dense_main_weight = float(train_loss_weight_cfg.get("dense_main", DEFAULT_TRAIN_LOSS_WEIGHT_CFG["dense_main"]))
    
    if ema_alpha_cfg is None: ema_alpha_cfg = dict(DEFAULT_EMA_ALPHA_CFG)
    if model_config is None: model_config = dict(DEFAULT_MODEL_CONFIG)
    if dense_guidance_loss_cfg is None: dense_guidance_loss_cfg = dict(DEFAULT_DENSE_GUIDANCE_LOSS_CFG)

    print(f"📋 训练域列表: {domains}")
    if is_baseline:
        print("📋 基线模型模式: 不使用 EMA Memory Bank 和 Dense Guidance Loss")
    else:
        print("📋 TaskSegmentV3 模式: 使用 raw episode task tokens 更新 EMA Bank")

    def compute_segmentation_loss(pred_logits, target_mask):
        boundary_map = get_boundary_weight(target_mask.unsqueeze(1).float(), 3, 5.0)
        return ce_loss(pred_logits, target_mask) + dice_loss(pred_logits, target_mask) + boundary_loss_weight * boundary_weighted_ce_loss(pred_logits, target_mask, boundary_map)

    for epoch in range(num_epochs):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            for p in optimizer.param_groups: p["lr"] = lr * float(epoch + 1) / float(warmup_epochs)
        
        model.train()
        total_iters = int(episodes_per_epoch) if episodes_per_epoch else max(1, len(train_dataset) // (2 * batch_size))
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(range(total_iters), desc=f"Epoch {epoch+1:03d}/{num_epochs:03d}", leave=False, ncols=100)
        optimizer.zero_grad(set_to_none=True)
        
        epoch_loss, epoch_train_dice_hard = 0.0, 0.0
        domain_schedule = make_balanced_domain_schedule(domains, total_iters)

        for step in pbar:
            domain = domain_schedule[step]
            support_indices, query_indices = train_dataset.sample_episode_indices(domain, num_support=batch_size, num_query=batch_size)
            
            xs = torch.stack([train_dataset[i][0] for i in support_indices]).to(device)
            ys = torch.stack([train_dataset[i][1] for i in support_indices]).unsqueeze(1).float().to(device)
            xq = torch.stack([train_dataset[i][0] for i in query_indices]).to(device)
            yq = torch.stack([train_dataset[i][1] for i in query_indices]).to(device)

            modality = modality_map.get(domain, "generic")
            xs, ys = augment_episode_medical(xs, ys, modality=modality)
            xq, yq_aug = augment_episode_medical(xq, yq.unsqueeze(1).float(), modality=modality)
            yq = yq_aug.squeeze(1).long()

            organ_text = text_bank.get(domain)
            Ftext = make_Ftext_batch(organ_text, batch_size) if organ_text is not None else None

            out = model(xs=xs, ys=ys, Ftext=Ftext, xq=xq)
            pred = out["pred_masks"]

            if not is_baseline:
                task_tokens = out["task_tokens"]
                ema_memory_bank.update(domain, task_tokens, importance=compute_task_importance(pred, yq, ys, cfg=task_importance_cfg))

            seg_loss = compute_segmentation_loss(pred, yq)
            
            if not is_baseline:
                dense_loss = compute_dense_guidance_supervision(out, yq.unsqueeze(1).float(), dense_guidance_loss_cfg)
                loss = seg_loss + dense_main_weight * dense_loss
            else:
                loss = seg_loss

            # 实时更新进度条上的显示信息
            pbar.set_postfix({'loss': f"{loss.item():.3f}", 'dom': domain})

            (loss / grad_accum_steps).backward()

            if ((step + 1) % grad_accum_steps == 0) or (step + 1 == total_iters):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                prob = torch.softmax(pred, dim=1)[:, 1]
                pred_binary = (prob > 0.5).float()
                target_fg = (yq == 1).float()
                intersection = (pred_binary * target_fg).sum(dim=(1, 2))
                union = pred_binary.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
                batch_dice_hard = ((2.0 * intersection + 1.0) / (union + 1.0)).mean().item()
                epoch_loss += loss.item()
                epoch_train_dice_hard += batch_dice_hard

        iter_count = max(1, total_iters)
        train_avg_loss = epoch_loss / iter_count
        train_avg_dice = epoch_train_dice_hard / iter_count

        val_metrics = evaluate(
            model=model, query_dataset=val_dataset, device=device, text_bank=text_bank,
            ema_memory_bank=ema_memory_bank, is_baseline=is_baseline,
            domain_thresholds=dict(DEFAULT_DOMAIN_THRESHOLDS), postprocess_cfg=dict(DEFAULT_POSTPROCESS_CFG), use_postprocess=True,
        )

        if epoch + 1 > warmup_epochs:
            scheduler.step(val_metrics["dice"])
        current_lr = optimizer.param_groups[0]["lr"]
        improved = val_metrics["dice"] > best_val_dice

        if improved:
            best_val_dice = val_metrics["dice"]
            patience_counter = 0
            
            # 🔥 新增：确保保存目录存在，防止 RuntimeError: Parent directory does not exist.
            save_dir_path = Path(save_path).parent
            save_dir_path.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_dice": best_val_dice,
                "ema_memory_bank": ema_memory_bank.state_dict() if ema_memory_bank is not None else {},
                "model_config": dict(model_config),
                "train_config": {
                    "arch": "baseline" if is_baseline else "ours",
                    "image_size": tuple(getattr(train_dataset, "image_size", (512, 512))),
                    "ema_alpha_cfg": dict(ema_alpha_cfg),
                    "task_token_source": "ema_memory_bank" if not is_baseline else "none",
                    "warmup_epochs": int(warmup_epochs),
                    "dense_main_weight": float(dense_main_weight),
                    "train_loss_weight_cfg": dict(train_loss_weight_cfg),
                    "task_importance_cfg": dict(task_importance_cfg),
                    "dense_guidance_loss_cfg": dict(dense_guidance_loss_cfg),
                    "domain_thresholds": dict(DEFAULT_DOMAIN_THRESHOLDS),
                    "postprocess_cfg": dict(DEFAULT_POSTPROCESS_CFG),
                    "text_dir": "./text_features_llm",
                },
            }, save_path)
        else:
            patience_counter += 1

        status = "saved" if improved else f"no_improve({patience_counter}/{patience})"
        print(f"Epoch {epoch + 1:03d}/{num_epochs:03d} | train_loss={train_avg_loss:.4f} | train_dice={train_avg_dice:.4f} | val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f} | val_hd95={val_metrics['hd95']:.2f} | lr={current_lr:.2e} | best={best_val_dice:.4f} | {status}", flush=True)

        if (not improved) and patience_counter >= patience:
            print(f"🛑 Early stopping triggered after epoch {epoch + 1}.", flush=True)
            break
