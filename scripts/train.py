# scripts/train.py
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Tuple

import torch

# ============================================================
# 让脚本可以直接通过 python scripts/train.py 运行
# 不再需要 PYTHONPATH=.
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasksegment.configs import (
    DataConfig,
    DEFAULT_DENSE_GUIDANCE_LOSS_CFG,
    DEFAULT_EMA_ALPHA_CFG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TASK_IMPORTANCE_CFG,
    DEFAULT_TRAIN_LOSS_WEIGHT_CFG,
)
from tasksegment.data import get_datasets, set_seed
from tasksegment.models import TaskSegmentModel, StandardUNet, UNetPlusPlus, NNUNetLike
from tasksegment.models.transformer_baselines import SwinUnet, TransUNet
from tasksegment.text import load_text_bank
from tasksegment.training import train

def _parse_image_size(values) -> Tuple[int, int]:
    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) == 2:
        return int(values[0]), int(values[1])
    raise ValueError("--image-size 只接受 1 个值 H 或 2 个值 H W，例如 --image-size 512 或 --image-size 512 512")

def parse_args():
    parser = argparse.ArgumentParser(description="Train TaskSegmentV3 / Baselines")
    
    # -------------------------
    # 架构选择 (新增)
    # -------------------------
    parser.add_argument(
        "--arch", type=str, default="ours",
        choices=["ours", "unet", "unetpp", "nnunet", "swinunet", "transunet"],
        help="选择训练架构"
    )

    # -------------------------
    # 路径参数
    # -------------------------
    parser.add_argument(
        "--data-root", type=str, default="./data",
        help="数据集根目录，下面应包含 thyroid、TN3K、BUSI_WHU、BUS-BRA 等子目录",
    )
    parser.add_argument(
        "--text-dir", type=str, default="./text_features_llm",
        help="文本特征目录，默认使用 ./text_features_llm",
    )
    parser.add_argument(
        "--save-path", type=str, default="USTS.pth",
        help="模型保存路径，默认保存为 USTS.pth",
    )

    # -------------------------
    # 显式实验配置
    # -------------------------
    parser.add_argument(
        "--image-size", type=int, nargs="+", default=[512, 512], metavar=("H", "W"),
        help="输入图像尺寸。支持 --image-size 512 或 --image-size 512 512",
    )
    ftext_group = parser.add_mutually_exclusive_group()
    ftext_group.add_argument(
        "--use-ftext", dest="use_ftext", action="store_true", default=bool(DEFAULT_MODEL_CONFIG["use_ftext"]),
        help="启用文本 token / Ftext",
    )
    ftext_group.add_argument(
        "--no-use-ftext", dest="use_ftext", action="store_false",
        help="禁用文本 token / Ftext",
    )
    parser.add_argument("--text-dim", type=int, default=int(DEFAULT_MODEL_CONFIG["text_dim"]))
    parser.add_argument("--hidden-dim", type=int, default=int(DEFAULT_MODEL_CONFIG["hidden_dim"]))
    parser.add_argument("--num-query-tokens", type=int, default=int(DEFAULT_MODEL_CONFIG["num_query_tokens"]))
    parser.add_argument("--guidance-dropout-prob", type=float, default=float(DEFAULT_MODEL_CONFIG.get("guidance_dropout_prob", 0.15)))
    parser.add_argument("--guidance-scale-min", type=float, default=float(DEFAULT_MODEL_CONFIG.get("guidance_scale_min", 0.50)))
    parser.add_argument("--guidance-scale-max", type=float, default=float(DEFAULT_MODEL_CONFIG.get("guidance_scale_max", 1.00)))
    parser.add_argument("--guidance-clamp-max", type=float, default=float(DEFAULT_MODEL_CONFIG.get("guidance_clamp_max", 0.80)))
    parser.add_argument("--cache-dataset", action="store_true")
    
    clahe_group = parser.add_mutually_exclusive_group()
    clahe_group.add_argument("--cpu-clahe", dest="cpu_clahe", action="store_true")
    clahe_group.add_argument("--no-cpu-clahe", dest="cpu_clahe", action="store_false")
    parser.set_defaults(cpu_clahe=True)
    
    parser.add_argument("--train-query-positive-ratio", type=float, default=0.7)

    # -------------------------
    # 训练参数
    # -------------------------
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--episodes-per-epoch", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=3)

    parser.add_argument("--ema-bank-momentum", type=float, default=0.99)

    parser.add_argument("--boundary-loss-weight", type=float, default=0.5)
    parser.add_argument("--raw-aux-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--token-consistency-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--raw-dense-aux-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--dense-main-weight", type=float, default=DEFAULT_TRAIN_LOSS_WEIGHT_CFG["dense_main"])
    parser.add_argument("--task-importance-dice-weight", type=float, default=DEFAULT_TASK_IMPORTANCE_CFG["dice_weight"])
    parser.add_argument("--task-importance-confidence-weight", type=float, default=DEFAULT_TASK_IMPORTANCE_CFG["confidence_weight"])
    parser.add_argument("--task-importance-support-size-weight", type=float, default=DEFAULT_TASK_IMPORTANCE_CFG["support_size_weight"])
    parser.add_argument("--task-importance-support-size-norm", type=float, default=DEFAULT_TASK_IMPORTANCE_CFG["support_size_norm"])
    parser.add_argument("--task-importance-min", type=float, default=DEFAULT_TASK_IMPORTANCE_CFG["min_importance"])

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    return parser.parse_args()

def check_paths(args):
    data_root = Path(args.data_root)
    text_dir = Path(args.text_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")
    if not text_dir.exists():
        print(f"[警告] 文本特征目录不存在: {text_dir}\n如果你启用了文本特征，请先运行:\n python scripts/generate_llm_text_features.py\n")

def main():
    args = parse_args()
    check_paths(args)
    image_size = _parse_image_size(args.image_size)
    set_seed(args.seed, deterministic=args.deterministic)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 指定了 cuda，但当前环境不可用，将自动切换到 CPU。")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print(f"Training Architecture: {args.arch}")
    print("=" * 80)
    print(f"项目根目录: {ROOT}")
    print(f"数据目录: {Path(args.data_root).resolve()}")
    print(f"设备: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {image_size}")

    data_cfg = DataConfig(data_root=args.data_root, image_size=image_size)
    train_dataset, val_dataset, _ = get_datasets(
        data_cfg.roots(), image_size=data_cfg.image_size, seed=args.seed,
        train_query_positive_ratio=args.train_query_positive_ratio,
        prefer_empty_train_queries=True, cache_in_memory=args.cache_dataset, use_cpu_clahe=args.cpu_clahe,
    )
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    text_bank = load_text_bank(args.text_dir, device, expected_text_dim=args.text_dim)
    
    # 判断是否为基线模型（不包含 Ours 的结构）
    is_baseline = args.arch in ["unet", "unetpp", "nnunet", "swinunet", "transunet"]

    if is_baseline:
        img_size_h = image_size[0]  # 提取高度作为 Swin/TransUNet 的 img_size
        
        if args.arch == "unet":
            model = StandardUNet(in_channels=1, num_classes=2).to(device)
        elif args.arch == "unetpp":
            model = UNetPlusPlus(in_channels=1, num_classes=2).to(device)
        elif args.arch == "nnunet":
            model = NNUNetLike(in_channels=1, num_classes=2).to(device)
        elif args.arch == "swinunet":
            model = SwinUnet(in_channels=1, num_classes=2, img_size=img_size_h).to(device)
        elif args.arch == "transunet":
            model = TransUNet(in_channels=1, num_classes=2, img_size=img_size_h).to(device)
        
        model_config = {"arch": args.arch}
    else:
        model_config = dict(DEFAULT_MODEL_CONFIG)
        model_config.update({
            "use_ftext": bool(args.use_ftext), "text_dim": int(args.text_dim),
            "hidden_dim": int(args.hidden_dim), "num_query_tokens": int(args.num_query_tokens),
            "use_dense_text_guidance": True, "guidance_dropout_prob": float(args.guidance_dropout_prob),
            "guidance_scale_min": float(args.guidance_scale_min), "guidance_scale_max": float(args.guidance_scale_max),
            "guidance_clamp_max": float(args.guidance_clamp_max),
        })
        model = TaskSegmentModel(**model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("模型配置:")
    for k, v in model_config.items(): print(f" {k}: {v}")
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")

    train_loss_weight_cfg = dict(DEFAULT_TRAIN_LOSS_WEIGHT_CFG)
    train_loss_weight_cfg.update({"dense_main": float(args.dense_main_weight)})
    dense_guidance_loss_cfg = dict(DEFAULT_DENSE_GUIDANCE_LOSS_CFG)
    task_importance_cfg = dict(DEFAULT_TASK_IMPORTANCE_CFG)
    task_importance_cfg.update({
        "dice_weight": float(args.task_importance_dice_weight),
        "confidence_weight": float(args.task_importance_confidence_weight),
        "support_size_weight": float(args.task_importance_support_size_weight),
        "support_size_norm": float(args.task_importance_support_size_norm),
        "min_importance": float(args.task_importance_min),
    })

    train(
        model=model, train_dataset=train_dataset, val_dataset=val_dataset, device=device,
        text_bank=text_bank, is_baseline=is_baseline,
        num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, episodes_per_epoch=args.episodes_per_epoch, grad_accum_steps=args.grad_accum_steps,
        warmup_epochs=args.warmup_epochs, save_path=args.save_path, ema_bank_momentum=args.ema_bank_momentum,
        ema_alpha_cfg=dict(DEFAULT_EMA_ALPHA_CFG), boundary_loss_weight=args.boundary_loss_weight,
        model_config=model_config, dense_guidance_loss_cfg=dense_guidance_loss_cfg,
        train_loss_weight_cfg=train_loss_weight_cfg, task_importance_cfg=task_importance_cfg,
    )

if __name__ == "__main__":
    main()
