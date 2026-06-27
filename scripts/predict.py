# scripts/predict.py
from __future__ import annotations
import sys, argparse
from pathlib import Path
from typing import List, Tuple
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasksegment.configs import DataConfig, DEFAULT_DOMAIN_THRESHOLDS, DEFAULT_MODEL_CONFIG, DEFAULT_POSTPROCESS_CFG
from tasksegment.data import MultiOrganDataset, get_image_transform, set_seed
from tasksegment.inference import predict_all_organs, load_model
from tasksegment.text import load_text_bank

def _parse_image_size(values) -> Tuple[int, int]:
    if values is None: raise ValueError("image size values cannot be None")
    if len(values) == 1: return int(values[0]), int(values[0])
    if len(values) == 2: return int(values[0]), int(values[1])
    raise ValueError("--image-size 只接受 1 个值 H 或 2 个值 H W")

def _parse_int_list(value: str) -> List[int]:
    if value is None or value.strip() == "": return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]

def parse_args():
    parser = argparse.ArgumentParser(description="Predict with TaskSegmentV3 / Baselines")
    parser.add_argument("--model-path", type=str, default="USTS.pth")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--text-dir", type=str, default="./text_features_llm")
    parser.add_argument("--save-dir", type=str, default="./vis_dense")
    parser.add_argument("--image-size", type=int, nargs="+", default=None, metavar=("H", "W"))
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--num-vis-per-organ", type=int, default=5)
    parser.add_argument("--vis-data-idx", type=str, default="")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--closing-kernel", type=int, default=None)
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    vis_data_indices = _parse_int_list(args.vis_data_idx)
    set_seed(args.seed, deterministic=False)
    device = torch.device("cpu") if args.device == "cuda" and not torch.cuda.is_available() else torch.device(args.device)

    # ============================================================
    # 1. 加载模型 (修改：接收 is_baseline)
    # ============================================================
    model, ema_bank, train_config, is_baseline = load_model(args.model_path, device)

    if args.image_size is not None:
        image_size = _parse_image_size(args.image_size)
    elif "image_size" in train_config:
        image_size = tuple(int(x) for x in train_config["image_size"])
    else:
        image_size = (512, 512)

    domain_thresholds = dict(train_config.get("domain_thresholds", DEFAULT_DOMAIN_THRESHOLDS))
    if args.threshold is not None:
        domain_thresholds = {k: float(args.threshold) for k in domain_thresholds.keys()}
        
    postprocess_cfg = dict(train_config.get("postprocess_cfg", DEFAULT_POSTPROCESS_CFG))
    if args.min_area is not None: postprocess_cfg["min_area"] = int(args.min_area)
    if args.closing_kernel is not None: postprocess_cfg["closing_kernel"] = int(args.closing_kernel)
    if args.keep_largest: postprocess_cfg["keep_largest"] = True

    if not is_baseline and args.guidance_scale is not None:
        model.decoder.dense_guidance_strength = float(args.guidance_scale)
    effective_guidance_scale = float(getattr(model.decoder, "dense_guidance_strength", 0.0)) if not is_baseline else 0.0

    print("=" * 80)
    print(f"Predicting Model: {'Baseline' if is_baseline else 'TaskSegmentV3'}")
    print(f"模型路径: {Path(args.model_path).resolve()}")
    print(f"设备: {device}")
    print(f"Image size: {image_size}")
    print("=" * 80)

    data_cfg = DataConfig(data_root=args.data_root, image_size=image_size)
    test_dataset = MultiOrganDataset(data_cfg.roots(), image_transform=get_image_transform(), split="test", image_size=data_cfg.image_size, seed=args.seed, cache_in_memory=args.cache_dataset, use_cpu_clahe=False)
    print(f"Test 数据集样本数: {len(test_dataset)}")

    text_bank = load_text_bank(args.text_dir, device, expected_text_dim=int(DEFAULT_MODEL_CONFIG["text_dim"]))
    
    # ============================================================
    # 2. 推理 (修改：传入 is_baseline)
    # ============================================================
    predict_all_organs(
        model=model, test_dataset=test_dataset, device=device, text_bank=text_bank,
        ema_bank=ema_bank, is_baseline=is_baseline, save_dir=args.save_dir,
        domain_thresholds=domain_thresholds, postprocess_cfg=postprocess_cfg,
        use_postprocess=not args.no_postprocess, num_vis_per_organ=args.num_vis_per_organ,
        vis_data_indices=vis_data_indices, guidance_scale=effective_guidance_scale,
    )

if __name__ == "__main__":
    main()

