# tasksegment/inference.py
from __future__ import annotations
import csv, json, os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .configs import DEFAULT_DOMAIN_THRESHOLDS, DEFAULT_MODEL_CONFIG, DEFAULT_POSTPROCESS_CFG
from .data.datasets import MultiOrganDataset
from .models.segmentation_model import TaskSegmentModel

# 🌟 统一从 baselines 导入所有基线模型
from .models.baselines import StandardUNet, UNetPlusPlus, NNUNetLike, SwinUnet, TransUNet

from .training.evaluation import logits_to_postprocessed_binary_mask, get_ema_task_tokens
from .training.metrics import binary_dice, binary_hd95, binary_iou
from .training.memory import EMATaskMemoryBank

def _safe_torch_load(path: str, map_location: Union[str, torch.device]) -> object:
    try: return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError: return torch.load(path, map_location=map_location)

def _validate_checkpoint_schema(checkpoint: object, model_path: str) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict): raise ValueError(f"checkpoint schema 错误: {model_path} 不是 dict。")
    if "model_state_dict" not in checkpoint: raise ValueError(f"checkpoint schema 错误: {model_path} 缺少 model_state_dict。")
    return checkpoint

def _debug_value_to_jsonable(value: Any) -> Any:
    if value is None: return None
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return float(value.item()) if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray): return float(value.item()) if value.size == 1 else value.tolist()
    if isinstance(value, (list, tuple)): return [_debug_value_to_jsonable(v) for v in value]
    if isinstance(value, dict): return {str(k): _debug_value_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (int, float, str, bool)): return value
    return str(value)

def _safe_json_dumps(value: Any) -> str: return json.dumps(_debug_value_to_jsonable(value), ensure_ascii=False)

def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames: fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _save_case_rankings(save_dir: str, case_records: List[Dict[str, Any]], top_k: int = 20) -> None:
    _write_csv(os.path.join(save_dir, "per_case_metrics.csv"), case_records)
    organs = sorted({str(row["organ"]) for row in case_records})
    for organ in organs:
        sub = [row for row in case_records if row["organ"] == organ]
        _write_csv(os.path.join(save_dir, f"{organ}_worst_dice_top{top_k}.csv"), sorted(sub, key=lambda r: float(r["dice"]))[:top_k])
        _write_csv(os.path.join(save_dir, f"{organ}_worst_hd95_top{top_k}.csv"), sorted(sub, key=lambda r: float(r["hd95"]), reverse=True)[:top_k])
        _write_csv(os.path.join(save_dir, f"{organ}_largest_fp_top{top_k}.csv"), sorted(sub, key=lambda r: int(r["fp_area"]), reverse=True)[:top_k])
        _write_csv(os.path.join(save_dir, f"{organ}_largest_fn_top{top_k}.csv"), sorted(sub, key=lambda r: int(r["fn_area"]), reverse=True)[:top_k])
    _write_csv(os.path.join(save_dir, f"global_worst_dice_top{top_k}.csv"), sorted(case_records, key=lambda r: float(r["dice"]))[:top_k])
    _write_csv(os.path.join(save_dir, f"global_worst_hd95_top{top_k}.csv"), sorted(case_records, key=lambda r: float(r["hd95"]), reverse=True)[:top_k])

def denormalize_for_vis(img_tensor: torch.Tensor) -> np.ndarray:
    img_np = img_tensor.squeeze().detach().cpu().numpy()
    img_np = ((img_np + 1.0) * 127.5).clip(0, 255)
    return img_np.astype(np.uint8)

def to_heatmap_np(x: torch.Tensor) -> Optional[np.ndarray]:
    if x is None: return None
    x = torch.sigmoid(x).squeeze().detach().cpu().numpy()
    x = np.nan_to_num(x)
    return np.clip(x, 0.0, 1.0)

def load_model(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, Optional[EMATaskMemoryBank], Dict[str, Any], bool]:
    checkpoint = _validate_checkpoint_schema(_safe_torch_load(model_path, map_location=device), model_path)
    train_config = checkpoint.get("train_config", {})
    is_baseline = train_config.get("arch") == "baseline"
    model_config_saved = checkpoint.get("model_config", {})
    
    if is_baseline:
        # 从保存的模型配置中精准获取架构名称
        arch_name = model_config_saved.get("arch", "unet")
        
        if arch_name == "unet": 
            model = StandardUNet().to(device)
        elif arch_name == "unetpp": 
            model = UNetPlusPlus().to(device)
        elif arch_name == "nnunet": 
            model = NNUNetLike().to(device)
        elif arch_name == "swinunet":
            img_size_h = train_config.get("image_size", (512, 512))[0]
            model = SwinUnet(in_channels=1, num_classes=2, img_size=img_size_h).to(device)
        elif arch_name == "transunet":
            img_size_h = train_config.get("image_size", (512, 512))[0]
            model = TransUNet(in_channels=1, num_classes=2, img_size=img_size_h).to(device)
        else: 
            raise ValueError(f"Unknown baseline arch: {arch_name}")
            
        ema_bank = None
        model_config = {"arch": arch_name}
    else:
        cfg = dict(DEFAULT_MODEL_CONFIG)
        cfg.update(model_config_saved)
        cfg.pop("decoder_num_classes", None)
        model = TaskSegmentModel(**cfg).to(device)
        ema_bank = EMATaskMemoryBank()
        if "ema_memory_bank" in checkpoint: ema_bank.load_state_dict(checkpoint["ema_memory_bank"])
        model_config = cfg

    compatible_state = {}
    model_state = model.state_dict()
    loaded_keys_count = 0
    for key, value in checkpoint["model_state_dict"].items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible_state[key] = value
            loaded_keys_count += 1
            
    # 打印加载成功率供排查问题
    print(f"[{model_config.get('arch', 'ours')}] 已成功加载参数数量: {loaded_keys_count} / {len(model_state)}")
            
    model.load_state_dict(compatible_state, strict=False)
    model.eval()
    
    return model, ema_bank, train_config, is_baseline

def save_interpretability_panel(save_path: str, img_np: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray, out_dict: Dict[str, torch.Tensor], dice_val: float, hd95_val: Optional[float] = None) -> None:
    fg_prob = torch.softmax(out_dict["pred_masks"], dim=1)[:, 1].squeeze(0).detach().cpu().numpy()
    gt_bool, pred_bool = gt_np.astype(bool), pred_np.astype(bool)
    err = np.zeros_like(gt_np, dtype=np.uint8)
    err[np.logical_and(pred_bool, gt_bool)] = 1
    err[np.logical_and(pred_bool, ~gt_bool)] = 2
    err[np.logical_and(~pred_bool, gt_bool)] = 3

    panels = [
        ("Image", img_np, "gray"), ("GT", gt_np, "gray"), ("Pred", pred_np, "gray"),
        ("Error 1=TP 2=FP 3=FN", err, "viridis"), ("FG prob", fg_prob, "viridis"),
        ("Guidance", to_heatmap_np(out_dict.get("dense_guidance_map")), "magma"),
    ]
    n = len(panels)
    plt.figure(figsize=(3.2 * n, 3.4))
    for i, (title, arr, cmap) in enumerate(panels, 1):
        plt.subplot(1, n, i)
        if arr is None: arr = np.zeros_like(gt_np, dtype=np.float32)
        plt.imshow(arr, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    hd95_text = "nan" if hd95_val is None else f"{hd95_val:.2f}"
    plt.suptitle(f"Dice={dice_val:.3f} | HD95={hd95_text}", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def predict_all_organs(
    model: torch.nn.Module, test_dataset: MultiOrganDataset, device: torch.device,
    text_bank: Dict[str, torch.Tensor], ema_bank: Optional[EMATaskMemoryBank], 
    is_baseline: bool = False,
    save_dir: str = "./vis", domain_thresholds: Optional[Dict[str, float]] = None,
    postprocess_cfg: Optional[Dict[str, Union[int, bool]]] = None, use_postprocess: bool = True,
    num_vis_per_organ: int = 5, vis_data_indices: Optional[List[int]] = None, guidance_scale: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    del text_bank
    os.makedirs(save_dir, exist_ok=True)
    if not is_baseline and guidance_scale is not None:
        if hasattr(model, "decoder") and hasattr(model.decoder, "dense_guidance_strength"):
            model.decoder.dense_guidance_strength = float(guidance_scale)

    if domain_thresholds is None: domain_thresholds = dict(DEFAULT_DOMAIN_THRESHOLDS)
    if postprocess_cfg is None: postprocess_cfg = dict(DEFAULT_POSTPROCESS_CFG)
    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    case_records: List[Dict[str, Any]] = []
    vis_data_idx_set = set(int(x) for x in (vis_data_indices or []))

    with torch.inference_mode():
        for organ in list(test_dataset.organ_to_indices.keys()):
            organ_dir = os.path.join(save_dir, organ)
            os.makedirs(organ_dir, exist_ok=True)
            indices = test_dataset.get_organ_samples(organ, num_samples=len(test_dataset.organ_to_indices[organ]))
            dices, ious, hd95s = [], [], []
            for sample_i, data_idx in enumerate(tqdm(indices, desc=f"测试 {organ}")):
                img, mask, _ = test_dataset[data_idx]
                xq = img.unsqueeze(0).to(device)
                yq = mask.unsqueeze(0).long().to(device)

                if not is_baseline:
                    query_pyramid = model.encode_query(xq)
                    task_tokens = get_ema_task_tokens(ema_bank, organ, device)
                    out = model.segment_with_task(query_feats=query_pyramid, task_tokens=task_tokens, output_size=xq.shape[-2:])
                else:
                    out = model.segment_with_task(xq=xq, output_size=xq.shape[-2:])

                logits = out["pred_masks"]
                pred = logits_to_postprocessed_binary_mask(logits, organ_name=organ, domain_thresholds=domain_thresholds, postprocess_cfg=postprocess_cfg, use_postprocess=use_postprocess)
                target = yq.squeeze(0).detach().cpu().numpy().astype(np.uint8)
                
                dice_val, iou_val, hd95_val = binary_dice(pred, target), binary_iou(pred, target), binary_hd95(pred, target)
                dices.append(dice_val); ious.append(iou_val); hd95s.append(hd95_val)
                
                # 🌟 计算 FP (假阳性) 和 FN (假阴性)
                gt_bool = target.astype(bool)
                pred_bool = pred.astype(bool)
                fp_area = int(np.logical_and(pred_bool, ~gt_bool).sum())
                fn_area = int(np.logical_and(~pred_bool, gt_bool).sum())
                
                case_records.append({
                    "organ": organ, 
                    "sample_i": int(sample_i), 
                    "data_idx": int(data_idx), 
                    "dice": float(dice_val), 
                    "iou": float(iou_val), 
                    "hd95": float(hd95_val),
                    "fp_area": fp_area,
                    "fn_area": fn_area
                })

                should_save_default_vis = sample_i < num_vis_per_organ
                should_save_target_vis = int(data_idx) in vis_data_idx_set
                if should_save_default_vis or should_save_target_vis:
                    file_name = f"{organ}_dataidx_{int(data_idx):04d}_dice_{dice_val:.3f}.png" if should_save_target_vis else f"{organ}_{sample_i:03d}.png"
                    save_interpretability_panel(os.path.join(organ_dir, file_name), img_np=denormalize_for_vis(xq), gt_np=target, pred_np=pred, out_dict=out, dice_val=dice_val, hd95_val=hd95_val)

            results[organ] = {"dice": float(np.mean(dices)), "iou": float(np.mean(ious)), "hd95": float(np.mean(hd95s))}
            print(f"[{organ}] Dice={results[organ]['dice']:.4f} IoU={results[organ]['iou']:.4f} HD95={results[organ]['hd95']:.2f}")

    _save_case_rankings(save_dir, case_records, top_k=20)
    return results