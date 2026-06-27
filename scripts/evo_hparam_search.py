from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch


ROOT = Path(__file__).resolve().parents[1]


SEARCH_SPACE = {
    "lr": [1e-5, 2e-5, 5e-5, 1e-4],
    "weight_decay": [0.0, 1e-5, 5e-5, 1e-4, 5e-4],
    "warmup_epochs": [1, 2, 3, 5],
    "grad_accum_steps": [2, 4, 8],

    "ema_bank_momentum": [0.90, 0.95, 0.97, 0.99, 0.995],

    "boundary_loss_weight": [0.1, 0.3, 0.5, 0.7, 1.0],
    "dense_main_weight": [0.1, 0.3, 0.5, 0.7, 1.0],

    "guidance_dropout_prob": [0.0, 0.1, 0.15, 0.2, 0.3],
    "guidance_scale_min": [0.3, 0.5, 0.7],
    "guidance_scale_max": [0.8, 1.0, 1.2],
    "guidance_clamp_max": [0.6, 0.8, 1.0],

    "train_query_positive_ratio": [0.5, 0.6, 0.7, 0.8],

    "task_importance_dice_weight": [0.4, 0.5, 0.6, 0.7],
    "task_importance_confidence_weight": [0.1, 0.2, 0.25, 0.3],
    "task_importance_support_size_weight": [0.05, 0.1, 0.15, 0.2],
    "task_importance_support_size_norm": [0.03, 0.05, 0.08, 0.10],
    "task_importance_min": [0.01, 0.03, 0.05, 0.08],
}


def sample_individual() -> Dict[str, Any]:
    """随机生成一个超参数个体。"""
    ind = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}

    # 保证 dense guidance 的 scale_min <= scale_max
    if ind["guidance_scale_min"] > ind["guidance_scale_max"]:
        ind["guidance_scale_min"], ind["guidance_scale_max"] = (
            ind["guidance_scale_max"],
            ind["guidance_scale_min"],
        )

    # task importance 三个权重可以不强制和为 1，因为你的代码只是加权使用。
    return ind


def crossover(parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
    """两个父代配置交叉生成子代。"""
    child = {}
    for key in SEARCH_SPACE.keys():
        child[key] = parent_a[key] if random.random() < 0.5 else parent_b[key]

    if child["guidance_scale_min"] > child["guidance_scale_max"]:
        child["guidance_scale_min"], child["guidance_scale_max"] = (
            child["guidance_scale_max"],
            child["guidance_scale_min"],
        )
    return child


def mutate(ind: Dict[str, Any], mutation_prob: float = 0.25) -> Dict[str, Any]:
    """按概率随机改变部分超参数。"""
    new_ind = dict(ind)
    for key, values in SEARCH_SPACE.items():
        if random.random() < mutation_prob:
            new_ind[key] = random.choice(values)

    if new_ind["guidance_scale_min"] > new_ind["guidance_scale_max"]:
        new_ind["guidance_scale_min"], new_ind["guidance_scale_max"] = (
            new_ind["guidance_scale_max"],
            new_ind["guidance_scale_min"],
        )
    return new_ind


def individual_to_args(ind: Dict[str, Any]) -> List[str]:
    """把超参数字典转成 scripts/train.py 可接收的命令行参数。"""
    return [
        "--lr", str(ind["lr"]),
        "--weight-decay", str(ind["weight_decay"]),
        "--warmup-epochs", str(ind["warmup_epochs"]),
        "--grad-accum-steps", str(ind["grad_accum_steps"]),

        "--ema-bank-momentum", str(ind["ema_bank_momentum"]),

        "--boundary-loss-weight", str(ind["boundary_loss_weight"]),
        "--dense-main-weight", str(ind["dense_main_weight"]),

        "--guidance-dropout-prob", str(ind["guidance_dropout_prob"]),
        "--guidance-scale-min", str(ind["guidance_scale_min"]),
        "--guidance-scale-max", str(ind["guidance_scale_max"]),
        "--guidance-clamp-max", str(ind["guidance_clamp_max"]),

        "--train-query-positive-ratio", str(ind["train_query_positive_ratio"]),

        "--task-importance-dice-weight", str(ind["task_importance_dice_weight"]),
        "--task-importance-confidence-weight", str(ind["task_importance_confidence_weight"]),
        "--task-importance-support-size-weight", str(ind["task_importance_support_size_weight"]),
        "--task-importance-support-size-norm", str(ind["task_importance_support_size_norm"]),
        "--task-importance-min", str(ind["task_importance_min"]),
    ]


def read_best_dice_from_checkpoint(ckpt_path: Path) -> float:
    """优先从 checkpoint 中读取 best_val_dice。"""
    if not ckpt_path.exists():
        return -1.0

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return float(ckpt.get("best_val_dice", -1.0))
    except Exception:
        return -1.0


def read_best_dice_from_log(log_path: Path) -> float:
    """如果 checkpoint 读取失败，则从日志中解析 val_dice。"""
    if not log_path.exists():
        return -1.0

    pattern = re.compile(r"val_dice=([0-9.]+)")
    best = -1.0

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for m in pattern.finditer(text):
        best = max(best, float(m.group(1)))

    return best


def run_one_candidate(
    ind: Dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
    candidate_name: str,
) -> Dict[str, Any]:
    """运行一个候选超参数配置。"""
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = run_dir / "best_model.pth"
    log_path = run_dir / "train.log"
    config_path = run_dir / "hparams.json"

    config_path.write_text(
        json.dumps(ind, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),

        "--data-root", args.data_root,
        "--text-dir", args.text_dir,
        "--save-path", str(ckpt_path),

        "--epochs", str(args.proxy_epochs),
        "--episodes-per-epoch", str(args.proxy_episodes_per_epoch),
        "--patience", str(args.proxy_patience),

        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--device", args.device,
        "--seed", str(args.seed),

        "--cache-dataset",
    ]

    if args.no_cpu_clahe:
        cmd.append("--no-cpu-clahe")

    cmd.extend(individual_to_args(ind))

    print(f"\n[RUN] {candidate_name}")
    print(" ".join(cmd))

    with log_path.open("w", encoding="utf-8") as f:
        process = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if process.returncode != 0:
        print(f"[FAILED] {candidate_name}, returncode={process.returncode}")
        fitness = -1.0
    else:
        fitness = read_best_dice_from_checkpoint(ckpt_path)
        if fitness < 0:
            fitness = read_best_dice_from_log(log_path)

    result = {
        "candidate": candidate_name,
        "fitness": fitness,
        "hparams": ind,
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "log_path": str(log_path),
    }

    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[DONE] {candidate_name} | best_val_dice={fitness:.4f}")
    return result


def save_generation_results(results: List[Dict[str, Any]], path: Path):
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser("Evolutionary hyperparameter search for TaskSegment")

    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--text-dir", type=str, default="./text_features_llm")
    parser.add_argument("--search-dir", type=str, default="./runs/evo_hparam_search")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)

    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--elite-size", type=int, default=2)
    parser.add_argument("--mutation-prob", type=float, default=0.25)

    parser.add_argument("--proxy-epochs", type=int, default=20)
    parser.add_argument("--proxy-episodes-per-epoch", type=int, default=300)
    parser.add_argument("--proxy-patience", type=int, default=5)

    parser.add_argument("--no-cpu-clahe", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)

    search_dir = ROOT / args.search_dir
    search_dir.mkdir(parents=True, exist_ok=True)

    population = [sample_individual() for _ in range(args.population_size)]
    all_results: List[Dict[str, Any]] = []

    for gen in range(args.generations):
        print("\n" + "=" * 80)
        print(f"Generation {gen + 1}/{args.generations}")
        print("=" * 80)

        gen_results = []

        for i, ind in enumerate(population):
            candidate_name = f"gen{gen:02d}_cand{i:02d}"
            run_dir = search_dir / candidate_name

            result = run_one_candidate(
                ind=ind,
                run_dir=run_dir,
                args=args,
                candidate_name=candidate_name,
            )
            gen_results.append(result)
            all_results.append(result)

        gen_results = sorted(gen_results, key=lambda x: x["fitness"], reverse=True)

        save_generation_results(
            gen_results,
            search_dir / f"generation_{gen:02d}_results.json",
        )
        save_generation_results(
            sorted(all_results, key=lambda x: x["fitness"], reverse=True),
            search_dir / "all_results_sorted.json",
        )

        print("\n[Generation Ranking]")
        for rank, r in enumerate(gen_results, start=1):
            print(f"#{rank}: {r['candidate']} | fitness={r['fitness']:.4f}")

        elites = gen_results[: args.elite_size]
        elite_hparams = [r["hparams"] for r in elites]

        # 生成下一代
        next_population = []

        # 精英保留
        next_population.extend(elite_hparams)

        # 交叉 + 变异
        while len(next_population) < args.population_size:
            p1, p2 = random.sample(elite_hparams, k=2) if len(elite_hparams) >= 2 else (elite_hparams[0], elite_hparams[0])
            child = crossover(p1, p2)
            child = mutate(child, mutation_prob=args.mutation_prob)
            next_population.append(child)

        population = next_population

    final_sorted = sorted(all_results, key=lambda x: x["fitness"], reverse=True)
    save_generation_results(final_sorted, search_dir / "final_results_sorted.json")

    print("\n" + "=" * 80)
    print("Search Finished")
    print("=" * 80)

    print("\nTop 5 configurations:")
    for rank, r in enumerate(final_sorted[:5], start=1):
        print(f"\n#{rank} | fitness={r['fitness']:.4f}")
        print(json.dumps(r["hparams"], indent=2, ensure_ascii=False))
        print(f"run_dir: {r['run_dir']}")


if __name__ == "__main__":
    main()