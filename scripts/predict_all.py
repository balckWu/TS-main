import os
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="一键运行所有对比实验的预测")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints", help="模型权重所在的文件夹路径")
    parser.add_argument("--save-base", type=str, default=".", help="可视化结果存放的根目录")
    parser.add_argument("--data-root", type=str, default="./data", help="如果 predict.py 需要数据集路径，可以在这里统一配置")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    save_base = Path(args.save_base)

    if not ckpt_dir.exists():
        print(f"❌ 错误: 找不到权重目录 {ckpt_dir.resolve()}")
        return

    # 1. 定义你要跑的基线模型和对应的权重文件名
    # 只要在 checkpoints 目录下有这些文件，就会自动执行
    models_to_run = {
        "ours": "ours.pth",
        "unet": "unet.pth",
        "unetpp": "unetpp.pth",
        "nnunet": "nnunet.pth",
        "swinunet": "swinunet.pth",   # 如果有这些 transformer 基线也可以加上
        "transunet": "transunet.pth"
    }

    print("🚀 开始批量执行对比实验预测...")
    print("=" * 60)

    success_list = []
    fail_list = []

    # 2. 遍历执行
    for model_name, ckpt_name in models_to_run.items():
        ckpt_path = ckpt_dir / ckpt_name
        
        if not ckpt_path.exists():
            print(f"⚠️ 找不到 {model_name} 的权重文件: {ckpt_path}，已自动跳过。")
            continue
            
        save_dir = save_base / f"vis_{model_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 [正在预测] 模型: {model_name}")
        print(f"   📂 权重: {ckpt_path}")
        print(f"   💾 输出: {save_dir}")
        
        # 3. 构造子进程命令 (这里的参数名请确保与你的 predict.py 对应)
        cmd = [
            "python", "scripts/predict.py",
            "--model-path", str(ckpt_path),
            "--save-dir", str(save_dir),
            # 如果你的 predict.py 还需要其他参数，可以直接加在这里：
            # "--data-root", args.data_root,
            # "--device", args.device
        ]
        
        # 4. 执行并捕获状态
        try:
            # 运行命令，将输出实时打印到控制台
            subprocess.run(cmd, check=True)
            print(f"✅ [{model_name}] 预测完成！")
            success_list.append(model_name)
        except subprocess.CalledProcessError as e:
            print(f"❌ [{model_name}] 预测失败，子进程返回错误码: {e.returncode}")
            fail_list.append(model_name)
            
    # 5. 打印最终汇总报告
    print("\n" + "=" * 60)
    print("🎉 批量预测任务执行完毕！汇总报告：")
    print(f"✅ 成功 ({len(success_list)}): {', '.join(success_list) if success_list else '无'}")
    if fail_list:
        print(f"❌ 失败 ({len(fail_list)}): {', '.join(fail_list)}")

if __name__ == "__main__":
    main()