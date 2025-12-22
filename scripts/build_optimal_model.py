#!/usr/bin/env python3
"""
根据验证曲线选择最佳轮次，构造最优模型

用法:
    python scripts/build_optimal_model.py \
        --experiment compare_hog \
        --output-name compare_hog_optimal
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier


def find_best_round(monitor):
    """
    从监控数据中找出最佳轮次
    
    Parameters
    ----------
    monitor : BoostMonitor
        训练监控对象
    
    Returns
    -------
    int
        最佳轮次索引（从0开始）
    float
        最佳验证准确率
    """
    if not monitor.val_acc_history:
        raise ValueError("监控数据中没有验证准确率历史！")
    
    val_acc = np.array(monitor.val_acc_history)
    best_idx = np.argmax(val_acc)
    best_acc = val_acc[best_idx]
    
    # val_idx 记录了验证发生的实际轮次
    if monitor.val_idx and len(monitor.val_idx) > 0:
        actual_round = monitor.val_idx[best_idx]
    else:
        actual_round = best_idx + 1
    
    return actual_round, best_acc, best_idx


def build_truncated_model(full_model, n_estimators):
    """
    从完整模型构造截断模型（只使用前 n 个弱学习器）
    
    Parameters
    ----------
    full_model : AdaBoostClassifier
        完整训练的模型
    n_estimators : int
        要保留的弱学习器数量
    
    Returns
    -------
    AdaBoostClassifier
        截断后的模型
    """
    if n_estimators > len(full_model.estimators_):
        raise ValueError(
            f"指定轮次 {n_estimators} 超过模型总轮次 {len(full_model.estimators_)}"
        )
    
    # 创建新模型（复制参数）
    optimal_model = AdaBoostClassifier(
        estimator=full_model.estimator,
        n_estimators=n_estimators,
        learning_rate=full_model.learning_rate,
        random_state=full_model.random_state,
    )
    
    # 手动设置训练后的属性
    optimal_model.estimators_ = full_model.estimators_[:n_estimators]
    optimal_model.estimator_weights_ = np.array(full_model.estimator_weights_[:n_estimators])
    optimal_model.estimator_errors_ = np.array(full_model.estimator_errors_[:n_estimators])
    optimal_model.classes_ = full_model.classes_
    optimal_model.n_classes_ = full_model.n_classes_
    optimal_model.n_features_in_ = full_model.n_features_in_
    
    return optimal_model


def main():
    parser = argparse.ArgumentParser(
        description='根据验证曲线构造最优轮次的模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 compare_hog 构造最优模型
  python scripts/build_optimal_model.py \\
      --experiment compare_hog \\
      --output-name compare_hog_optimal
  
  # 手动指定轮次
  python scripts/build_optimal_model.py \\
      --experiment compare_hog \\
      --round 250 \\
      --output-name compare_hog_r250
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=True,
        help='实验名称（必需有完整的训练结果）'
    )
    
    parser.add_argument(
        '--round', '-r',
        type=int,
        help='手动指定轮次（如不指定，自动选择验证准确率最高的轮次）'
    )
    
    parser.add_argument(
        '--output-name', '-o',
        type=str,
        help='输出实验名称（默认为 <experiment>_optimal）'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='experiments',
        help='实验根目录（默认: experiments）'
    )
    
    args = parser.parse_args()
    
    exp_dir = Path(args.base_dir) / args.experiment
    
    if not exp_dir.exists():
        print(f"❌ 实验目录不存在: {exp_dir}")
        return 1
    
    print("\n" + "=" * 70)
    print("构造最优轮次模型".center(70))
    print("=" * 70)
    print(f"\n📂 加载实验: {args.experiment}")
    
    # 加载监控数据
    monitor_path = exp_dir / "results" / "monitor.joblib"
    if not monitor_path.exists():
        print(f"❌ 找不到监控数据: {monitor_path}")
        return 1
    
    monitor = joblib.load(monitor_path)
    print(f"✓ 已加载监控数据")
    
    # 加载完整模型
    model_path = exp_dir / "results" / "model.joblib"
    if not model_path.exists():
        print(f"❌ 找不到模型文件: {model_path}")
        return 1
    
    full_model = joblib.load(model_path)
    total_rounds = len(full_model.estimators_)
    print(f"✓ 已加载完整模型（总轮次: {total_rounds}）")
    
    # 确定最优轮次
    if args.round:
        optimal_round = args.round
        if optimal_round > total_rounds:
            print(f"❌ 指定轮次 {optimal_round} 超过总轮次 {total_rounds}")
            return 1
        
        # 找到对应的验证准确率
        val_idx = monitor.val_idx if monitor.val_idx else list(range(1, len(monitor.val_acc_history)+1))
        try:
            idx_in_val = val_idx.index(optimal_round)
            optimal_acc = monitor.val_acc_history[idx_in_val]
        except (ValueError, IndexError):
            print(f"⚠️  第 {optimal_round} 轮没有验证数据，使用手动指定")
            optimal_acc = None
        
        print(f"\n🎯 使用手动指定轮次: {optimal_round}")
    else:
        optimal_round, optimal_acc, best_idx = find_best_round(monitor)
        print(f"\n🎯 自动选择最佳轮次: {optimal_round}")
    
    # 显示性能对比
    print("\n" + "=" * 70)
    print("性能对比".center(70))
    print("=" * 70)
    
    if optimal_acc is not None:
        print(f"\n最优轮次 (第 {optimal_round} 轮):")
        print(f"  验证准确率: {optimal_acc:.4f}")
    
    final_acc = monitor.val_acc_history[-1] if monitor.val_acc_history else None
    if final_acc is not None:
        print(f"\n完整模型 (第 {total_rounds} 轮):")
        print(f"  验证准确率: {final_acc:.4f}")
        
        if optimal_acc is not None:
            improvement = optimal_acc - final_acc
            if improvement > 0:
                print(f"\n✨ 使用最优轮次可提升: +{improvement:.4f} ({improvement*100:.2f}%)")
            elif improvement < 0:
                print(f"\n⚠️  最优轮次性能略低: {improvement:.4f}")
            else:
                print(f"\n= 性能相同")
    
    # 构造最优模型
    print("\n" + "=" * 70)
    print(f"🔨 构造前 {optimal_round} 轮的模型...")
    
    optimal_model = build_truncated_model(full_model, optimal_round)
    
    # 创建输出目录
    output_name = args.output_name or f"{args.experiment}_optimal"
    output_dir = Path(args.base_dir) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 保存最优模型
    output_model_path = results_dir / "model.joblib"
    joblib.dump(optimal_model, output_model_path)
    print(f"✓ 模型已保存: {output_model_path}")
    
    # 保存配置信息
    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        # 更新配置
        config["experiment"]["name"] = output_name
        config["model"]["n_estimators"] = optimal_round
        
        output_config_path = output_dir / "config.json"
        with open(output_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ 配置已保存: {output_config_path}")
    
    # 保存元信息
    meta_info = {
        "source_experiment": args.experiment,
        "optimal_round": int(optimal_round),
        "total_rounds": int(total_rounds),
        "optimal_val_acc": float(optimal_acc) if optimal_acc else None,
        "final_val_acc": float(final_acc) if final_acc else None,
        "improvement": float(optimal_acc - final_acc) if (optimal_acc and final_acc) else None,
    }
    
    meta_path = results_dir / "optimal_model_info.json"
    with open(meta_path, 'w') as f:
        json.dump(meta_info, f, indent=2)
    print(f"✓ 元信息已保存: {meta_path}")
    
    print("\n" + "=" * 70)
    print("✅ 最优模型构造完成！".center(70))
    print("=" * 70)
    
    print(f"\n📦 输出位置: {output_dir}")
    print(f"   - 模型: {output_model_path}")
    print(f"   - 配置: {output_config_path if config_path.exists() else 'N/A'}")
    print(f"   - 元信息: {meta_path}")
    
    print(f"\n💡 下一步:")
    print(f"   1. 测试最优模型的泛化能力")
    print(f"   2. 与完整模型对比性能")
    print(f"   3. 在测试集上评估")
    
    print("\n" + "=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())


