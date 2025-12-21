#!/usr/bin/env python3
"""
对比多个实验的性能指标

用法示例:
    # 对比三种特征的效果
    python scripts/compare_classifiers.py \
        --experiments compare_original compare_hog compare_hu \
        --save outputs/figures/classifier_comparison.png
    
    # 对比不同超参数配置
    python scripts/compare_classifiers.py \
        --experiments exp1 exp2 exp3 exp4
"""

import argparse
import sys
from pathlib import Path

# 添加 src 目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from adalab_viz.plotter import compare_experiments


def main():
    parser = argparse.ArgumentParser(
        description='对比多个实验的性能指标，找出最佳分类器配置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 对比三种特征类型
  python scripts/compare_classifiers.py \\
      --experiments compare_original compare_hog compare_hu \\
      --save outputs/figures/feature_comparison.png
  
  # 对比不同深度的决策树
  python scripts/compare_classifiers.py \\
      --experiments depth2_exp depth3_exp depth4_exp depth5_exp
  
  # 对比不同学习率
  python scripts/compare_classifiers.py \\
      --experiments lr05_exp lr08_exp lr10_exp \\
      --save outputs/figures/lr_comparison.png
        """
    )
    
    parser.add_argument(
        '--experiments', '-e',
        nargs='+',
        required=True,
        help='实验名称列表（用空格分隔），如: compare_hog compare_hu compare_original'
    )
    
    parser.add_argument(
        '--save', '-s',
        type=str,
        help='保存图表的路径，如: outputs/figures/comparison.png'
    )
    
    parser.add_argument(
        '--base-dir', '-d',
        type=str,
        default='experiments',
        help='实验根目录，默认为 "experiments"'
    )
    
    args = parser.parse_args()
    
    # 验证实验数量
    if len(args.experiments) < 2:
        print("⚠️  警告: 至少需要2个实验进行对比")
        return 1
    
    print(f"\n开始对比 {len(args.experiments)} 个实验...")
    print(f"实验列表: {', '.join(args.experiments)}\n")
    
    try:
        # 执行对比分析
        df_results = compare_experiments(
            experiment_names=args.experiments,
            save_path=args.save,
            base_dir=args.base_dir
        )
        
        # 保存对比结果到CSV（可选）
        if args.save:
            csv_path = args.save.replace('.png', '_comparison.csv')
            df_results.to_csv(csv_path, index=False)
            print(f"\n✓ 对比数据已保存至: {csv_path}")
        
        print("\n✓ 对比分析完成！")
        
        # 输出使用建议
        best_exp = df_results.loc[df_results['best_val_acc'].idxmax(), 'name']
        print(f"\n💡 建议使用配置: {best_exp}")
        print(f"   下一步可以:")
        print(f"   1. 测试泛化能力（test_shift）")
        print(f"   2. 如有过拟合，根据验证曲线选择最优时间步")
        print(f"   3. 测试噪声鲁棒性\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 对比分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

