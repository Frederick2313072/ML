#!/bin/bash
# 实验3第一步：寻找最佳分类器的完整流程
# 
# 用法：bash scripts/run_experiment3_step1.sh

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "  实验3-步骤1: 寻找最佳分类器"
echo "========================================================================"
echo ""

# 检查是否已有实验结果
EXPS=("compare_original" "compare_hog" "compare_hu")
ALL_EXIST=true

for exp in "${EXPS[@]}"; do
    if [ ! -f "experiments/$exp/results/monitor.joblib" ]; then
        ALL_EXIST=false
        break
    fi
done

if [ "$ALL_EXIST" = true ]; then
    echo "✓ 检测到所有实验结果已存在"
    echo ""
    read -p "是否跳过训练，直接进行对比分析？[Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        echo "跳过训练步骤..."
        SKIP_TRAINING=true
    else
        SKIP_TRAINING=false
    fi
else
    SKIP_TRAINING=false
fi

# 步骤1: 运行三种特征的对比实验
if [ "$SKIP_TRAINING" = false ]; then
    echo "========================================================================"
    echo "  步骤1: 运行对比实验（3个实验，预计20-40分钟）"
    echo "========================================================================"
    echo ""
    
    echo "[1/3] 运行 compare_original (原始像素特征)..."
    PYTHONPATH=src python src/adalab/cli/main.py --config configs/compare_original.json
    echo "✓ compare_original 完成"
    echo ""
    
    echo "[2/3] 运行 compare_hog (HOG特征)..."
    PYTHONPATH=src python src/adalab/cli/main.py --config configs/compare_hog.json
    echo "✓ compare_hog 完成"
    echo ""
    
    echo "[3/3] 运行 compare_hu (HU矩特征)..."
    PYTHONPATH=src python src/adalab/cli/main.py --config configs/compare_hu.json
    echo "✓ compare_hu 完成"
    echo ""
fi

# 步骤2: 对比实验结果
echo "========================================================================"
echo "  步骤2: 对比分析实验结果"
echo "========================================================================"
echo ""

mkdir -p outputs/figures

python scripts/compare_classifiers.py \
    --experiments compare_original compare_hog compare_hu \
    --save outputs/figures/exp3_step1_feature_comparison.png

echo ""
echo "========================================================================"
echo "  实验3-步骤1 完成！"
echo "========================================================================"
echo ""
echo "✓ 对比图表已保存至: outputs/figures/exp3_step1_feature_comparison.png"
echo "✓ 对比数据已保存至: outputs/figures/exp3_step1_feature_comparison_comparison.csv"
echo ""
echo "💡 下一步建议："
echo "   1. 查看对比结果，确定最佳特征类型"
echo "   2. 测试最佳模型的泛化能力（test_shift）"
echo "   3. 检查是否存在过拟合"
echo ""
echo "详细指南请查看: docs/experiment3_best_classifier_guide.md"
echo ""

