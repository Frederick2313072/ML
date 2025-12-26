#!/bin/bash
# 批量重新生成所有特征鲁棒性可视化

# 设置工作目录
cd "$(dirname "$0")/.." || exit 1

# 设置Python路径
export PYTHONPATH=src:$PYTHONPATH

echo "========================================================"
echo "   批量生成特征鲁棒性可视化"
echo "========================================================"
echo ""

# 默认参数
DIGIT="${1:-8}"
SAMPLES="${2:-200}"

echo "📋 运行参数:"
echo "  数字类别: $DIGIT"
echo "  样本数: $SAMPLES"
echo ""

# 配置文件列表
CONFIGS=("original" "hog" "hu")

# 计数器
SUCCESS=0
FAILED=0

# 循环生成
for config in "${CONFIGS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 处理: compare_${config}.json"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python scripts/visualization/visualize_feature_robustness.py \
        --config "configs/compare/compare_${config}.json" \
        --output "outputs/figures/robustness_${config}.png" \
        --digit "$DIGIT" \
        --samples "$SAMPLES" 2>&1 | grep -E "^(📄|✓|🔄|🔧|📊|✅|  -|  处理:)"
    
    if [ $? -eq 0 ]; then
        ((SUCCESS++))
        echo "✅ 成功: robustness_${config}.png"
    else
        ((FAILED++))
        echo "❌ 失败: compare_${config}.json"
    fi
    echo ""
done

echo "========================================================"
echo "📊 生成完成！"
echo "========================================================"
echo "✅ 成功: $SUCCESS 个"
if [ $FAILED -gt 0 ]; then
    echo "❌ 失败: $FAILED 个"
fi
echo ""
echo "📁 输出目录: outputs/figures/"
echo "  - robustness_original.png"
echo "  - robustness_hog.png"
echo "  - robustness_hu.png"
echo ""

# 更新默认图表
if [ -f "outputs/figures/robustness_hog.png" ]; then
    cp outputs/figures/robustness_hog.png outputs/figures/feature_robustness.png
    echo "✓ 已更新默认图表: feature_robustness.png"
fi

echo "========================================================"



