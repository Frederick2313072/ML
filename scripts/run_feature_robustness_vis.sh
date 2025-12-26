#!/bin/bash
# 特征鲁棒性可视化运行脚本

# 设置工作目录
cd "$(dirname "$0")/.." || exit 1

# 设置Python路径
export PYTHONPATH=src:$PYTHONPATH

echo "================================================"
echo "   特征空间鲁棒性可视化"
echo "================================================"
echo ""

# 默认参数
CONFIG="${1:-configs/test_feature_robustness.json}"
OUTPUT="${2:-outputs/figures/feature_robustness.png}"
DIGIT="${3:-8}"
SAMPLES="${4:-200}"

echo "📋 运行参数:"
echo "  配置文件: $CONFIG"
echo "  输出路径: $OUTPUT"
echo "  数字类别: $DIGIT"
echo "  样本数: $SAMPLES"
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 运行可视化
python scripts/visualization/visualize_feature_robustness.py \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --digit "$DIGIT" \
    --samples "$SAMPLES"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ 可视化完成！"
    echo "================================================"
    echo "📊 图表位置: $OUTPUT"
    echo ""
else
    echo ""
    echo "❌ 可视化失败，请检查错误信息"
    exit 1
fi



