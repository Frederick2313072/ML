#!/bin/bash
# HOG 特征聚类可视化 - 一键运行脚本

set -e

echo "========================================================================"
echo "  HOG 特征聚类可视化"
echo "========================================================================"
echo ""

# 创建输出目录
mkdir -p outputs/figures

# 1. 基本 2D 可视化（PCA）
echo "[1/4] 生成 2D 聚类图（PCA降维）..."
python scripts/visualization/visualize_hog_clustering.py \
    --reduction pca \
    --classes 0 1 8 \
    --samples-per-class 150 \
    --output outputs/figures/hog_clustering_pca.png

echo "✓ 完成: outputs/figures/hog_clustering_pca.png"
echo ""

# 2. 2D 可视化（t-SNE）
echo "[2/4] 生成 2D 聚类图（t-SNE降维，计算较慢）..."
python scripts/visualization/visualize_hog_clustering.py \
    --reduction tsne \
    --classes 0 1 8 \
    --samples-per-class 100 \
    --output outputs/figures/hog_clustering_tsne.png

echo "✓ 完成: outputs/figures/hog_clustering_tsne.png"
echo ""

# 3. 3D 可视化
echo "[3/4] 生成 3D 聚类图..."
python scripts/visualization/visualize_hog_clustering_3d.py \
    --classes 0 1 8 \
    --samples-per-class 150 \
    --output outputs/figures/hog_clustering_3d.png

echo "✓ 完成: outputs/figures/hog_clustering_3d.png"
echo ""

# 4. 更多类别（5个类别）
echo "[4/4] 生成5类别聚类图..."
python scripts/visualization/visualize_hog_clustering.py \
    --reduction pca \
    --classes 0 1 2 3 4 \
    --samples-per-class 100 \
    --output outputs/figures/hog_clustering_5class.png

echo "✓ 完成: outputs/figures/hog_clustering_5class.png"
echo ""

echo "========================================================================"
echo "  ✅ 所有可视化完成！"
echo "========================================================================"
echo ""
echo "生成的图表："
echo "  1. outputs/figures/hog_clustering_pca.png      - 2D PCA 聚类"
echo "  2. outputs/figures/hog_clustering_tsne.png     - 2D t-SNE 聚类"
echo "  3. outputs/figures/hog_clustering_3d.png       - 3D PCA 聚类"
echo "  4. outputs/figures/hog_clustering_5class.png   - 5类别 PCA 聚类"
echo ""
echo "💡 查看详细说明: docs/hog_clustering_visualization_guide.md"
echo ""


