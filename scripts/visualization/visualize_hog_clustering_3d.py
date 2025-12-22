#!/usr/bin/env python3
"""
HOG 特征聚类 3D 可视化

功能：
1. 3D 散点图展示特征空间
2. 交互式可视化（可旋转）
3. 对比不同 HOG 参数的聚类效果

用法：
    python scripts/visualization/visualize_hog_clustering_3d.py \
        --output outputs/figures/hog_clustering_3d.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from skimage.feature import hog
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
try:
    from mplfonts.bin.cli import init
    init()
    rcParams['font.family'] = 'Source Han Sans CN'
except ImportError:
    rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_mnist_subset(n_samples_per_class=200, classes=None):
    """加载 MNIST 数据子集"""
    print(f"📂 加载 MNIST 数据...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.to_numpy().reshape(-1, 28, 28) / 255.0
    y = mnist.target.to_numpy().astype(int)
    
    if classes is None:
        classes = [0, 1, 8]
    
    indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        selected = np.random.choice(cls_indices, n_samples_per_class, replace=False)
        indices.extend(selected)
    
    indices = np.array(indices)
    X_subset = X[indices]
    y_subset = y[indices]
    
    print(f"✓ 加载了 {len(classes)} 个类别，每类 {n_samples_per_class} 个样本")
    return X_subset, y_subset


def extract_hog_features(X, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """提取 HOG 特征"""
    features_list = []
    for i in range(len(X)):
        feat = hog(
            X[i],
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        features_list.append(feat)
    return np.array(features_list)


def visualize_3d_clustering(X, y, hog_configs, save_path=None):
    """
    3D 可视化 HOG 特征聚类
    
    Parameters
    ----------
    X : ndarray
        图像数据
    y : ndarray
        标签
    hog_configs : list of dict
        HOG 配置列表
    save_path : str, optional
        保存路径
    """
    n_configs = len(hog_configs)
    
    # 创建子图
    fig = plt.figure(figsize=(6 * n_configs, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    markers = ['o', 's', '^']
    
    for config_idx, config in enumerate(hog_configs):
        print(f"\n处理配置 {config_idx + 1}...")
        
        # 提取 HOG 特征
        features = extract_hog_features(
            X,
            orientations=config['orientations'],
            pixels_per_cell=tuple(config['pixels_per_cell']),
            cells_per_block=tuple(config['cells_per_block'])
        )
        
        print(f"  特征维度: {features.shape[1]}")
        
        # PCA 降维到 3D
        pca = PCA(n_components=3, random_state=42)
        features_3d = pca.fit_transform(features)
        
        explained_var = pca.explained_variance_ratio_
        print(f"  解释方差: {explained_var[0]:.3f}, {explained_var[1]:.3f}, {explained_var[2]:.3f}")
        
        # 绘制 3D 散点图
        ax = fig.add_subplot(1, n_configs, config_idx + 1, projection='3d')
        
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            mask = y == label
            ax.scatter(
                features_3d[mask, 0],
                features_3d[mask, 1],
                features_3d[mask, 2],
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                label=f'类别 {label}',
                alpha=0.6,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )
        
        # 标题和标签
        config_str = (f"HOG 配置 {config_idx + 1}\n"
                     f"orientations={config['orientations']}, "
                     f"ppc={config['pixels_per_cell']}")
        ax.set_title(config_str, fontsize=12, fontweight='bold', pad=20)
        
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=10)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=10)
        ax.set_zlabel(f'PC3 ({explained_var[2]:.1%})', fontsize=10)
        
        ax.legend(loc='upper right', fontsize=9)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 图表已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='HOG 特征聚类 3D 可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/figures/hog_clustering_3d.png',
        help='输出图表路径'
    )
    
    parser.add_argument(
        '--classes', '-c',
        type=int,
        nargs='+',
        default=[0, 1, 8],
        help='要可视化的类别（默认: 0 1 8）'
    )
    
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=200,
        help='每个类别的样本数（默认: 200）'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("HOG 特征聚类 3D 可视化".center(70))
    print("=" * 70)
    
    # 加载数据
    X, y = load_mnist_subset(
        n_samples_per_class=args.samples_per_class,
        classes=args.classes
    )
    
    # 定义 HOG 配置
    print("\n🔧 HOG 配置:")
    hog_configs = [
        {
            'orientations': 9,
            'pixels_per_cell': [4, 4],
            'cells_per_block': [2, 2],
        },
        {
            'orientations': 9,
            'pixels_per_cell': [8, 8],
            'cells_per_block': [2, 2],
        },
        {
            'orientations': 12,
            'pixels_per_cell': [4, 4],
            'cells_per_block': [2, 2],
        },
    ]
    
    for i, config in enumerate(hog_configs):
        print(f"  配置 {i+1}: {config}")
    
    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 可视化
    print(f"\n📊 生成 3D 聚类图...")
    visualize_3d_clustering(X, y, hog_configs, save_path=args.output)
    
    print("\n" + "=" * 70)
    print("✅ 可视化完成！".center(70))
    print("=" * 70)
    
    print(f"\n📊 输出文件: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())


