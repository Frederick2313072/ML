#!/usr/bin/env python3
"""
HOG 特征聚类可视化

功能：
1. 提取不同 HOG 参数配置下的特征
2. 使用降维方法（PCA、t-SNE、UMAP）可视化
3. 对比不同风格数据集的聚类效果
4. 展示不同超参数的影响

用法：
    python scripts/visualization/visualize_hog_clustering.py \
        --output outputs/figures/hog_clustering.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from skimage.feature import hog
from skimage.transform import resize
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
    """
    加载 MNIST 数据子集
    
    Parameters
    ----------
    n_samples_per_class : int
        每个类别的样本数
    classes : list, optional
        要加载的类别列表，如 [0, 1, 8]
    
    Returns
    -------
    X : ndarray, shape (n_samples, 28, 28)
        图像数据
    y : ndarray
        标签
    """
    print(f"📂 加载 MNIST 数据...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.to_numpy().reshape(-1, 28, 28) / 255.0
    y = mnist.target.to_numpy().astype(int)
    
    if classes is None:
        classes = [0, 1, 8]  # 默认选择 0, 1, 8 三个类别
    
    # 选择指定类别
    indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        selected = np.random.choice(cls_indices, n_samples_per_class, replace=False)
        indices.extend(selected)
    
    indices = np.array(indices)
    X_subset = X[indices]
    y_subset = y[indices]
    
    print(f"✓ 加载了 {len(classes)} 个类别，每类 {n_samples_per_class} 个样本")
    print(f"  类别: {classes}")
    
    return X_subset, y_subset


def apply_perturbations(X, style='original'):
    """
    对数据应用不同风格的扰动
    
    Parameters
    ----------
    X : ndarray
        原始图像
    style : str
        扰动风格: 'original', 'bright', 'dark', 'noisy', 'blurred'
    
    Returns
    -------
    X_perturbed : ndarray
        扰动后的图像
    """
    X_perturbed = X.copy()
    
    if style == 'bright':
        # 提高亮度
        X_perturbed = np.clip(X_perturbed + 0.3, 0, 1)
    elif style == 'dark':
        # 降低亮度
        X_perturbed = np.clip(X_perturbed - 0.3, 0, 1)
    elif style == 'noisy':
        # 添加高斯噪声
        noise = np.random.normal(0, 0.1, X_perturbed.shape)
        X_perturbed = np.clip(X_perturbed + noise, 0, 1)
    elif style == 'blurred':
        # 简单模糊（平均池化）
        from scipy.ndimage import uniform_filter
        for i in range(len(X_perturbed)):
            X_perturbed[i] = uniform_filter(X_perturbed[i], size=3)
    
    return X_perturbed


def extract_hog_features(X, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    提取 HOG 特征
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, height, width)
        图像数据
    orientations : int
        HOG 方向数
    pixels_per_cell : tuple
        每个单元格的像素数
    cells_per_block : tuple
        每个块的单元格数
    
    Returns
    -------
    features : ndarray, shape (n_samples, n_features)
        HOG 特征
    """
    n_samples = len(X)
    features_list = []
    
    for i in range(n_samples):
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
    
    features = np.array(features_list)
    return features


def reduce_dimensions(features, method='pca', n_components=2):
    """
    降维
    
    Parameters
    ----------
    features : ndarray
        高维特征
    method : str
        降维方法: 'pca', 'tsne'
    n_components : int
        目标维度
    
    Returns
    -------
    features_reduced : ndarray
        降维后的特征
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    features_reduced = reducer.fit_transform(features)
    return features_reduced


def visualize_clustering(
    X_list,
    y_list,
    style_names,
    hog_configs,
    reduction_method='pca',
    save_path=None
):
    """
    可视化 HOG 特征聚类
    
    Parameters
    ----------
    X_list : list of ndarray
        不同风格的图像数据列表
    y_list : list of ndarray
        对应的标签列表
    style_names : list of str
        风格名称列表
    hog_configs : list of dict
        HOG 配置列表
    reduction_method : str
        降维方法
    save_path : str, optional
        保存路径
    """
    n_configs = len(hog_configs)
    n_styles = len(X_list)
    
    # 创建子图
    fig = plt.figure(figsize=(6 * n_configs, 5 * n_styles))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    markers = ['o', 's', '^', 'D', 'v']
    
    plot_idx = 1
    
    for style_idx, (X, y, style_name) in enumerate(zip(X_list, y_list, style_names)):
        for config_idx, config in enumerate(hog_configs):
            print(f"\n处理: {style_name} - 配置{config_idx + 1}")
            
            # 提取 HOG 特征
            features = extract_hog_features(
                X,
                orientations=config['orientations'],
                pixels_per_cell=tuple(config['pixels_per_cell']),
                cells_per_block=tuple(config['cells_per_block'])
            )
            
            print(f"  特征维度: {features.shape[1]}")
            
            # 降维
            features_2d = reduce_dimensions(features, method=reduction_method, n_components=2)
            
            # 绘制
            ax = fig.add_subplot(n_styles, n_configs, plot_idx)
            
            unique_labels = np.unique(y)
            for i, label in enumerate(unique_labels):
                mask = y == label
                ax.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    label=f'类别 {label}',
                    alpha=0.6,
                    s=50,
                    edgecolors='white',
                    linewidth=0.5
                )
            
            # 标题
            config_str = f"orient={config['orientations']}, ppc={config['pixels_per_cell']}"
            if style_idx == 0:
                title = f"{config_str}\n{style_name}"
            else:
                title = style_name
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # 只在第一列显示 y 轴标签
            if config_idx == 0:
                ax.set_ylabel(f'{reduction_method.upper()} 维度2', fontsize=10)
            else:
                ax.set_yticklabels([])
            
            # 只在最后一行显示 x 轴标签
            if style_idx == n_styles - 1:
                ax.set_xlabel(f'{reduction_method.upper()} 维度1', fontsize=10)
            else:
                ax.set_xticklabels([])
            
            # 只在第一个子图显示图例
            if style_idx == 0 and config_idx == 0:
                ax.legend(loc='best', fontsize=9)
            
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 图表已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='HOG 特征聚类可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python scripts/visualization/visualize_hog_clustering.py \\
      --output outputs/figures/hog_clustering.png
  
  # 使用 t-SNE 降维
  python scripts/visualization/visualize_hog_clustering.py \\
      --reduction tsne \\
      --output outputs/figures/hog_clustering_tsne.png
  
  # 自定义类别
  python scripts/visualization/visualize_hog_clustering.py \\
      --classes 0 1 2 3 4 \\
      --output outputs/figures/hog_clustering_5class.png
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/figures/hog_clustering.png',
        help='输出图表路径'
    )
    
    parser.add_argument(
        '--reduction', '-r',
        type=str,
        choices=['pca', 'tsne'],
        default='pca',
        help='降维方法（默认: pca）'
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
        default=150,
        help='每个类别的样本数（默认: 150）'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("HOG 特征聚类可视化".center(70))
    print("=" * 70)
    
    # 加载数据
    X_original, y = load_mnist_subset(
        n_samples_per_class=args.samples_per_class,
        classes=args.classes
    )
    
    # 创建不同风格的数据集
    print("\n📝 创建不同风格的数据集...")
    styles = [
        ('原始', 'original'),
        ('高亮度', 'bright'),
        ('低亮度', 'dark'),
    ]
    
    X_list = []
    y_list = []
    style_names = []
    
    for name, style in styles:
        X_perturbed = apply_perturbations(X_original, style)
        X_list.append(X_perturbed)
        y_list.append(y.copy())
        style_names.append(name)
        print(f"  ✓ {name}")
    
    # 定义多个 HOG 配置
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
        print(f"  配置 {i+1}: orientations={config['orientations']}, "
              f"pixels_per_cell={config['pixels_per_cell']}, "
              f"cells_per_block={config['cells_per_block']}")
    
    # 可视化
    print(f"\n📊 使用 {args.reduction.upper()} 降维...")
    
    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualize_clustering(
        X_list,
        y_list,
        style_names,
        hog_configs,
        reduction_method=args.reduction,
        save_path=args.output
    )
    
    print("\n" + "=" * 70)
    print("✅ 可视化完成！".center(70))
    print("=" * 70)
    
    print(f"\n📊 输出文件: {args.output}")
    print(f"\n💡 说明:")
    print(f"  - 不同颜色代表不同类别")
    print(f"  - 不同行代表不同风格的数据")
    print(f"  - 不同列代表不同的 HOG 超参数配置")
    print(f"  - 聚类越紧密，说明特征区分度越好")
    
    return 0


if __name__ == "__main__":
    exit(main())


