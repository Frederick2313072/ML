#!/usr/bin/env python3
"""
特征空间鲁棒性可视化

该脚本用于可视化不同特征提取方法（原始像素、HOG、HU矩）
对 test_shift_config 中定义的各种数据扰动的鲁棒性。

用法:
    python scripts/visualization/visualize_feature_robustness.py \\
        --config configs/compare_original.json \\
        --output outputs/figures/feature_robustness.png

作者: adalab
日期: 2024
"""

import sys
sys.path.insert(0, 'src')

import argparse
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from skimage.feature import hog
import cv2
from pathlib import Path

# 导入adalab模块
from adalab.data import MNISTPerturber, FeatureExtractor

# 设置中文字体
try:
    from mplfonts.bin.cli import init
    init()
    matplotlib.rcParams['font.family'] = 'Source Han Sans CN'
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("警告: mplfonts未安装，图表可能无法正确显示中文")
    pass


def load_mnist_subset(digit=8, n_samples=300, random_state=42):
    """
    加载MNIST数据集的子集
    
    Parameters
    ----------
    digit : int
        要选择的数字类别
    n_samples : int
        每个数字的样本数
    random_state : int
        随机种子
    
    Returns
    -------
    X : ndarray, shape (n_samples, 784)
        图像数据（展平为向量）
    y : ndarray, shape (n_samples,)
        标签
    """
    print(f"📦 加载 MNIST 数据集（数字: {digit}, 样本数: {n_samples}）...")
    
    # 加载MNIST
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X_all = np.array(mnist.data, dtype=np.float32) / 255.0
    y_all = np.array(mnist.target, dtype=int)
    
    # 筛选指定数字
    idx = np.where(y_all == digit)[0]
    
    # 随机选择样本
    rng = np.random.RandomState(random_state)
    selected_idx = rng.choice(idx, min(n_samples, len(idx)), replace=False)
    
    X = X_all[selected_idx]
    y = y_all[selected_idx]
    
    print(f"✓ 加载完成: {len(X)} 个样本")
    return X, y


def apply_shift_config(X, config, perturber):
    """
    根据 test_shift_config 应用扰动
    
    Parameters
    ----------
    X : ndarray
        原始图像数据
    config : dict
        扰动配置（如 {"contrast": {"factor_range": [0.7, 1.3]}}）
    perturber : MNISTPerturber
        扰动器实例
    
    Returns
    -------
    X_shifted : ndarray
        扰动后的图像数据
    """
    X_shifted = X.copy()
    
    for shift_type, params in config.items():
        if shift_type == "contrast":
            fr = params.get("factor_range", [0.5, 1.5])
            X_shifted = perturber.adjust_contrast(X_shifted, factor_range=tuple(fr))
        
        elif shift_type == "brightness":
            sr = params.get("shift_range", 0.3)
            X_shifted = perturber.add_brightness_shift(X_shifted, shift_range=sr)
        
        elif shift_type == "rotate":
            ar = params.get("angle_range", 15)
            X_shifted = perturber.rotate_slight(X_shifted, angle_range=ar)
        
        elif shift_type == "gaussian":
            std = params.get("std", 0.1)
            X_shifted = perturber.add_gaussian_noise(X_shifted, noise_std=std)
        
        elif shift_type == "salt_pepper":
            amount = params.get("amount", 0.05)
            X_shifted = perturber.add_salt_pepper_noise(X_shifted, amount=amount)
        
        elif shift_type == "blur":
            kernel_size = params.get("kernel_size", 3)
            X_shifted = perturber.add_blur(X_shifted, kernel_size=kernel_size)
    
    return X_shifted


def extract_hog_features(X, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """
    提取HOG特征
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, 784)
        图像数据
    orientations : int
        HOG方向数
    pixels_per_cell : tuple
        每个cell的像素数
    cells_per_block : tuple
        每个block的cell数
    
    Returns
    -------
    features : ndarray
        HOG特征
    """
    features = []
    for img in X:
        img_2d = img.reshape(28, 28)
        fd = hog(img_2d, 
                 orientations=orientations, 
                 pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block, 
                 block_norm='L2-Hys',
                 visualize=False)
        features.append(fd)
    return np.array(features)


def extract_hu_moments(X):
    """
    提取HU矩特征
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, 784)
        图像数据
    
    Returns
    -------
    features : ndarray, shape (n_samples, 7)
        HU矩特征
    """
    features = []
    for img in X:
        img_2d = img.reshape(28, 28)
        img_uint8 = (img_2d * 255).astype(np.uint8)
        moments = cv2.moments(img_uint8)
        hu_moments = cv2.HuMoments(moments).flatten()
        # log变换增强鲁棒性
        features.append(-np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10))
    return np.array(features)


def visualize_feature_robustness(
    X_styles_dict,
    style_names,
    feature_extractor,
    feature_name,
    reduction_method='pca',
    save_path=None
):
    """
    可视化单个特征空间的鲁棒性
    
    Parameters
    ----------
    X_styles_dict : dict
        {style_name: X_data} 不同风格的数据
    style_names : list
        风格名称列表
    feature_extractor : callable
        特征提取器函数
    feature_name : str
        特征名称
    reduction_method : str
        降维方法 ('pca' 或 'tsne')
    save_path : str, optional
        保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    fig.suptitle(f'Feature Space Robustness: {feature_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 为每个风格分配颜色
    colors = matplotlib.colormaps.get_cmap('tab10').resampled(len(style_names))
    
    # 提取所有风格的特征并合并
    all_features = []
    all_labels = []
    
    for style_idx, style_name in enumerate(style_names):
        X_style = X_styles_dict[style_name]
        features = feature_extractor(X_style)
        all_features.append(features)
        all_labels.extend([style_idx] * len(features))
    
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    # 降维到2D
    print(f"  使用 {reduction_method.upper()} 降维...")
    if reduction_method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(all_features)
        explained_var = reducer.explained_variance_ratio_.sum()
        print(f"  PCA 解释方差: {explained_var:.2%}")
    elif reduction_method == 'tsne':
        from sklearn.manifold import TSNE
        # t-SNE参数优化
        reducer = TSNE(
            n_components=2, 
            random_state=42,
            perplexity=30,  # 典型值：5-50
            max_iter=1000,  # 修正参数名
            learning_rate=200.0
        )
        features_2d = reducer.fit_transform(all_features)
        print(f"  t-SNE 降维完成")
    else:
        raise ValueError(f"不支持的降维方法: {reduction_method}")
    
    # 计算风格中心间的平均距离（评价鲁棒性）
    style_centers = []
    for style_idx in range(len(style_names)):
        mask = all_labels == style_idx
        center = features_2d[mask].mean(axis=0)
        style_centers.append(center)
    
    style_centers = np.array(style_centers)
    avg_dist = 0
    n_pairs = 0
    for i in range(len(style_centers)):
        for j in range(i+1, len(style_centers)):
            avg_dist += np.linalg.norm(style_centers[i] - style_centers[j])
            n_pairs += 1
    avg_dist /= n_pairs if n_pairs > 0 else 1
    
    # 绘制散点图
    for style_idx, style_name in enumerate(style_names):
        mask = all_labels == style_idx
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                  c=[colors(style_idx)], label=style_name,
                  alpha=0.7, s=50, edgecolors='white', linewidths=0.5)
    
    # 标题和标签
    reduction_name = reduction_method.upper()
    title = f'Avg. Style Center Distance: {avg_dist:.2f}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    if reduction_method == 'pca':
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
    else:
        ax.set_xlabel(f'{reduction_name} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{reduction_name} Dimension 2', fontsize=12)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='特征空间鲁棒性可视化',
        epilog='''
示例:
  # 基本用法
  python scripts/visualization/visualize_feature_robustness.py \\
      --config configs/compare_original.json \\
      --output outputs/figures/feature_robustness.png
  
  # 使用不同数字
  python scripts/visualization/visualize_feature_robustness.py \\
      --config configs/compare_hog.json \\
      --digit 3 \\
      --samples 200 \\
      --output outputs/figures/feature_robustness_digit3.png
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='配置文件路径（包含test_shift_config）')
    parser.add_argument('--output', '-o', type=str, 
                       default='outputs/figures/feature_robustness.png',
                       help='输出图表路径')
    parser.add_argument('--digit', type=int, default=8,
                       help='要可视化的数字类别（默认: 8）')
    parser.add_argument('--samples', type=int, default=300,
                       help='每种风格的样本数（默认: 300）')
    parser.add_argument('--reduction', type=str, default='tsne',
                       choices=['pca', 'tsne'],
                       help='降维方法（默认: tsne）')
    
    args = parser.parse_args()
    
    # 加载配置文件
    print(f"\n📄 加载配置文件: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    test_shift_config = config['data']['test_shift_config']
    feature_config = config['data'].get('feature_config', {})
    use_feature = config['data'].get('use_feature', 'original')
    
    print(f"✓ 检测到 {len(test_shift_config)} 种扰动配置:")
    for name in test_shift_config.keys():
        print(f"  - {name}")
    print(f"✓ 使用特征: {use_feature}")
    
    # 1. 加载MNIST数据
    X_original, y = load_mnist_subset(digit=args.digit, n_samples=args.samples)
    
    # 2. 创建扰动器
    perturber = MNISTPerturber(random_state=42)
    
    # 3. 应用不同的扰动配置
    print(f"\n🔄 应用扰动配置...")
    X_styles_dict = {'Original': X_original}
    style_names = ['Original']
    
    for shift_name, shift_config in test_shift_config.items():
        print(f"  处理: {shift_name}")
        X_shifted = apply_shift_config(X_original, shift_config, perturber)
        X_styles_dict[shift_name] = X_shifted
        style_names.append(shift_name)
    
    print(f"✓ 共生成 {len(style_names)} 种风格的数据")
    
    # 4. 根据配置文件选择特征提取器
    print(f"\n🔧 准备特征提取器...")
    if use_feature == 'original':
        feature_extractor = lambda X: X
        feature_name = 'Original Pixels (784-dim)'
    elif use_feature == 'hog':
        hog_params = feature_config.get('hog_params', {})
        orientations = hog_params.get('orientations', 9)
        pixels_per_cell = tuple(hog_params.get('pixels_per_cell', [4, 4]))
        cells_per_block = tuple(hog_params.get('cells_per_block', [2, 2]))
        
        print(f"  HOG参数: orientations={orientations}, pixels_per_cell={pixels_per_cell}, cells_per_block={cells_per_block}")
        
        feature_extractor = lambda X: extract_hog_features(
            X, 
            orientations=orientations, 
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block
        )
        feature_name = f'HOG Features (orientations={orientations}, ppc={pixels_per_cell})'
    elif use_feature == 'hu':
        feature_extractor = extract_hu_moments
        feature_name = 'Hu Moments (7-dim)'
    else:
        raise ValueError(f"不支持的特征类型: {use_feature}")
    
    print(f"✓ 特征提取器: {feature_name}")
    
    # 5. 可视化
    print(f"\n📊 生成可视化...")
    print(f"✓ 降维方法: {args.reduction.upper()}")
    visualize_feature_robustness(
        X_styles_dict,
        style_names,
        feature_extractor,
        feature_name,
        reduction_method=args.reduction,
        save_path=args.output
    )
    
    print(f"\n✅ 完成！")


if __name__ == '__main__':
    main()
