"""
特征重要性可视化示例
演示如何分析和可视化 AdaBoost 模型的特征重要性
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os

from src.monitor import BoostMonitor
from src.patch import boost_with_monitor
from src.utils import prepare_data
from src.evaluation import ModelEvaluator

# 应用猴子补丁
AdaBoostClassifier._boost = boost_with_monitor


def main():
    """主函数：演示特征重要性分析"""

    print("=" * 60)
    print("AdaBoost 特征重要性分析示例".center(56))
    print("=" * 60)

    # ========== 1. 准备数据 ==========
    print("\n步骤1: 准备数据")
    print("-" * 60)

    # 使用干净数据（特征重要性分析不需要噪声）
    X_train, X_test, y_train, y_test, train_noise_indices, train_clean_indices = (
        prepare_data(noise_ratio=0)
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"特征维度: {X_train.shape[1]} (28x28图像)")

    # ========== 2. 训练模型 ==========
    print("\n步骤2: 训练 AdaBoost 模型")
    print("-" * 60)

    monitor = BoostMonitor(
        train_noise_indices, train_clean_indices, is_data_noisy=False
    )

    # 使用较深的决策树以获得更好的特征重要性
    base = DecisionTreeClassifier(max_depth=3)  # 深度3的树
    clf = AdaBoostClassifier(
        estimator=base,
        n_estimators=50,
        learning_rate=0.5,
        random_state=42,
    )

    clf._monitor = monitor
    print("开始训练...")
    clf.fit(X_train, y_train)
    print("训练完成！")

    # 快速查看性能
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # ========== 3. 创建评估器 ==========
    print("\n步骤3: 创建评估器")
    print("-" * 60)

    evaluator = ModelEvaluator(clf, X_train, y_train, X_test, y_test)

    # ========== 4. 特征重要性分析 ==========
    print("\n步骤4: 特征重要性分析")
    print("-" * 60)

    # 创建结果目录
    results_dir = "results_feature_importance"
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果将保存到: {results_dir}/\n")

    # 4.1 基本特征重要性图
    print("(1) 生成特征重要性柱状图和分布图...")
    evaluator.plot_feature_importance(
        save_path=f"{results_dir}/feature_importance_basic.png", top_n=30
    )

    # 4.2 特征重要性热力图
    print("\n(2) 生成特征重要性热力图...")
    evaluator.plot_feature_importance_heatmap(
        image_shape=(28, 28), save_path=f"{results_dir}/feature_importance_heatmap.png"
    )

    # 4.3 各类别平均特征图
    print("\n(3) 生成各类别平均特征图...")
    evaluator.plot_per_class_feature_importance(
        X_test, y_test, image_shape=(28, 28), save_path=f"{results_dir}/class_features.png"
    )

    # 4.4 错误分类样本分析
    print("\n(4) 分析错误分类的样本...")
    evaluator.analyze_misclassified_samples(
        n_samples=20,
        image_shape=(28, 28),
        save_path=f"{results_dir}/misclassified_samples.png",
    )

    # ========== 5. 总结 ==========
    print("\n" + "=" * 60)
    print("分析总结".center(56))
    print("=" * 60)

    # 获取特征重要性
    importances = clf.feature_importances_

    print(f"✓ 总特征数: {len(importances)}")
    print(f"✓ 非零重要性特征: {np.sum(importances > 0)}")
    print(f"✓ 最大重要性: {importances.max():.6f}")
    print(f"✓ 平均重要性: {importances.mean():.6f}")

    # 找出最重要的像素位置
    top_5_indices = np.argsort(importances)[::-1][:5]
    print(f"\n最重要的5个像素位置:")
    for rank, idx in enumerate(top_5_indices, 1):
        row = idx // 28
        col = idx % 28
        print(
            f"  {rank}. 像素 {idx} (行{row}, 列{col}): 重要性 = {importances[idx]:.6f}"
        )

    print(f"\n✓ 所有图表已保存到: {results_dir}/")
    print("\n特征重要性分析完成！")
    print("=" * 60)

    # ========== 6. 解释结果 ==========
    print("\n" + "=" * 60)
    print("结果解释".center(56))
    print("=" * 60)

    print(
        """
1. 特征重要性柱状图：
   - 显示前30个最重要的特征（像素）
   - 重要性分数越高，该像素对分类贡献越大

2. 特征重要性热力图：
   - 将784维特征映射回28x28图像
   - 热力图显示哪些像素区域最重要
   - 蓝点标记前5%最重要的像素
   - 通常中心区域（数字笔画区域）重要性较高

3. 各类别平均特征图：
   - 显示每个数字（0-9）的平均特征
   - 可以看出每个数字的典型特征模式
   - 帮助理解模型如何区分不同数字

4. 错误分类样本：
   - 展示被模型误判的样本
   - 通常是手写风格特殊或模糊的样本
   - 可以发现模型的弱点

关键发现：
- MNIST数字识别中，边缘像素重要性通常较低
- 中心区域（数字笔画所在位置）重要性较高
- 不同数字的关键特征区域不同
    """
    )


if __name__ == "__main__":
    # 运行示例
    main()

