"""
评估系统使用示例
演示如何使用新的评估模块进行详细的模型性能分析
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
    """主函数：演示完整的训练和评估流程"""
    
    # ========== 1. 准备数据 ==========
    print("=" * 60)
    print("步骤1: 准备数据")
    print("=" * 60)
    
    # 使用含噪声的数据（5%噪声比例）
    X_train, X_test, y_train, y_test, train_noise_indices, train_clean_indices = (
        prepare_data(noise_ratio=0.05)
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"噪声样本数: {len(train_noise_indices)}\n")
    
    # ========== 2. 创建监控器 ==========
    print("=" * 60)
    print("步骤2: 创建训练监控器")
    print("=" * 60)
    
    monitor = BoostMonitor(
        train_noise_indices, 
        train_clean_indices, 
        is_data_noisy=True  # 启用噪声追踪
    )
    print("监控器创建完成\n")
    
    # ========== 3. 训练模型 ==========
    print("=" * 60)
    print("步骤3: 训练 AdaBoost 模型")
    print("=" * 60)
    
    base = DecisionTreeClassifier(max_depth=1)  # 决策树桩
    clf = AdaBoostClassifier(
        estimator=base,
        n_estimators=50,
        learning_rate=0.5,
        random_state=42,
    )
    
    # 绑定监控器
    clf._monitor = monitor
    
    # 训练
    print("开始训练...")
    clf.fit(X_train, y_train)
    print("训练完成！\n")
    
    # ========== 4. 创建评估器 ==========
    print("=" * 60)
    print("步骤4: 创建评估器")
    print("=" * 60)
    
    evaluator = ModelEvaluator(
        clf, 
        X_train, y_train, 
        X_test, y_test,
        class_names=[str(i) for i in range(10)]  # 数字 0-9
    )
    print("评估器创建完成\n")
    
    # ========== 5. 打印文本报告 ==========
    print("=" * 60)
    print("步骤5: 生成文本报告")
    print("=" * 60)
    
    # 5.1 基本指标
    evaluator.print_basic_metrics()
    
    # 5.2 分类报告
    evaluator.print_classification_report()
    
    # 5.3 各类别详细指标
    evaluator.print_per_class_metrics()
    
    # 5.4 噪声影响分析
    noise_stats = evaluator.analyze_noise_impact(
        train_noise_indices, 
        train_clean_indices
    )
    
    # ========== 6. 生成可视化图表 ==========
    print("\n" + "=" * 60)
    print("步骤6: 生成可视化图表")
    print("=" * 60)
    
    # 创建结果目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果将保存到: {results_dir}/\n")
    
    # 6.1 混淆矩阵
    print("生成混淆矩阵...")
    evaluator.plot_confusion_matrix(
        save_path=f"{results_dir}/confusion_matrix.png"
    )
    
    # 6.2 归一化混淆矩阵
    print("生成归一化混淆矩阵...")
    evaluator.plot_normalized_confusion_matrix(
        save_path=f"{results_dir}/confusion_matrix_normalized.png"
    )
    
    # 6.3 各类别性能图
    print("生成各类别性能图...")
    evaluator.plot_class_performance(
        save_path=f"{results_dir}/class_performance.png"
    )
    
    # 6.4 训练历史曲线
    print("生成训练历史曲线...")
    evaluator.plot_training_history(
        monitor,
        save_path=f"{results_dir}/training_history.png"
    )
    
    # ========== 7. 总结 ==========
    print("\n" + "=" * 60)
    print("步骤7: 评估总结")
    print("=" * 60)
    
    test_acc = evaluator.y_test_pred
    test_acc = (evaluator.y_test == test_acc).mean()
    
    print(f"✓ 测试集准确率: {test_acc:.4f}")
    if noise_stats:
        print(f"✓ 噪声样本准确率: {noise_stats['noise_accuracy']:.4f}")
        print(f"✓ 干净样本准确率: {noise_stats['clean_accuracy']:.4f}")
        print(f"✓ 准确率差距: {noise_stats['accuracy_gap']:.4f}")
    print(f"✓ 所有图表已保存到: {results_dir}/")
    print("\n评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行示例
    main()

