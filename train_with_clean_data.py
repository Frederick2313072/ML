import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.monitor import BoostMonitor
from src.patch import boost_with_monitor
from src.utils import prepare_data
from src.evaluation import quick_evaluate

# 用 patched 版本替换 sklearn 的 _boost
AdaBoostClassifier._boost = boost_with_monitor

if __name__ == "__main__":
    # get original clean data (无噪声数据)
    X_train, X_test, y_train, y_test, train_noise_indices, train_clean_indices = (
        prepare_data(noise_ratio=0)
    )
    
    # 创建监控器
    monitor = BoostMonitor(train_noise_indices, train_clean_indices)

    # 训练 AdaBoost
    print("训练 AdaBoost 模型...")
    base = DecisionTreeClassifier(max_depth=1)  # 使用决策树桩作为基学习器

    clf = AdaBoostClassifier(
        estimator=base,           # 基学习器
        n_estimators=50,          # 弱学习器数量
        learning_rate=0.5,        # 学习率
        random_state=42,          # 随机种子
    )

    # 绑定监控器到分类器
    clf._monitor = monitor
    
    # 开始训练
    clf.fit(X_train, y_train)

    print("训练完成！\n")

    # ========== 使用新的评估系统 ==========
    print("开始生成详细评估报告...")
    evaluator = quick_evaluate(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        monitor=monitor,
        noise_indices=train_noise_indices,
        clean_indices=train_clean_indices,
    )
