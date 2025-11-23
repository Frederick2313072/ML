import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.monitor import BoostMonitor
from src.patch import boost_with_monitor
from src.utils import prepare_data

# 替换 AdaBoostClassifier._boost
AdaBoostClassifier._boost = boost_with_monitor


if __name__ == "__main__":
    # get data with default amount of noise
    X_train, X_test, y_train, y_test, train_noise_indices, train_clean_indices = (
        prepare_data()
    )

    # monitor
    boost_monitor = BoostMonitor(
        train_noise_indices, train_clean_indices, is_data_noisy=True
    )

    # 训练 AdaBoost
    print("Training AdaBoost...")
    base = DecisionTreeClassifier(max_depth=1)

    clf = AdaBoostClassifier(
        estimator=base,
        n_estimators=50,
        learning_rate=0.5,
        random_state=42,
    )

    clf._monitor = boost_monitor
    clf.fit(X_train, y_train)

    print("Training finished!")

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(f"acc on training data：{accuracy_score(y_train, y_pred_train):.4f}")
    print(f"acc on testing data：{accuracy_score(y_test, y_pred_test):.4f}")
