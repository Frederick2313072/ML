import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ==============================
# 全局容器：保存每轮 sample_weight
# ==============================
sample_weights_history = []
error_history = []
alpha_history = []

# 保存原始的 _boost 函数
orig_boost = AdaBoostClassifier._boost


# ============ Monkey Patch ==============
def patched_boost(self, iboost, X, y, sample_weight, random_state):
    # 记录 boosting 前的 sample weight
    sample_weights_history.append(sample_weight.copy())

    # 调用 sklearn 原始 boost
    sample_weight_new, estimator_weight, estimator_error = orig_boost(
        self, iboost, X, y, sample_weight, random_state
    )

    # 如果 estimator_error=None（早停），跳过
    if estimator_error is not None and (iboost + 1) % 5 == 0:
        error_history.append(estimator_error)
        alpha_history.append(estimator_weight)

        print(
            f"Boost {iboost + 1}/{self.n_estimators} | "
            f"error = {estimator_error:.4f} | "
            f"alpha = {estimator_weight:.4f}"
        )
    else:
        print(
            f"Boost {iboost + 1}/{self.n_estimators} stopped early "
            "(estimator worse than random)"
        )

    return sample_weight_new, estimator_weight, estimator_error


# 用 patched 版本替换 sklearn 的 _boost
AdaBoostClassifier._boost = patched_boost

if __name__ == "__main__":
    print("Downloading MNIST")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    # 转换标签为整数
    y = y.astype(np.int64)

    # 归一化到[0,1]
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base = DecisionTreeClassifier(max_depth=1)

    print("Training")
    clf = AdaBoostClassifier(
        estimator=base,
        n_estimators=50,  # 弱分类器数目
        learning_rate=0.5,  # 学习率
        random_state=42,
    )

    clf.fit(X_train, y_train)

    print("Training finished")

    # ---------- 评估 ----------
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(f"acc on training data：{accuracy_score(y_train, y_pred_train):.4f}")
    print(f"acc on testing data：{accuracy_score(y_test, y_pred_test):.4f}")
