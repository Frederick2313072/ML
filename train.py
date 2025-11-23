import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
    n_estimators=20,  # 弱分类器数目
    learning_rate=0.5,  # 学习率
    # algorithm="SAMME",  # 多分类
    random_state=42,
)

clf.fit(X_train, y_train)

print("Training finished")

# ---------- 评估 ----------
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print(f"acc on traing data：{accuracy_score(y_train, y_pred_train):.4f}")
print(f"acc on testing data：{accuracy_score(y_test, y_pred_test):.4f}")
