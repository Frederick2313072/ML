# 实验3步骤2：根据验证曲线构造最优模型

## 背景

如果训练过程中发现过拟合（验证准确率先升后降），应该选择验证准确率最高的轮次，重新构造模型，避免使用过拟合的后续轮次。

---

## 过拟合判断标准

### 1. 验证曲线退化
- **现象**: 验证准确率先上升后下降
- **原因**: 模型在训练集上过度拟合，泛化能力下降

### 2. 训练-验证差距过大
- **现象**: 训练准确率 - 验证准确率 > 5%
- **原因**: 模型记住了训练数据的噪声

---

## 当前实验结果分析

从实验3第一步的结果看：

| 模型 | 训练Acc | 验证Acc | 差距 | 验证曲线 | 结论 |
|------|---------|---------|------|----------|------|
| compare_original | 0.9335 | 0.9184 | 1.51% | 持续上升 | ✓ 无过拟合 |
| compare_hog | 0.9723 | 0.9580 | 1.43% | 持续上升 | ✓ 无过拟合 |
| compare_hu | 0.5269 | 0.5208 | 0.61% | 持续上升 | ✓ 无过拟合 |

**结论**: 三个模型都没有明显过拟合，验证曲线一直上升到最后一轮。

---

## 如果发现过拟合怎么办

### 步骤1: 识别最佳轮次

使用构造最优模型工具：

```bash
# 自动选择验证准确率最高的轮次
python scripts/build_optimal_model.py \
    --experiment compare_hog \
    --output-name compare_hog_optimal
```

工具会：
1. 分析验证曲线
2. 找出验证准确率最高的轮次
3. 构造只包含前N个弱学习器的模型
4. 保存最优模型和元信息

### 步骤2: 查看结果

输出示例：

```
======================================================================
                         构造最优轮次模型
======================================================================

📂 加载实验: compare_hog
✓ 已加载监控数据
✓ 已加载完整模型（总轮次: 500）

🎯 自动选择最佳轮次: 250  # 假设第250轮最佳

======================================================================
                           性能对比
======================================================================

最优轮次 (第 250 轮):
  验证准确率: 0.9650

完整模型 (第 500 轮):
  验证准确率: 0.9580  # 后续轮次过拟合，性能下降

✨ 使用最优轮次可提升: +0.0070 (0.70%)

======================================================================
🔨 构造前 250 轮的模型...
✓ 模型已保存: experiments/compare_hog_optimal/results/model.joblib
```

### 步骤3: 手动指定轮次

如果想测试特定轮次：

```bash
# 指定使用第300轮
python scripts/build_optimal_model.py \
    --experiment compare_hog \
    --round 300 \
    --output-name compare_hog_r300
```

---

## 完整示例流程

### 场景：假设 HOG 模型在第250轮达到最佳

```bash
# 1. 构造最优模型
python scripts/build_optimal_model.py \
    --experiment compare_hog \
    --output-name compare_hog_optimal

# 2. 可视化对比（完整模型 vs 最优模型）
python scripts/compare_classifiers.py \
    --experiments compare_hog compare_hog_optimal \
    --save outputs/figures/overfitting_comparison.png

# 3. 测试最优模型的泛化能力
# （需要配置 test_shift_config 并运行评估）
```

---

## 工具参数说明

```bash
python scripts/build_optimal_model.py [选项]
```

### 必需参数

- `--experiment`, `-e`: 源实验名称
  - 必须是已完成训练且有 monitor.joblib 的实验

### 可选参数

- `--round`, `-r`: 手动指定轮次
  - 如不指定，自动选择验证准确率最高的轮次
  
- `--output-name`, `-o`: 输出实验名称
  - 默认为 `<experiment>_optimal`
  
- `--base-dir`: 实验根目录
  - 默认为 `experiments`

---

## 输出文件

构造的最优模型会保存在：

```
experiments/<output_name>/
├── config.json                    # 配置文件（n_estimators已更新）
└── results/
    ├── model.joblib              # 最优模型
    └── optimal_model_info.json   # 元信息
```

### optimal_model_info.json 内容

```json
{
  "source_experiment": "compare_hog",
  "optimal_round": 250,
  "total_rounds": 500,
  "optimal_val_acc": 0.9650,
  "final_val_acc": 0.9580,
  "improvement": 0.0070
}
```

---

## 验证最优模型

### 方法1: 在测试集上评估

```python
import joblib
from sklearn.metrics import accuracy_score, f1_score

# 加载最优模型
optimal_model = joblib.load('experiments/compare_hog_optimal/results/model.joblib')

# 加载测试数据（需要从原实验获取）
# X_test, y_test = ...

# 评估
y_pred = optimal_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"最优模型 - 测试准确率: {acc:.4f}, F1: {f1:.4f}")
```

### 方法2: 测试泛化能力

使用 test_shift_config 测试在扰动数据上的性能。

---

## 何时需要构造最优模型

### 需要构造的情况

1. **验证曲线明显下降**
   - 例如：验证准确率从 0.96 降到 0.94
   
2. **训练-验证差距持续扩大**
   - 例如：差距从 2% 增长到 8%
   
3. **Early stopping 信号**
   - 验证准确率连续N轮不提升

### 不需要构造的情况

1. **验证曲线持续上升**（当前情况）
   - 继续训练可能带来更好性能
   
2. **训练-验证差距稳定且小**
   - 模型泛化良好
   
3. **最佳轮次就是最后一轮**
   - 已经是最优模型

---

## 进阶：自动早停训练

如果想在训练时自动早停，可以使用 `RobustAdaBoost`:

```python
from src.adalab.robust_adaboost import RobustAdaBoost
from sklearn.tree import DecisionTreeClassifier

# 配置早停
clf = RobustAdaBoost(
    base_estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=500,
    learning_rate=0.8,
    use_early_stopping=True,      # 启用早停
    validation_fraction=0.1,       # 10%数据用于验证
    early_stopping_rounds=20,      # 20轮不提升则停止
)

clf.fit(X_train, y_train)

print(f"早停轮次: {clf.best_n_estimators_}")
print(f"最佳验证得分: {max(clf.val_scores_):.4f}")
```

---

## 总结

### 当前实验

✅ **三个模型都无过拟合，可直接使用完整模型进行下一步测试**

### 工具用途

- 提供了 `build_optimal_model.py` 工具
- 用于处理发现过拟合的情况
- 可手动或自动选择最佳轮次

### 下一步

1. ✅ 找到最佳分类器（HOG特征）
2. ✅ 检查过拟合（无明显过拟合）
3. 📊 **测试泛化能力**（test_shift_config）
4. 🔊 测试噪声鲁棒性

---

## 参考

- **过拟合可视化**: `docs/overfitting_visualization_guide.md`
- **鲁棒AdaBoost**: `docs/robust_adaboost_guide.md`
- **实验3指南**: `docs/experiment3_best_classifier_guide.md`


