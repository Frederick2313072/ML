# 实验3：寻找最佳分类器指南

本指南说明如何完成实验3的第一步：找到在MNIST和课程数据上效果最好的分类器。

## 目标

找到最优的分类器配置，包括：
- **特征类型**: original（原始像素）、HOG、HU矩
- **决策树参数**: max_depth、criterion、max_features
- **AdaBoost参数**: n_estimators、learning_rate

**约束条件**: 数据不添加噪声（noise_ratio = 0）

---

## 步骤1：准备对比实验配置

已有配置文件：
- `configs/compare_original.json` - 原始像素特征
- `configs/compare_hog.json` - HOG特征
- `configs/compare_hu.json` - HU矩特征

所有配置使用相同的模型参数：
- max_depth: 3
- learning_rate: 0.8
- n_estimators: 500

---

## 步骤2：运行对比实验

```bash
# 运行三种特征类型的实验
python -m adalab.cli configs/compare_original.json
python -m adalab.cli configs/compare_hog.json
python -m adalab.cli configs/compare_hu.json
```

每个实验会生成：
- `experiments/<name>/results/model.joblib` - 训练好的模型
- `experiments/<name>/results/monitor.joblib` - 训练监控数据
- `experiments/<name>/results/final_results.csv` - 结果CSV
- `experiments/<name>/config.json` - 配置备份

---

## 步骤3：对比实验结果

```bash
# 生成对比图表
python scripts/compare_classifiers.py \
    --experiments compare_original compare_hog compare_hu \
    --save outputs/figures/feature_comparison.png
```

**输出内容**：

1. **对比表格**：显示每个实验的关键指标
   - 最佳验证准确率
   - 最终验证准确率
   - 最佳验证F1
   - 最终验证F1

2. **最佳模型推荐**：自动识别性能最优的配置

3. **过拟合检测**：标记训练-验证差距过大的模型

4. **对比图表**（3个子图）：
   - 验证准确率对比
   - F1分数对比
   - 准确率 vs F1 散点图

---

## 步骤4：参数调优（可选）

如果需要进一步优化，可以调整以下参数：

### 4.1 调整决策树深度

创建新配置文件测试不同深度：

```json
// configs/compare_hog_depth4.json
{
  "experiment": {
    "name": "compare_hog_depth4"
  },
  "data": {
    "noise_ratio": 0.0,
    "use_feature": "hog",
    "hog_params": {...}
  },
  "model": {
    "estimator": {
      "max_depth": 4  // 增加深度
    },
    "n_estimators": 500,
    "learning_rate": 0.8
  }
}
```

### 4.2 调整学习率

```json
// configs/compare_hog_lr10.json
{
  "model": {
    "learning_rate": 1.0  // 增加学习率
  }
}
```

### 4.3 调整estimator数量

```json
// configs/compare_hog_est800.json
{
  "model": {
    "n_estimators": 800  // 增加弱学习器数量
  }
}
```

然后重新运行对比：

```bash
python -m adalab.cli configs/compare_hog_depth4.json
python -m adalab.cli configs/compare_hog_lr10.json
python -m adalab.cli configs/compare_hog_est800.json

python scripts/compare_classifiers.py \
    --experiments compare_hog compare_hog_depth4 compare_hog_lr10 compare_hog_est800 \
    --save outputs/figures/hyperparameter_tuning.png
```

---

## 步骤5：可视化最佳模型的训练过程

找到最佳模型后，查看其详细训练曲线：

```bash
# 假设 compare_hog 是最佳模型
python scripts/visualization/visualize_from_results.py \
    --joblib experiments/compare_hog/results/monitor.joblib \
    --save outputs/figures/best_model_training.png
```

---

## 预期结果

根据之前的实验经验，预期结果：

| 特征类型 | 验证准确率 | F1分数 | 特点 |
|---------|-----------|--------|------|
| HOG     | ~0.95+    | ~0.95+ | 通常最优，特征维度适中 |
| Original| ~0.92+    | ~0.92+ | 性能尚可，但特征维度高 |
| HU      | ~0.88+    | ~0.88+ | 特征维度低，信息损失较多 |

---

## 下一步

完成步骤3后，你将得到：

✅ **最佳分类器配置**（特征+超参数）

然后可以进行实验3的后续步骤：

1. **测试泛化能力**（test_shift_config）
2. **过拟合分析**（如果发现过拟合）
3. **噪声鲁棒性测试**

---

## 常见问题

### Q1: 如何判断哪个模型最好？

**A**: 优先看**最佳验证准确率**（best_val_acc），其次看**最佳验证F1**。如果两个模型性能接近，选择：
- 训练-验证差距更小的（泛化更好）
- 特征维度更低的（计算更快）
- 达到最佳性能所需轮次更少的（训练更快）

### Q2: 发现过拟合怎么办？

**A**: 如果训练准确率 - 验证准确率 > 0.05，说明过拟合。解决方法：
1. 减小 max_depth（如从3降到2）
2. 减小 max_features（如从0.3降到0.2）
3. 减小 learning_rate（如从0.8降到0.5）
4. 根据验证曲线选择最优轮次构造模型（early stopping）

### Q3: 实验运行需要多久？

**A**: 每个实验约需：
- HOG特征: 5-10分钟
- Original特征: 10-20分钟（特征维度高）
- HU特征: 2-5分钟（特征维度低）

建议在服务器或性能较好的机器上运行。

---

## 脚本快速参考

```bash
# 1. 运行实验
python -m adalab.cli configs/<config_name>.json

# 2. 对比多个实验
python scripts/compare_classifiers.py \
    --experiments exp1 exp2 exp3 \
    --save outputs/figures/comparison.png

# 3. 可视化单个实验
python scripts/visualization/visualize_from_results.py \
    --joblib experiments/<exp_name>/results/monitor.joblib
```

---

**开始实验吧！** 🚀

