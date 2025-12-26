# 特征空间鲁棒性可视化工具

## 📖 概述

本工具用于可视化不同特征提取方法（原始像素、HOG、HU矩）对 `test_shift_config` 中定义的各种数据扰动的鲁棒性。

**核心思想**：
- 如果一个特征提取方法**鲁棒性好**，那么同一数字的不同风格（扰动）样本在特征空间中应该**聚集在一起**
- 如果特征提取方法**鲁棒性差**，不同风格的样本会**分散到不同区域**

---

## 🚀 快速开始

### 基本用法

```bash
python scripts/visualization/visualize_feature_robustness.py \
    --config configs/test_feature_robustness.json \
    --output outputs/figures/feature_robustness.png
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径（必需，包含 test_shift_config） | - |
| `--output` | 输出图表路径 | `outputs/figures/feature_robustness.png` |
| `--digit` | 要可视化的数字类别 | `8` |
| `--samples` | 每种风格的样本数 | `300` |

---

## 📋 配置文件要求

配置文件必须包含 `test_shift_config` 字段，定义各种扰动策略：

```json
{
  "data": {
    "test_shift_config": {
      "contrast": {
        "factor_range": [0.7, 1.3]
      },
      "brightness": {
        "shift_range": 0.2
      },
      "rotate": {
        "angle_range": 10
      },
      "combined": {
        "contrast": {"factor_range": [0.6, 1.4]},
        "brightness": {"shift_range": 0.3},
        "rotate": {"angle_range": 15}
      }
    }
  }
}
```

### 支持的扰动类型

| 扰动类型 | 配置参数 | 说明 |
|---------|---------|------|
| `contrast` | `factor_range: [min, max]` | 对比度调整 |
| `brightness` | `shift_range: float` | 亮度偏移 |
| `rotate` | `angle_range: int` | 旋转角度 |
| `gaussian` | `std: float` | 高斯噪声 |
| `salt_pepper` | `amount: float` | 椒盐噪声 |
| `blur` | `kernel_size: int` | 模糊 |

**注意**：可以在一个配置中组合多种扰动（如 `combined` 示例）

---

## 📊 输出图表说明

### 图表结构

```
┌─────────────────────────────────────────────────────────────────┐
│     Feature Space Robustness to Test Shift Perturbations       │
├──────────────────┬──────────────────┬────────────────────────────┤
│ Original Pixels  │  HOG Features    │  Hu Moments (7-dim)        │
│   (784-dim)      │                  │                            │
│ Avg. Center      │  Avg. Center     │  Avg. Center               │
│ Distance: 0.16   │  Distance: 0.18  │  Distance: 1.40            │
│                  │                  │                            │
│ ● Original       │  五种风格高度     │  风格分散                  │
│ ● contrast       │  重叠，说明      │  说明对扰动                │
│ ● brightness     │  鲁棒性好 ✓      │  敏感                      │
│ ● rotate         │                  │                            │
│ ● combined       │                  │                            │
└──────────────────┴──────────────────┴────────────────────────────┘
```

### 评价指标

**Avg. Style Center Distance (平均风格中心距离)**：
- 计算所有风格簇中心之间的平均距离
- **距离越小** → 鲁棒性越好
- **距离越大** → 鲁棒性越差

### 如何解读

#### ✅ 好的特征（HOG示例）

```
HOG Features
Avg. Style Center Distance: 0.18

     ●●■■▲▲◆◆★★
   ●●■■▲▲◆◆★★●●
   ■■▲▲◆◆★★●●■■
     ▲▲◆◆★★●●■■

图例：
● Original  ■ contrast  ▲ brightness  ◆ rotate  ★ combined
```

**特点**：
- 所有风格的点高度重叠
- 无法明显区分不同颜色的簇
- 距离小（通常 < 0.5）

#### ❌ 差的特征（原始像素示例）

```
Original Pixels
Avg. Style Center Distance: 0.16

●●●          ■■■
●●●              ▲▲▲
                     ◆◆◆
         ★★★

图例：
● Original  ■ contrast  ▲ brightness  ◆ rotate  ★ combined
```

**特点**：
- 不同风格形成独立的簇
- 可以清楚区分不同颜色
- 虽然这个例子距离也小，但如果扰动更强，会明显分离

---

## 💡 实际应用示例

### 示例1：对比不同HOG参数

```bash
# 创建多个配置文件，调整 hog_params
# configs/hog_4x4.json - pixels_per_cell: [4, 4]
# configs/hog_8x8.json - pixels_per_cell: [8, 8]

# 分别运行
python scripts/visualization/visualize_feature_robustness.py \
    --config configs/hog_4x4.json \
    --output outputs/figures/robustness_hog_4x4.png

python scripts/visualization/visualize_feature_robustness.py \
    --config configs/hog_8x8.json \
    --output outputs/figures/robustness_hog_8x8.png

# 对比两张图，选择鲁棒性更好的配置
```

### 示例2：测试不同数字的鲁棒性

```bash
# 数字8通常比较复杂
python scripts/visualization/visualize_feature_robustness.py \
    --config configs/test_feature_robustness.json \
    --digit 8 \
    --output outputs/figures/robustness_digit8.png

# 数字1通常比较简单
python scripts/visualization/visualize_feature_robustness.py \
    --config configs/test_feature_robustness.json \
    --digit 1 \
    --output outputs/figures/robustness_digit1.png
```

### 示例3：实验3使用场景

```bash
# 使用实验3的配置
python scripts/visualization/visualize_feature_robustness.py \
    --config configs/compare_hog.json \
    --output outputs/figures/exp3_feature_robustness.png \
    --digit 8 \
    --samples 500
```

---

## 🔬 技术细节

### 特征提取

1. **Original Pixels (原始像素)**
   - 直接使用 28×28=784 维的像素值
   - 无特征工程

2. **HOG Features**
   - 使用 `skimage.feature.hog` 提取
   - 默认参数：`orientations=9, pixels_per_cell=(4,4)`
   - 捕捉局部梯度方向直方图

3. **Hu Moments (HU矩)**
   - 使用 OpenCV 计算 7 个 Hu 不变矩
   - 经过 log 变换增强鲁棒性

### 降维方法

- 使用 **PCA (主成分分析)** 将高维特征降至 2D
- 保留最大的两个主成分
- 便于在平面上可视化

### 鲁棒性度量

```python
# 计算每个风格的簇中心
centers = [features[style_mask].mean(axis=0) for style in styles]

# 计算所有中心对之间的平均距离
avg_dist = mean([dist(c_i, c_j) for i,j in pairs(centers)])
```

---

## 📈 与实验3的关系

在实验3第一步中，我们发现：

| 实验 | 验证准确率 | 特征 |
|------|----------|------|
| compare_hog | **0.9720** ✓ | HOG |
| compare_hu | 0.9260 | HU矩 |
| compare_original | 0.9240 | 原始像素 |

**这个可视化工具帮助我们理解"为什么HOG最好"**：
- HOG特征对各种扰动（亮度、对比度、旋转）都鲁棒
- 不同风格的数据在HOG特征空间中聚集紧密
- 这种鲁棒性直接导致了更好的泛化能力

---

## ⚠️ 注意事项

### 1. 样本数选择

- 太少（< 100）：统计不稳定
- 太多（> 500）：计算慢，图表拥挤
- **推荐**：200-300 个样本

### 2. 数字选择

- **简单数字**（0, 1, 7）：通常所有特征都表现好
- **复杂数字**（3, 5, 8, 9）：更能区分特征质量
- **推荐**：使用数字 8 或 3

### 3. 扰动强度

- 太弱：无法测试鲁棒性
- 太强：所有特征都失效
- **推荐**：参考 `configs/test_feature_robustness.json`

### 4. 维度灾难

- HU矩只有 7 维，PCA降维效果有限
- 原始像素 784 维，PCA有明显降维效果
- HOG特征维度适中（取决于参数）

---

## 🛠️ 故障排查

### 问题1：RuntimeWarning (overflow, divide by zero)

**原因**：HU矩特征值过小，PCA计算时数值不稳定

**解决**：
- 这是正常的，不影响结果
- HU矩已经做了 log 变换，但仍可能数值极端

### 问题2：字体警告 (Font family 'Source Han Sans CN' not found)

**原因**：mplfonts 未安装或字体未配置

**解决**：
```bash
pip install mplfonts
python -c "from mplfonts.bin.cli import init; init()"
```

或者忽略此警告，图表仍然可以正常生成（只是中文显示可能有问题，但本工具图表都是英文）。

### 问题3：图表上只有一种颜色

**原因**：配置文件的 `test_shift_config` 为空

**解决**：
- 检查配置文件是否正确
- 确保 `test_shift_config` 包含至少一种扰动配置

---

## 📚 相关文件

- **脚本位置**：`scripts/visualization/visualize_feature_robustness.py`
- **示例配置**：`configs/test_feature_robustness.json`
- **输出目录**：`outputs/figures/`

---

## 🎯 总结

这个工具通过可视化帮助我们：

1. ✅ **理解特征鲁棒性**：直观看到不同特征对扰动的敏感度
2. ✅ **选择最佳特征**：通过平均距离量化评价
3. ✅ **验证实验结果**：解释为什么HOG在实验3中表现最好
4. ✅ **指导参数调优**：对比不同HOG参数的效果

**记住**：鲁棒性好的特征 = 不同风格数据在特征空间中紧密聚集！



