# HOG 特征聚类可视化指南

## 功能概述

可视化不同 HOG 超参数配置下的特征空间聚类效果，帮助理解：
1. HOG 特征如何区分不同类别
2. 不同超参数对特征空间的影响
3. 不同风格数据的特征分布差异

---

## 快速开始

### 1. 基本 2D 可视化

```bash
python scripts/visualization/visualize_hog_clustering.py \
    --output outputs/figures/hog_clustering.png
```

**输出**：
- 3行（不同数据风格）× 3列（不同HOG配置）的子图网格
- 使用 PCA 降维到 2D
- 展示原始、高亮度、低亮度三种风格

### 2. 3D 可视化

```bash
python scripts/visualization/visualize_hog_clustering_3d.py \
    --output outputs/figures/hog_clustering_3d.png
```

**输出**：
- 3个 3D 散点图（对应3种HOG配置）
- 使用 PCA 降维到 3D
- 可以看到更立体的特征分布

---

## 参数说明

### visualize_hog_clustering.py

```bash
python scripts/visualization/visualize_hog_clustering.py [选项]
```

**参数**：

- `--output`, `-o`：输出图表路径
  - 默认：`outputs/figures/hog_clustering.png`

- `--reduction`, `-r`：降维方法
  - 选项：`pca`, `tsne`
  - 默认：`pca`

- `--classes`, `-c`：要可视化的类别
  - 默认：`0 1 8`（数字0、1、8）
  - 示例：`--classes 0 1 2 3 4`

- `--samples-per-class`：每个类别的样本数
  - 默认：`150`
  - 建议范围：50-300

### visualize_hog_clustering_3d.py

```bash
python scripts/visualization/visualize_hog_clustering_3d.py [选项]
```

**参数**：

- `--output`, `-o`：输出图表路径
- `--classes`, `-c`：要可视化的类别
- `--samples-per-class`：每个类别的样本数

---

## 使用示例

### 示例1：可视化更多类别

```bash
# 可视化 5 个类别
python scripts/visualization/visualize_hog_clustering.py \
    --classes 0 1 2 3 4 \
    --samples-per-class 100 \
    --output outputs/figures/hog_clustering_5class.png
```

### 示例2：使用 t-SNE 降维

```bash
# t-SNE 通常能得到更清晰的聚类边界
python scripts/visualization/visualize_hog_clustering.py \
    --reduction tsne \
    --output outputs/figures/hog_clustering_tsne.png
```

**注意**: t-SNE 计算较慢，建议减少样本数

### 示例3：对比数字 6 和 9

```bash
# 这两个数字容易混淆，看看 HOG 特征能否区分
python scripts/visualization/visualize_hog_clustering.py \
    --classes 6 9 \
    --samples-per-class 200 \
    --output outputs/figures/hog_clustering_6vs9.png
```

### 示例4：3D 可视化

```bash
python scripts/visualization/visualize_hog_clustering_3d.py \
    --classes 0 1 8 \
    --samples-per-class 200 \
    --output outputs/figures/hog_clustering_3d.png
```

---

## HOG 超参数说明

脚本中预设了 3 种 HOG 配置：

### 配置 1：细粒度特征

```python
{
    'orientations': 9,         # 9个方向
    'pixels_per_cell': [4, 4], # 4×4 像素/单元格（更细）
    'cells_per_block': [2, 2]  # 2×2 单元格/块
}
```

- **特点**：特征维度高（约 1764 维），捕捉细节
- **适用**：图像尺寸较大，细节丰富

### 配置 2：粗粒度特征

```python
{
    'orientations': 9,
    'pixels_per_cell': [8, 8], # 8×8 像素/单元格（更粗）
    'cells_per_block': [2, 2]
}
```

- **特点**：特征维度中等（约 324 维），计算快
- **适用**：平衡性能和计算开销（**推荐**）

### 配置 3：更多方向

```python
{
    'orientations': 12,        # 12个方向（更多）
    'pixels_per_cell': [4, 4],
    'cells_per_block': [2, 2]
}
```

- **特点**：捕捉更丰富的梯度方向
- **适用**：形状复杂的对象

---

## 图表解读

### 2D 聚类图

#### 布局

```
        配置1          配置2          配置3
风格1   [子图1-1]      [子图1-2]      [子图1-3]
风格2   [子图2-1]      [子图2-2]      [子图2-3]
风格3   [子图3-1]      [子图3-2]      [子图3-3]
```

#### 观察要点

1. **聚类紧密度**
   - 同类样本聚在一起 → 特征区分度好
   - 散布广泛 → 类内差异大

2. **类别分离度**
   - 不同类别距离远 → 容易分类
   - 重叠严重 → 容易混淆

3. **超参数影响**
   - 横向对比：同一风格下不同配置的效果
   - 预期：配置1（细粒度）聚类更紧密

4. **风格鲁棒性**
   - 纵向对比：同一配置下不同风格的效果
   - 预期：好的特征在不同风格下都能保持聚类结构

### 3D 聚类图

#### 观察要点

1. **立体分布**
   - 3D 可以看到 2D 中重叠的点实际是分离的
   - 旋转视角观察不同方向的分离度

2. **主成分贡献**
   - 坐标轴标注了每个主成分的解释方差比例
   - 如：PC1 (45%) 表示第一主成分解释了 45% 的方差

---

## 预期结果

### 典型的好特征

```
类别 0 (圆形)：
  ● ● ●
  ● ● ●    ← 紧密聚类

类别 1 (竖线)：
        ■ ■ ■
        ■ ■ ■  ← 与类别0距离远

类别 8 (两个圆)：
    ▲ ▲ ▲
    ▲ ▲ ▲      ← 独立聚类
```

### 可能的问题

1. **聚类分散**
   ```
   ● ●    ●  ●
      ●   ●     ← 类内方差大
   ● ●  ●
   ```
   **原因**：
   - 样本多样性（手写体差异）
   - 特征不够判别性
   - 风格变化太大

2. **类别重叠**
   ```
   ● ● ■ ■
   ● ■ ■ ●   ← 类别0和类别1重叠
   ■ ● ● ■
   ```
   **原因**：
   - 数字本身相似（如6和9）
   - HOG 参数不合适
   - 降维损失信息

---

## 扩展实验

### 1. 测试更多 HOG 配置

修改脚本中的 `hog_configs` 列表：

```python
hog_configs = [
    {'orientations': 6, 'pixels_per_cell': [8, 8], 'cells_per_block': [2, 2]},
    {'orientations': 9, 'pixels_per_cell': [8, 8], 'cells_per_block': [2, 2]},
    {'orientations': 12, 'pixels_per_cell': [8, 8], 'cells_per_block': [2, 2]},
    {'orientations': 15, 'pixels_per_cell': [8, 8], 'cells_per_block': [2, 2]},
]
```

### 2. 测试更多数据风格

修改脚本中的 `styles` 列表：

```python
styles = [
    ('原始', 'original'),
    ('高亮度', 'bright'),
    ('低亮度', 'dark'),
    ('噪声', 'noisy'),
    ('模糊', 'blurred'),
]
```

### 3. 使用自己的数据

替换 `load_mnist_subset()` 函数，加载自己的图像数据。

---

## 与实验3的关系

### 选择最佳 HOG 参数

通过聚类可视化，可以：

1. **直观评估**：哪个配置的聚类效果最好
2. **理解原因**：为什么某个配置在分类中表现更好
3. **调优指导**：如何进一步调整参数

### 实验3中使用的 HOG 配置

```json
{
  "orientations": 9,
  "pixels_per_cell": [4, 4],
  "cells_per_block": [2, 2]
}
```

- 在聚类可视化中对应"配置 1"
- 特征维度约 1764 维
- 在实验3中达到 **95.80%** 准确率

---

## 常见问题

### Q1: 为什么 t-SNE 计算很慢？

**A**: t-SNE 复杂度高，建议：
- 减少样本数：`--samples-per-class 50`
- 或使用 PCA（默认）

### Q2: 如何解释 PCA 的解释方差？

**A**: 
- PC1 (45%) 表示第一主成分保留了 45% 的信息
- 前3个主成分总和 > 70% 说明降维效果好

### Q3: 不同风格的聚类结构应该相似吗？

**A**: 
- **理想情况**：是的，说明特征对风格变化鲁棒
- **实际情况**：会有差异，但聚类模式应该保持

### Q4: 如何选择要可视化的类别？

**A**:
- **易区分**：0（圆）、1（竖线）、8（两圆）
- **易混淆**：6和9、3和8、4和9
- **形状多样**：2、3、5、7

---

## 技术细节

### 降维方法对比

| 方法 | 速度 | 保持全局结构 | 保持局部结构 | 确定性 |
|------|------|-------------|-------------|--------|
| PCA | 快 | ✓ | × | ✓ |
| t-SNE | 慢 | × | ✓ | × |

### 特征维度估算

HOG 特征维度 = `orientations × n_cells_x × n_cells_y × cells_per_block²`

示例（28×28图像）：
- 配置1：`9 × 6 × 6 × 4 = 1296`维
- 配置2：`9 × 3 × 3 × 4 = 324`维

---

## 输出示例

运行脚本后会生成类似这样的可视化：

```
原始数据     [配置1: 紧密聚类]  [配置2: 中等聚类]  [配置3: 细节聚类]
高亮度       [配置1: 略散]      [配置2: 保持]      [配置3: 保持]
低亮度       [配置1: 略散]      [配置2: 保持]      [配置3: 保持]
```

**观察**：配置2在不同风格下都保持稳定 → 鲁棒性好 → 适合实际应用

---

## 参考

- **HOG 原理**: Dalal & Triggs (2005)
- **降维方法**: 
  - PCA: Jolliffe (2002)
  - t-SNE: van der Maaten & Hinton (2008)
- **实验3指南**: `docs/experiment3_best_classifier_guide.md`

---

**通过可视化深入理解 HOG 特征！** 📊


