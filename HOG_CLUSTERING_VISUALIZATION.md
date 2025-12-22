# HOG 特征聚类可视化工具

## 功能说明

可视化 HOG 特征在不同超参数配置下的聚类效果，帮助理解：
- ✅ HOG 特征如何区分不同数字类别
- ✅ 不同超参数（orientations、pixels_per_cell）的影响
- ✅ 不同数据风格（原始、高亮度、低亮度）的特征分布
- ✅ 2D 和 3D 特征空间的可视化

---

## 🚀 快速开始

### 一键生成所有可视化

```bash
bash scripts/run_hog_clustering_vis.sh
```

这会生成 4 张图表：
1. **2D PCA 聚类** - 最常用
2. **2D t-SNE 聚类** - 局部结构更清晰
3. **3D PCA 聚类** - 立体视角
4. **5类别聚类** - 更多类别对比

**预计时间**: 2-5 分钟

---

## 📊 手动运行

### 1. 基本 2D 可视化

```bash
python scripts/visualization/visualize_hog_clustering.py \
    --output outputs/figures/hog_clustering.png
```

**输出布局**:
```
        HOG配置1       HOG配置2       HOG配置3
        (细粒度)       (粗粒度)       (更多方向)
原始    [聚类图]       [聚类图]       [聚类图]
高亮度  [聚类图]       [聚类图]       [聚类图]
低亮度  [聚类图]       [聚类图]       [聚类图]
```

### 2. 使用 t-SNE 降维

```bash
python scripts/visualization/visualize_hog_clustering.py \
    --reduction tsne \
    --output outputs/figures/hog_clustering_tsne.png
```

**特点**: 
- 保持局部结构，聚类边界更清晰
- 计算较慢（建议减少样本数）

### 3. 3D 可视化

```bash
python scripts/visualization/visualize_hog_clustering_3d.py \
    --output outputs/figures/hog_clustering_3d.png
```

**特点**:
- 3个主成分（PC1, PC2, PC3）
- 可以看到 2D 中重叠的点实际是分离的
- 每个轴标注了解释方差比例

---

## 🔧 预设的 HOG 配置

脚本预设了 3 种 HOG 配置进行对比：

### 配置 1：细粒度特征
```python
orientations=9, pixels_per_cell=[4,4], cells_per_block=[2,2]
```
- 特征维度: ~1764 维
- 适合: 捕捉细节，图像较大

### 配置 2：粗粒度特征（推荐）
```python
orientations=9, pixels_per_cell=[8,8], cells_per_block=[2,2]
```
- 特征维度: ~324 维
- 适合: 平衡性能和计算开销
- **实验3使用的配置** ✓

### 配置 3：更多方向
```python
orientations=12, pixels_per_cell=[4,4], cells_per_block=[2,2]
```
- 特征维度: ~2352 维
- 适合: 形状复杂的对象

---

## 📈 如何解读结果

### 好的聚类效果

```
类别 0:  ●●●●●
         ●●●●●  ← 紧密聚类
         
类别 1:        ■■■■■
               ■■■■■  ← 与类别0距离远
```

**特征**:
- 同类样本聚在一起（类内紧密）
- 不同类样本分开（类间分离）
- 在不同风格下保持结构（鲁棒）

### 需要改进的情况

```
类别 0:  ● ●  ●
           ●   ● ●  ← 分散
         ● ●
         
类别 1:  ■ ● ■
         ● ■ ●     ← 与类别0重叠
```

**可能原因**:
- HOG 参数不合适
- 类别本身相似（如 6 和 9）
- 数据风格变化太大

---

## 🎯 与实验3的关系

### 验证 HOG 配置的选择

实验3中，HOG 特征达到 **95.80%** 准确率，使用的配置：
```json
{
  "orientations": 9,
  "pixels_per_cell": [4, 4],
  "cells_per_block": [2, 2]
}
```

通过聚类可视化可以：
1. **直观看到**为什么这个配置表现好（聚类紧密且分离）
2. **理解原因**：特征维度合适，区分度高
3. **优化方向**：是否需要调整参数

---

## 📚 高级用法

### 1. 可视化更多类别

```bash
python scripts/visualization/visualize_hog_clustering.py \
    --classes 0 1 2 3 4 5 6 7 8 9 \
    --samples-per-class 50 \
    --output outputs/figures/hog_clustering_all.png
```

### 2. 对比易混淆的数字

```bash
# 6 和 9 很相似
python scripts/visualization/visualize_hog_clustering.py \
    --classes 6 9 \
    --samples-per-class 200 \
    --output outputs/figures/hog_clustering_6vs9.png
```

### 3. 减少计算时间

```bash
# 使用更少样本
python scripts/visualization/visualize_hog_clustering.py \
    --samples-per-class 50 \
    --output outputs/figures/hog_clustering_fast.png
```

---

## 📁 输出文件

### 一键脚本生成的文件

```
outputs/figures/
├── hog_clustering_pca.png      # 2D PCA（最常用）
├── hog_clustering_tsne.png     # 2D t-SNE（局部清晰）
├── hog_clustering_3d.png       # 3D PCA（立体视角）
└── hog_clustering_5class.png   # 5类别对比
```

---

## 💡 实验建议

### 对比实验

1. **参数影响**
   - 横向对比：同一风格下，不同HOG配置
   - 观察：哪个配置聚类最紧密？

2. **风格鲁棒性**
   - 纵向对比：同一配置下，不同数据风格
   - 观察：聚类结构是否保持？

3. **降维方法**
   - PCA vs t-SNE
   - 观察：哪个更能展示聚类结构？

### 预期发现

- **配置2（pixels_per_cell=[8,8]）** 通常最稳定
- **原始数据** 聚类效果最好
- **高/低亮度** 聚类结构保持，说明 HOG 对亮度变化鲁棒

---

## 🛠️ 技术细节

### 降维方法对比

| 方法 | 计算速度 | 保持全局结构 | 保持局部结构 | 确定性 | 推荐场景 |
|------|----------|-------------|-------------|--------|----------|
| PCA | 快 | ✓ | × | ✓ | 快速探索 |
| t-SNE | 慢 | × | ✓ | × | 精细分析 |

### 特征维度计算

```
HOG 维度 = orientations × n_cells_x × n_cells_y × cells_per_block²

例如（28×28 图像）：
- pixels_per_cell=[4,4]: 7×7 = 49 cells
  → 9 × 7 × 7 × 4 = 1764 维
  
- pixels_per_cell=[8,8]: 3×3 = 9 cells
  → 9 × 3 × 3 × 4 = 324 维
```

---

## ❓ 常见问题

### Q1: t-SNE 运行很慢？

**A**: 
```bash
# 减少样本数
python scripts/visualization/visualize_hog_clustering.py \
    --reduction tsne \
    --samples-per-class 50 \  # 从150减到50
    --output outputs/figures/hog_clustering_tsne.png
```

### Q2: 为什么需要降维？

**A**: HOG 特征维度高（324-2352维），无法直接可视化。降维到 2D/3D 便于观察聚类结构。

### Q3: PCA 的解释方差是什么意思？

**A**: 
```
PC1 (45%) - 第一主成分保留了 45% 的信息
PC2 (25%) - 第二主成分保留了 25% 的信息
PC3 (15%) - 第三主成分保留了 15% 的信息
总计 85% - 3个主成分保留了 85% 的原始信息
```

### Q4: 如何选择要可视化的类别？

**A**:
- **易区分**: 0, 1, 8（形状差异大）
- **易混淆**: 6和9, 3和8, 4和9
- **全部**: 0-9（需要更多计算时间）

---

## 📖 相关文档

- **详细指南**: `docs/hog_clustering_visualization_guide.md`
- **实验3总结**: `EXPERIMENT3_STATUS.md`
- **HOG 特征说明**: 实验配置中的 `feature_config`

---

## ✅ 已创建的文件

1. `scripts/visualization/visualize_hog_clustering.py` - 2D 可视化脚本
2. `scripts/visualization/visualize_hog_clustering_3d.py` - 3D 可视化脚本
3. `scripts/run_hog_clustering_vis.sh` - 一键运行脚本
4. `docs/hog_clustering_visualization_guide.md` - 详细指南
5. `HOG_CLUSTERING_VISUALIZATION.md` - 本文档

---

**立即开始可视化 HOG 特征！** 🎨

```bash
bash scripts/run_hog_clustering_vis.sh
```


