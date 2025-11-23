"""
可视化AdaBoost过拟合过程
简洁的脚本，展示模型随着弱学习器数量增加的过拟合行为
"""

from sklearn.tree import DecisionTreeClassifier
from src.utils import prepare_data
from src.evaluation import visualize_overfitting_process


def main():
    """主函数：可视化过拟合过程"""

    print("\n" + "█" * 60)
    print("AdaBoost 过拟合可视化".center(56))
    print("█" * 60)

    # ========== 1. 选择数据类型 ==========
    print("\n选择数据类型:")
    print("1. 干净数据（无噪声）")
    print("2. 含噪声数据（5%噪声）")
    print("3. 含噪声数据（10%噪声）")

    # 默认使用选项2
    choice = 2  # 可以修改为1或3

    if choice == 1:
        noise_ratio = 0
        data_type = "干净数据"
    elif choice == 2:
        noise_ratio = 0.05
        data_type = "5%噪声数据"
    else:
        noise_ratio = 0.10
        data_type = "10%噪声数据"

    print(f"\n使用: {data_type}")
    print("-" * 60)

    # ========== 2. 准备数据 ==========
    print("\n准备数据...")
    X_train, X_test, y_train, y_test, _, _ = prepare_data(noise_ratio=noise_ratio)

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # ========== 3. 选择配置 ==========
    print("\n" + "=" * 60)
    print("配置选项".center(56))
    print("=" * 60)

    # 配置1: 快速测试（推荐）
    config = {
        "base_estimator": DecisionTreeClassifier(max_depth=1),  # 决策树桩
        "n_estimators_list": [1, 5, 10, 20, 30, 40, 50, 75, 100],  # 测试点
        "learning_rate": 0.5,  # 学习率
        "random_state": 42,
    }

    # 配置2: 精细分析（更多测试点，需要更长时间）
    # config = {
    #     "base_estimator": DecisionTreeClassifier(max_depth=1),
    #     "n_estimators_list": list(range(1, 101, 5)),  # [1, 6, 11, ..., 96]
    #     "learning_rate": 0.5,
    #     "random_state": 42,
    # }

    # 配置3: 深树测试（观察更复杂基学习器的影响）
    # config = {
    #     "base_estimator": DecisionTreeClassifier(max_depth=3),
    #     "n_estimators_list": [1, 5, 10, 20, 30, 40, 50],
    #     "learning_rate": 0.5,
    #     "random_state": 42,
    # }

    print(f"基学习器: 决策树 (max_depth={config['base_estimator'].max_depth})")
    print(f"测试点数量: {len(config['n_estimators_list'])}")
    print(f"弱学习器范围: {config['n_estimators_list'][0]} - {config['n_estimators_list'][-1]}")
    print(f"学习率: {config['learning_rate']}")

    # ========== 4. 可视化过拟合 ==========
    print("\n开始训练和可视化...")
    print("-" * 60)

    results = visualize_overfitting_process(
        X_train,
        y_train,
        X_test,
        y_test,
        base_estimator=config["base_estimator"],
        n_estimators_list=config["n_estimators_list"],
        learning_rate=config["learning_rate"],
        random_state=config["random_state"],
        save_path=None,  # 设为路径可保存图表，如 'overfitting.png'
    )

    # ========== 5. 额外分析（可选） ==========
    print("\n" + "=" * 60)
    print("建议".center(56))
    print("=" * 60)

    best_idx = results["test_accuracy"].index(max(results["test_accuracy"]))
    best_n = results["n_estimators"][best_idx]
    final_n = results["n_estimators"][-1]
    final_overfit = results["overfitting_degree"][-1]

    # 根据结果给出建议
    if final_overfit > 0.15:
        print("\n⚠️  严重过拟合警告:")
        print(f"   - 当前过拟合程度: {final_overfit:.2%}")
        print(f"   - 建议减少弱学习器数量至 {best_n} 左右")
        print(f"   - 或使用更小的学习率（如 0.1）")
    elif final_overfit > 0.10:
        print("\n⚠️  中度过拟合:")
        print(f"   - 当前过拟合程度: {final_overfit:.2%}")
        print(f"   - 建议使用早停，在 n={best_n} 处停止训练")
    elif final_overfit < 0.05:
        print("\n✓ 模型拟合良好:")
        print(f"   - 过拟合程度低: {final_overfit:.2%}")
        print(f"   - 可以考虑增加弱学习器数量以提升性能")
    else:
        print("\n✓ 模型表现良好:")
        print(f"   - 过拟合程度: {final_overfit:.2%} (可接受)")
        print(f"   - 建议使用 n={best_n} 个弱学习器")

    # 噪声数据的额外建议
    if noise_ratio > 0:
        print(f"\n💡 噪声数据建议:")
        print(f"   - 当前数据有 {noise_ratio*100:.0f}% 噪声")
        print(f"   - AdaBoost 对噪声敏感，容易过拟合")
        print(f"   - 建议:")
        print(f"     1. 使用较少的弱学习器（{best_n} 左右）")
        print(f"     2. 降低学习率（从 0.5 到 0.3）")
        print(f"     3. 考虑数据清洗或噪声鲁棒方法")

    print("\n" + "=" * 60)
    print("\n✓ 可视化完成！")
    print("\n💡 提示:")
    print("   - 图表会自动显示（关闭窗口继续）")
    print("   - 要保存图表，设置 save_path='overfitting.png'")
    print("   - 要测试不同配置，修改脚本中的 config 字典")
    print("=" * 60)


if __name__ == "__main__":
    main()

