import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from pathlib import Path


def load_experiment_results(exp_name: str, base_dir: str = "experiments") -> dict:
    """
    加载单个实验的结果摘要
    
    Parameters
    ----------
    exp_name : str
        实验名称
    base_dir : str
        实验根目录，默认 "experiments"
    
    Returns
    -------
    dict
        包含实验关键指标的字典
    """
    monitor_path = Path(base_dir) / exp_name / "results" / "monitor.joblib"
    
    if not monitor_path.exists():
        raise FileNotFoundError(f"找不到实验结果: {monitor_path}")
    
    monitor = joblib.load(monitor_path)
    
    # 提取配置信息（如果有的话）
    config_path = Path(base_dir) / exp_name / "config.json"
    config_info = {}
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            config_info = {
                "feature": config.get("data", {}).get("use_feature", "unknown"),
                "depth": config.get("model", {}).get("estimator", {}).get("max_depth", "?"),
                "lr": config.get("model", {}).get("learning_rate", "?"),
            }
    
    return {
        "name": exp_name,
        "feature": config_info.get("feature", "unknown"),
        "depth": config_info.get("depth", "?"),
        "lr": config_info.get("lr", "?"),
        "final_val_acc": monitor.val_acc_history[-1] if monitor.val_acc_history else None,
        "best_val_acc": max(monitor.val_acc_history) if monitor.val_acc_history else None,
        "best_val_acc_round": monitor.val_acc_history.index(max(monitor.val_acc_history)) + 1 if monitor.val_acc_history else None,
        "final_val_f1": monitor.val_f1_history[-1] if monitor.val_f1_history else None,
        "best_val_f1": max(monitor.val_f1_history) if monitor.val_f1_history else None,
        "best_val_f1_round": monitor.val_f1_history.index(max(monitor.val_f1_history)) + 1 if monitor.val_f1_history else None,
        "final_train_acc": monitor.acc_on_train_data[-1] if monitor.acc_on_train_data else None,
        "n_estimators": len(monitor.error_history),
    }


def compare_experiments(
    experiment_names: list, 
    save_path: str = None,
    base_dir: str = "experiments"
):
    """
    对比多个实验的性能指标
    
    Parameters
    ----------
    experiment_names : list[str]
        实验名称列表，如 ["compare_hog", "compare_hu", "compare_original"]
    save_path : str, optional
        保存路径
    base_dir : str
        实验根目录，默认 "experiments"
    
    Returns
    -------
    pd.DataFrame
        包含所有实验对比数据的DataFrame
    """
    print("\n" + "█" * 80)
    print("实验对比分析".center(80))
    print("█" * 80)
    
    # 加载所有实验
    results = []
    for name in experiment_names:
        try:
            result = load_experiment_results(name, base_dir)
            results.append(result)
            print(f"✓ 已加载: {name}")
        except Exception as e:
            print(f"✗ 加载失败 {name}: {e}")
    
    if not results:
        raise ValueError("没有成功加载任何实验结果！")
    
    df = pd.DataFrame(results)
    
    # 打印对比表格
    print("\n" + "=" * 80)
    print("实验对比结果".center(80))
    print("=" * 80)
    
    # 格式化输出
    display_df = df[["name", "feature", "depth", "lr", "best_val_acc", "final_val_acc", "best_val_f1", "final_val_f1"]].copy()
    display_df.columns = ["实验名", "特征", "深度", "学习率", "最佳验证Acc", "最终验证Acc", "最佳验证F1", "最终验证F1"]
    print(display_df.to_string(index=False))
    
    # 找出最佳实验
    print("\n" + "=" * 80)
    print("🏆 最佳模型推荐".center(80))
    print("=" * 80)
    
    best_by_val_acc = df.loc[df['best_val_acc'].idxmax()]
    best_by_val_f1 = df.loc[df['best_val_f1'].idxmax()]
    
    print(f"\n✨ 最佳验证准确率: {best_by_val_acc['name']}")
    print(f"   特征类型: {best_by_val_acc['feature']}")
    print(f"   准确率: {best_by_val_acc['best_val_acc']:.4f} (轮次: {best_by_val_acc['best_val_acc_round']})")
    print(f"   F1分数: {best_by_val_acc['best_val_f1']:.4f}")
    
    print(f"\n✨ 最佳验证F1: {best_by_val_f1['name']}")
    print(f"   特征类型: {best_by_val_f1['feature']}")
    print(f"   F1分数: {best_by_val_f1['best_val_f1']:.4f} (轮次: {best_by_val_f1['best_val_f1_round']})")
    print(f"   准确率: {best_by_val_f1['best_val_acc']:.4f}")
    
    # 检测过拟合
    print("\n" + "=" * 80)
    print("🔍 过拟合检测".center(80))
    print("=" * 80)
    
    for _, row in df.iterrows():
        if row['final_train_acc'] is not None:
            gap = row['final_train_acc'] - row['final_val_acc']
            status = "⚠️ 可能过拟合" if gap > 0.05 else "✓ 正常"
            print(f"{row['name']:30s} 训练-验证差距: {gap:.4f}  {status}")
    
    print("=" * 80)
    
    # 生成对比图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('实验性能对比', fontsize=16, fontweight='bold')
    
    x = range(len(results))
    names_short = [name.replace('compare_', '') for name in df['name']]
    
    # 子图1：验证准确率对比
    ax1 = axes[0]
    bars1 = ax1.bar([i - 0.2 for i in x], df['best_val_acc'], width=0.4, 
                     alpha=0.8, label='最佳验证Acc', color='steelblue')
    bars2 = ax1.bar([i + 0.2 for i in x], df['final_val_acc'], width=0.4,
                     alpha=0.8, label='最终验证Acc', color='coral')
    
    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names_short, rotation=0, ha='center')
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.set_title('验证准确率对比', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([min(df['best_val_acc'].min(), df['final_val_acc'].min()) - 0.02, 1.0])
    
    # 子图2：F1对比
    ax2 = axes[1]
    bars3 = ax2.bar([i - 0.2 for i in x], df['best_val_f1'], width=0.4,
                     alpha=0.8, label='最佳验证F1', color='forestgreen')
    bars4 = ax2.bar([i + 0.2 for i in x], df['final_val_f1'], width=0.4,
                     alpha=0.8, label='最终验证F1', color='gold')
    
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(names_short, rotation=0, ha='center')
    ax2.set_ylabel('F1分数', fontsize=12)
    ax2.set_title('F1分数对比', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([min(df['best_val_f1'].min(), df['final_val_f1'].min()) - 0.02, 1.0])
    
    # 子图3：综合对比（准确率 vs F1）
    ax3 = axes[2]
    for i, row in df.iterrows():
        ax3.scatter(row['best_val_acc'], row['best_val_f1'], 
                   s=200, alpha=0.7, label=names_short[i])
        ax3.annotate(names_short[i], 
                    (row['best_val_acc'], row['best_val_f1']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax3.set_xlabel('最佳验证准确率', fontsize=12)
    ax3.set_ylabel('最佳验证F1', fontsize=12)
    ax3.set_title('准确率 vs F1 散点图', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 对比图保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return df


def visualize_training_data(
    data, save_path=None, save_individual=False, output_dir="dummy_output"
):
    """
    可视化训练数据（重新布局：噪声 → 权重分布 → 误差 → alpha → acc → f1）
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Training Analysis from Saved Results (n={data['n_estimators']})",
        fontsize=16,
        fontweight="bold",
    )

    rounds = data["rounds"]
    val_idx = data["val_idx"]

    # ----------------------------------------------------------------------
    # 1. 左上：噪声样本 vs 干净样本权重（核心：AdaBoost 嘘声放大机制）
    # ----------------------------------------------------------------------
    ax1 = axes[0, 0]
    if data["is_data_noisy"] and len(data["noisy_weight_history"]) > 0:
        ax1.plot(
            rounds,
            data["noisy_weight_history"],
            "r-",
            linewidth=2,
            label="Noisy Samples",
            marker="o",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
        ax1.plot(
            rounds,
            data["clean_weight_history"],
            "g-",
            linewidth=2,
            label="Clean Samples",
            marker="s",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
        ax1.axhline(0.5, color="black", linestyle="--", alpha=0.3)
        ax1.set_title("Noisy vs Clean Sample Weights", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Boosting Round")
        ax1.set_ylabel("Total Weight")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "N/A\n(Clean Data)", ha="center", va="center", fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])

    # ----------------------------------------------------------------------
    # 2. 左下：样本权重分布（权重集中情况→过拟合特征）
    # ----------------------------------------------------------------------
    ax2 = axes[1, 0]
    if "sample_weights_history" in data and len(data["sample_weights_history"]) > 0:
        key_rounds = [0, len(rounds) // 3, len(rounds) * 2 // 3, len(rounds) - 1]
        data_to_plot, labels = [], []

        for idx in key_rounds:
            if idx < len(data["sample_weights_history"]):
                data_to_plot.append(data["sample_weights_history"][idx])
                labels.append(f"R{idx + 1}")

        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
        for box in bp["boxes"]:
            box.set_facecolor("lightblue")
            box.set_alpha(0.7)

        ax2.set_title("Sample Weight Distribution", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Sample Weight")
        ax2.grid(True, axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "N/A\n(Not in CSV)", ha="center", va="center", fontsize=14)
        ax2.set_xticks([])
        ax2.set_yticks([])

    # ----------------------------------------------------------------------
    # 3. 中上：错误率演化
    # ----------------------------------------------------------------------
    ax3 = axes[0, 1]
    ax3.plot(rounds, data["error_history"], "b-", linewidth=2, label="Weighted Error")
    if len(data["error_without_weight_history"]) > 0:
        ax3.plot(
            rounds,
            data["error_without_weight_history"],
            "r--",
            linewidth=2,
            label="Unweighted Error",
            alpha=0.7,
        )
    ax3.set_title("Error Rate Evolution", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Boosting Round")
    ax3.set_ylabel("Error Rate")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ----------------------------------------------------------------------
    # 4. 中下：Alpha 系数演化（弱学习器强度）
    # ----------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.plot(
        rounds,
        data["alpha_history"],
        "g-",
        linewidth=2,
        marker="o",
        markersize=4,
        markevery=max(1, len(rounds) // 20),
    )
    avg_alpha = np.mean(data["alpha_history"])
    ax4.axhline(
        avg_alpha,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Mean={avg_alpha:.3f}",
    )
    ax4.set_title("Alpha Coefficient Evolution", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Boosting Round")
    ax4.set_ylabel("Alpha")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ----------------------------------------------------------------------
    # 5. 右上：训练 vs 验证准确率
    # ----------------------------------------------------------------------
    ax5 = axes[0, 2]
    if len(data["acc_on_train_data"]) > 0:
        ax5.plot(
            val_idx,
            data["acc_on_train_data"],
            "b-",
            linewidth=2,
            label="Train Accuracy",
            marker="o",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
    if len(data["val_acc_history"]) > 0:
        ax5.plot(
            val_idx,
            data["val_acc_history"],
            "r-",
            linewidth=2,
            label="Val Accuracy",
            marker="s",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
    ax5.set_title("Accuracy Evolution", fontsize=14, fontweight="bold")
    ax5.set_xlabel("Boosting Round")
    ax5.set_ylabel("Accuracy")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ----------------------------------------------------------------------
    # 6. 右下：F1 演化
    # ----------------------------------------------------------------------
    ax6 = axes[1, 2]
    if len(data["f1_on_training_data"]) > 0:
        ax6.plot(
            val_idx,
            data["f1_on_training_data"],
            "b-",
            linewidth=2,
            label="Train F1",
            marker="o",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
    if len(data["val_f1_history"]) > 0:
        ax6.plot(
            val_idx,
            data["val_f1_history"],
            "r-",
            linewidth=2,
            label="Val F1",
            marker="s",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
    ax6.set_title("F1 Score Evolution", fontsize=14, fontweight="bold")
    ax6.set_xlabel("Boosting Round")
    ax6.set_ylabel("F1 Score")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # ----------------------------------------------------------------------
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n[Viz] Figure saved to: {save_path}")
    else:
        plt.show()

    # save every subplot individually
    if save_individual:
        os.makedirs(output_dir, exist_ok=True)

        subplot_titles = [
            "noisy_vs_clean",
            "sample_weight_distribution",
            "error_evolution",
            "alpha_evolution",
            "accuracy_evolution",
            "f1_evolution",
        ]

        # 原图宽度（用于计算缩放比例）
        orig_fig_width = fig.get_figwidth()

        for ax, name in zip(axes.flatten(), subplot_titles):
            fig_single = plt.figure(figsize=(6, 4), dpi=300)
            new_ax = fig_single.add_subplot(111)

            # 缩放因子
            scale = fig_single.get_figwidth() / orig_fig_width

            for line in ax.get_lines():
                x = line.get_xdata()
                y = line.get_ydata()

                # --- 缩放 markevery（防止 marker 密集） ---
                orig_markevery = line.get_markevery()
                if isinstance(orig_markevery, int):
                    markevery_small = max(
                        1,
                        int(
                            orig_markevery
                            * (orig_fig_width / fig_single.get_figwidth())
                        ),
                    )
                else:
                    markevery_small = orig_markevery

                # --- 缩放 marker 大小 ---
                orig_ms = line.get_markersize()
                new_ms = orig_ms * scale if orig_ms else None

                new_ax.plot(
                    x,
                    y,
                    linestyle=line.get_linestyle(),
                    marker=line.get_marker(),
                    color=line.get_color(),
                    linewidth=line.get_linewidth() * scale,
                    markersize=new_ms,
                    markevery=markevery_small,
                )

            new_ax.set_title(ax.get_title(), fontsize=14, fontweight="bold")
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            new_ax.grid(True, alpha=0.3)

            single_path = os.path.join(output_dir, f"{name}.png")
            fig_single.savefig(single_path, dpi=300, bbox_inches="tight")
            plt.close(fig_single)
            print(f"[Viz] saved: {single_path}")
    plt.close()
