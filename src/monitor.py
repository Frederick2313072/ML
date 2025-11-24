import os
import warnings
import pandas as pd


class BoostMonitor:
    def __init__(
        self, noise_indices, clean_indices, is_data_noisy=False, checkpoint_interval=50
    ):
        # data
        self.noise_indices = noise_indices
        self.clean_indices = clean_indices
        self.is_data_noisy = is_data_noisy

        # model history
        self.sample_weights_history = []
        self.noisy_weight_history = []
        self.clean_weight_history = []
        self.error_without_weight_history = []
        self.error_history = []
        self.alpha_history = []

        # validation history
        self.val_acc_history = []
        self.val_f1_history = []
        # checkpoint
        self.checkpoint_interval = checkpoint_interval

    def record_before_boost(self, sample_weight):
        """记录 boost 开始前的信息"""
        self.sample_weights_history.append(sample_weight.copy())
        if self.is_data_noisy:
            self.noisy_weight_history.append(sample_weight[self.noise_indices].sum())
            self.clean_weight_history.append(sample_weight[self.clean_indices].sum())

    def record_after_boost(
        self,
        estimator_error,
        estimator_weight,
        iboost,
        total,
        error_without_weight=None,
    ):
        """记录 boost 结束后的信息和日志"""
        if estimator_error is not None:
            self.error_history.append(estimator_error)
            self.alpha_history.append(estimator_weight)
        if error_without_weight is not None:
            self.error_without_weight_history.append(error_without_weight)

        self._log_boost_info(
            iboost,
            total,
            estimator_error,
            estimator_weight,
            error_without_weight,
        )

    def _log_boost_info(
        self,
        iboost,
        total,
        estimator_error,
        estimator_weight,
        error_without_weight=None,
    ):
        """统一的日志打印函数"""

        # 每 5 轮打印一次
        if (iboost + 1) % 5 != 0:
            return

        # 基础信息
        msg = (
            f"Boost {iboost + 1}/{total} | "
            f"error = {estimator_error:.4f} | "
            f"alpha = {estimator_weight:.4f}"
        )

        if error_without_weight is not None:
            msg += f" | unweighted_err = {error_without_weight:.4f}"

        if self.is_data_noisy:
            msg += f" | noisy_w = {self.noisy_weight_history[-1]:.6f}"

        print(msg)

    def record_validation(self, iboost, acc, f1):
        self.val_acc_history.append(acc)
        self.val_f1_history.append(f1)

        print(f"[VAL] round={iboost:03d} | acc={acc:.4f} | f1={f1:.4f}")

    def auto_checkpoint(self, iboost, dir="results", prefix="monitor_checkpoint"):
        """
        每隔 interval 轮自动保存监控数据，以防实验中断造成数据丢失。
        """

        # 只有在达到间隔时才保存
        if (iboost + 1) % self.checkpoint_interval != 0:
            return

        rounds = len(self.error_history)

        data = {
            "round": list(range(1, rounds + 1)),
            "weighted_error": self.error_history,
            "alpha": self.alpha_history,
        }

        # 普通错误率
        if len(self.error_without_weight_history) == rounds:
            data["unweighted_error"] = self.error_without_weight_history
        else:
            data["unweighted_error"] = [None] * rounds

        # noisy / clean 权重（已经是 before-boost）
        if self.is_data_noisy:
            data["noisy_weight"] = self.noisy_weight_history[:rounds]
            data["clean_weight"] = self.clean_weight_history[:rounds]
        else:
            data["noisy_weight"] = [None] * rounds
            data["clean_weight"] = [None] * rounds

        df = pd.DataFrame(data)

        # 自动生成 checkpoint 文件名
        filename = f"{prefix}_round_{iboost + 1:04d}.csv"
        path = os.path.join(dir, filename)
        df.to_csv(path, index=False)

        print(f"[CHECKPOINT] Saved '{path}' (round={iboost + 1}, rows={len(df)})")

    def dump(self, filename="monitor_log.csv"):
        """
        将所有监控数据保存为 CSV 文件（最终版）
        自动对齐所有历史记录，缺失部分会用 None 填充。
        """

        # 使用 error_history 作为主轴（每轮 after_boost 必记一条）
        rounds = len(self.error_history)

        if rounds == 0:
            warnings.warn("没有任何训练记录，dump 取消。")
            return

        # 构建主数据结构
        data = {
            "round": list(range(1, rounds + 1)),
            "weighted_error": self.error_history,
            "alpha": self.alpha_history,
        }

        # 普通错误率（unweighted）
        if len(self.error_without_weight_history) == rounds:
            data["unweighted_error"] = self.error_without_weight_history
        else:
            data["unweighted_error"] = [None] * rounds

        #  noisy / clean 权重均值
        if self.is_data_noisy:
            data["noisy_weight"] = self.noisy_weight_history[:rounds]
            data["clean_weight"] = self.clean_weight_history[:rounds]
        else:
            data["noisy_weight"] = [None] * rounds
            data["clean_weight"] = [None] * rounds

        # validation 指标
        if len(self.val_acc_history) == rounds:
            data["val_acc"] = self.val_acc_history
        else:
            data["val_acc"] = [None] * rounds

        if len(self.val_f1_history) == rounds:
            data["val_f1"] = self.val_f1_history
        else:
            data["val_f1"] = [None] * rounds

        # 输出 CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

        print(f"Monitor dumped to '{filename}' (rows={len(df)})")
