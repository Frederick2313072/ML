import warnings


class BoostMonitor:
    def __init__(self, noise_indices, clean_indices, n_estimators, is_data_noisy=False):
        self.n_estimators = n_estimators
        self.noise_indices = noise_indices
        self.clean_indices = clean_indices
        self.is_data_noisy = is_data_noisy

        self.sample_weights_history = []
        self.noisy_weight_history = []
        self.clean_weight_history = []
        self.error_without_weight_history = []
        self.error_history = []
        self.alpha_history = []

    def record_before_boost(self, sample_weight):
        """记录 boost 开始前的信息"""
        self.sample_weights_history.append(sample_weight.copy())
        if self.is_data_noisy:
            self.noisy_weight_history.append(sample_weight[self.noise_indices].mean())
            self.clean_weight_history.append(sample_weight[self.clean_indices].mean())

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

    def dump(self, filename="monitor_log.csv"):
        """
        将监控数据保存为 CSV 文件
        """
        # 训练轮数检查（使用 warning 而非 raise）
        if len(self.error_history) != self.n_estimators:
            warnings.warn(
                f"训练似乎未完成！当前记录轮次 = {len(self.error_history)}, "
                f"期望轮次 = {self.n_estimators}. "
                "将继续导出当前已有的数据。",
                RuntimeWarning,
            )
        import pandas as pd

        rounds = len(self.error_history)  # 以 error_history 为基准对齐

        data = {
            "round": list(range(1, rounds + 1)),
            "weighted_error": self.error_history,
            "alpha": self.alpha_history,
        }

        # 若存在普通错误率
        if len(self.error_without_weight_history) == rounds:
            data["unweighted_error"] = self.error_without_weight_history
        else:
            # 保持列齐整，用 NaN 填充
            data["unweighted_error"] = [None] * rounds

        # 若是 noisy 数据，则添加 noisy & clean 权重
        if self.is_data_noisy:
            # noisy_weight_history 是 BEFORE boost 记录的，会多 1 个元素
            # → 对齐后从 index=0 截取 rounds 个
            data["noisy_weight_mean"] = self.noisy_weight_history[:rounds]
            data["clean_weight_mean"] = self.clean_weight_history[:rounds]
        else:
            data["noisy_weight_mean"] = [None] * rounds
            data["clean_weight_mean"] = [None] * rounds

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

        print(f"Monitor dumped to '{filename}' (rows={len(df)})")
