class BoostMonitor:
    def __init__(self, noise_indices, clean_indices):
        self.noise_indices = noise_indices
        self.clean_indices = clean_indices

        self.sample_weights_history = []
        self.noisy_weight_history = []
        self.clean_weight_history = []
        self.error_history = []
        self.alpha_history = []

    def record_before_boost(self, sample_weight):
        """记录 boost 开始前的信息"""
        self.sample_weights_history.append(sample_weight.copy())
        self.noisy_weight_history.append(sample_weight[self.noise_indices].mean())
        self.clean_weight_history.append(sample_weight[self.clean_indices].mean())

    def record_after_boost(self, estimator_error, estimator_weight, iboost, total):
        """记录 boost 结束后的信息和日志"""
        if estimator_error is not None:
            self.error_history.append(estimator_error)
            self.alpha_history.append(estimator_weight)

            if (iboost + 1) % 5 == 0:
                print(
                    f"Boost {iboost + 1}/{total} | "
                    f"error = {estimator_error:.4f} | "
                    f"alpha = {estimator_weight:.4f} | "
                    f"noisy_w = {self.noisy_weight_history[-1]:.6f}"
                )
