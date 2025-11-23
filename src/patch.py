from sklearn.ensemble import AdaBoostClassifier

ori_boost = AdaBoostClassifier._boost


def boost_with_monitor(self, iboost, X, y, sample_weight, random_state):
    # ---- 记录 Boost 前 ----
    if hasattr(self, "_monitor"):
        self._monitor.record_before_boost(sample_weight)

    # ---- 调用原始 _boost ----
    sample_weight_new, estimator_weight, estimator_error = ori_boost(
        self, iboost, X, y, sample_weight, random_state
    )

    # ---- 记录 Boost 后 ----
    if hasattr(self, "_monitor"):
        self._monitor.record_after_boost(
            estimator_error, estimator_weight, iboost, self.n_estimators
        )

    return sample_weight_new, estimator_weight, estimator_error
