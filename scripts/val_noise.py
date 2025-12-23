import os
import json
import joblib
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adalab.monitor import BoostMonitor
from adalab.io import load_compressed
from adalab.evaluation import val_after_train_parallel
from adalab.workflow import (
    load_config,
    prep_testing_data_from_config,
    prep_training_data_from_config,
)


EXP_DIR = Path("../batch_exp/overfit/noise1_hog_depth2")
use_feature = "hog"


feature_config = {
    "hog_params": {
        "orientations": 9,
        "pixels_per_cell": [2, 2],
        "cells_per_block": [2, 2],
    }
}
test_shift_config = {
    "as_train": {"ratio": 0.2, "label_flip": True, "gaussian": {"std": 0.05}}
}


def prep_for_val(
    config_path: str | Path,
    course_folder: str = "./data/test_images",
):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    exp_name = config["experiment"]["name"]

    print(
        f"\033[36m[Pipeline] \nLoading visualization for existing experiment: {exp_name}\033[0m"
    )
    exp_dir = EXP_DIR
    print(f"\033[36m[Pipeline] Using experiment {exp_dir}\033[0m")

    result_dir = exp_dir / "results"
    clf_path = result_dir / "model.joblib.xz"
    monitor_path = result_dir / "monitor.joblib.xz"
    clf = load_compressed(clf_path)
    monitor: BoostMonitor = load_compressed(monitor_path)

    train_split = prep_training_data_from_config(config)
    test_split = prep_testing_data_from_config(config, train_split, course_folder)
    # breakpoint()

    alphas = np.asarray(monitor.alpha_history)
    return clf, alphas, train_split, test_split


def plot_curves(
    val_idx: np.ndarray,
    train_curve: np.ndarray,
    val_curve: np.ndarray,
    val_noise_curve: np.ndarray,
    ylabel: str,
    title: str,
    save_path: str | None = None,
):
    sns.set_theme(
        style="whitegrid",
        context="talk",  # 比 default 更适合论文/汇报
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        val_idx,
        train_curve,
        label="Train",
        linewidth=2,
        marker="o",
    )
    plt.plot(
        val_idx,
        val_curve,
        label="Validation",
        linewidth=2,
        marker="s",
    )
    plt.plot(
        val_idx,
        val_noise_curve,
        label="Validation (Noise)",
        linewidth=2,
        marker="^",
    )

    plt.xlabel("Boosting Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def main():
    config_path = "../configs/overfit_configs/noise1_hog_depth2.json"
    clf, alphas, train_split, test_split = prep_for_val(
        config_path, course_folder="../data/test_images"
    )
    acc_curv_train, f1_curv_train, val_idx = val_after_train_parallel(
        clf, alphas, X=train_split.X_train, y=train_split.y_train, val_freq=10, n_jobs=4
    )
    acc_curv_val, f1_curv_val, val_idx = val_after_train_parallel(
        clf, alphas, X=train_split.X_test, y=train_split.y_test, val_freq=10, n_jobs=4
    )

    acc_curv_val_noise, f1_curv_val_noise, val_idx = val_after_train_parallel(
        clf,
        alphas,
        X=test_split.X_mnist_shift["as_training"],
        y=test_split.y_mnist,
        val_freq=10,
        n_jobs=4,
    )
    # ===== Accuracy 曲线 =====
    plot_curves(
        val_idx=val_idx,
        train_curve=acc_curv_train,
        val_curve=acc_curv_val,
        val_noise_curve=acc_curv_val_noise,
        ylabel="Accuracy",
        title="Accuracy vs Boosting Rounds",
        save_path="acc_curve.png",
    )

    # ===== F1 曲线 =====
    plot_curves(
        val_idx=val_idx,
        train_curve=f1_curv_train,
        val_curve=f1_curv_val,
        val_noise_curve=f1_curv_val_noise,
        ylabel="F1 Score",
        title="F1 Score vs Boosting Rounds",
        save_path="f1_curve.png",
    )


if __name__ == "__main__":
    main()
