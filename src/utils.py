import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def prepare_data(noise_ratio=0.05, test_size=0.2, random_state=42):
    """
    下载 MNIST，并按指定比例添加标签噪声。
    自动返回：
        - X_train, X_test
        - y_train (含噪声) , y_test
        - train_noise_indices  (训练集内部噪声索引)
        - train_clean_indices  (训练集内部干净索引)
    若 noise_ratio=0，则返回完全干净的数据。

    Parameters
    ----------
    noise_ratio : float
        噪声比例（0 ~ 1），表示标签噪声的比例。
        若为 0，则不添加标签噪声。

    test_size : float
        train_test_split 的测试集占比

    random_state : int
        随机种子

    Returns
    -------
    X_train, X_test : ndarray
    y_train, y_test : ndarray
    train_noise_indices : ndarray (训练集内部的噪声样本位置)
    train_clean_indices : ndarray
    """

    print("Downloading MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.int64)
    X = X / 255.0

    n_samples = len(y)

    # -----------------------------------------
    # Case 1: 不添加噪声，返回原始数据
    # -----------------------------------------
    if noise_ratio <= 0:
        print("No noise added, returning clean dataset.")

        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, np.arange(n_samples), test_size=test_size, random_state=random_state
        )

        # 训练集全部是 clean
        train_noise_indices = np.array([], dtype=int)
        train_clean_indices = np.arange(len(y_train))

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            train_noise_indices,
            train_clean_indices,
        )

    # Case 2: 添加噪声
    n_noisy = int(n_samples * noise_ratio)
    rng = np.random.default_rng(random_state)

    noise_indices = rng.choice(n_samples, n_noisy, replace=False)

    y_noisy = y.copy()
    y_noisy[noise_indices] = rng.integers(0, 10, size=n_noisy)

    print(f"Injected label noise: {noise_ratio * 100:.1f}% ({n_noisy} samples)")

    # train/test split，保留原始索引
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y_noisy, np.arange(n_samples), test_size=test_size, random_state=random_state
    )

    # 计算训练集内部噪声位置
    train_noise_mask = np.isin(train_idx, noise_indices)
    train_noise_indices = np.where(train_noise_mask)[0]
    train_clean_indices = np.where(~train_noise_mask)[0]

    print(f"Training set noise samples = {len(train_noise_indices)}")

    return (X_train, X_test, y_train, y_test, train_noise_indices, train_clean_indices)
