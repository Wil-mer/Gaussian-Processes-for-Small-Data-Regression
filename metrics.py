from __future__ import annotations
import numpy as np
import math


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nll_gaussian(y_true: np.ndarray, mu: np.ndarray, std: np.ndarray) -> float:
    var = np.clip(std ** 2, 1e-12, None)
    return float(0.5 * np.mean(np.log(2 * math.pi * var) + ((y_true - mu) ** 2) / var))


def coverage(y_true: np.ndarray, mu: np.ndarray, std: np.ndarray, z: float = 1.96) -> float:
    lo = mu - z * std
    hi = mu + z * std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def sharpness(std: np.ndarray) -> float:
    return float(np.mean(std))