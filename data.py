from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from ucimlrepo import fetch_ucirepo 

# function using seed
def seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

# Function to extract features
def extractX(df: pd.DataFrame) -> List[str]:
    xs = [c for c in df.columns if c.lower().startswith("x")]
    def key(c):
        try:
            return int("".join(ch for ch in c if ch.isdigit()))
        except ValueError:
            return 10**9
    return sorted(xs, key=key)

# Function to load synthetic dataset from csv
def load_synthetic_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    x_cols = extractX(df)
    assert "y" in df.columns, f"Missing 'y' column in {csv_path}"
    X = df[x_cols].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return X, y

# functions to load datasets from UCI
def load_concrete() -> Tuple[np.ndarray, np.ndarray] | None:
    try:
        from ucimlrepo import fetch_ucirepo
        concrete = fetch_ucirepo(id=165)  # Concrete Compressive Strength
        X = concrete.data.features.to_numpy(dtype=float)
        y = concrete.data.targets.to_numpy(dtype=float).reshape(-1)
        return X, y
    except Exception:
        return None

def load_yacht(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    x_cols = extractX(df)
    assert "y" in df.columns, f"Missing 'y' column in {csv_path}"
    X = df[x_cols].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return X, y

def load_wine(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    x_cols = extractX(df)
    assert "y" in df.columns, f"Missing 'y' column in {csv_path}"
    X = df[x_cols].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return X, y


# generate subsample
def subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]

# generate train subsample
def subsample_train(Xtr: np.ndarray, ytr: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample exactly n rows from train, deterministic from (seed, n)."""
    if n >= Xtr.shape[0]:
        return Xtr, ytr
    rng = seed_rng(seed * 1000 + n)
    sel = rng.choice(Xtr.shape[0], size=n, replace=False)
    return Xtr[sel], ytr[sel]

# aged iid split (not used)
def split_iid(X: np.ndarray, y: np.ndarray, seed: int, test_size: float = 0.30) -> Tuple[np.ndarray, ...]:
    """70/30 i.i.d. split per seed."""
    n = X.shape[0]
    rng = seed_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# Interpolation split with validation set
def split_iid_with_val(X, y, train=0.70, val=0.15, test=0.15, seed=0) -> Tuple:
    n = X.shape[0]
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    ntr = int(round(n * train))
    nva = int(round(n * val))
    itr, iva, ite = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]
    return X[itr], y[itr], X[iva], y[iva], X[ite], y[ite]

# Extrapolation split
def split_extrap(X, y, j_star: int, tail: str = "upper", q: float = 0.9, val_frac: float = 0.15, seed: int = 0):
    xj = X[:, j_star]

    if tail == "upper":
        thr = np.nanquantile(xj, q)
        test_mask = xj >= thr
    else:
        thr = np.nanquantile(xj, 1.0 - q)
        test_mask = xj <= thr

    # fallback if test is empty
    if test_mask.sum() == 0:
        finite = np.isfinite(xj)
        xjf = xj[finite]
        k = max(1, int(np.ceil((1.0 - q) * xjf.shape[0])))
        order = np.argsort(xjf)
        sel_local = order[-k:] if tail == "upper" else order[:k]
        test_mask = np.zeros_like(xj, dtype=bool)
        test_mask[np.where(finite)[0][sel_local]] = True

    central_mask = ~test_mask
    Xc, yc = X[central_mask], y[central_mask]
    Xt, yt = X[test_mask], y[test_mask]

    n = Xc.shape[0]
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    nva = int(round(n * val_frac))
    iva, itr = idx[:nva], idx[nva:]
    return Xc[itr], yc[itr], Xc[iva], yc[iva], Xt, yt

