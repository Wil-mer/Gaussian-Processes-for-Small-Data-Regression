from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- kernels ----------
# pairwise
def _pairwise_r(X: np.ndarray, ell: np.ndarray) -> np.ndarray:
    # ARD (len(ell)=d) eller isotrop (len(ell)=1)
    Z = X / ell if ell.size > 1 else X / ell[0]
    sq = np.sum(Z*Z, axis=1, keepdims=True) + np.sum(Z*Z, axis=1) - 2*Z@Z.T
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq)

# rbf
def k_rbf(X: np.ndarray, ell: np.ndarray, sigma_f: float=1.0) -> np.ndarray:
    r = _pairwise_r(X, ell)
    return (sigma_f**2) * np.exp(-0.5 * r**2)

# matern32
def k_matern32(X: np.ndarray, ell: np.ndarray, sigma_f: float=1.0) -> np.ndarray:
    r = np.sqrt(3.0) * _pairwise_r(X, ell)
    return (sigma_f**2) * (1.0 + r) * np.exp(-r)

# matern52
def k_matern52(X: np.ndarray, ell: np.ndarray, sigma_f: float=1.0) -> np.ndarray:
    r = np.sqrt(5.0) * _pairwise_r(X, ell)
    return (sigma_f**2) * (1.0 + r + (r**2)/3.0) * np.exp(-r)
# ----------  ----------  ----------  ----------

# sample from GP prior 
def gp_sample(X: np.ndarray, kernel_fn, ell: np.ndarray, sigma_f: float = 1.0, sigma: float = 0.1, jitter: float = 1e-8, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    K = kernel_fn(X, ell=ell, sigma_f=sigma_f)
    K[np.diag_indices_from(K)] += jitter
    L = np.linalg.cholesky(K)
    f = L @ rng.standard_normal(len(X))
    y = f + sigma * rng.standard_normal(len(X))
    return f, y

# ---------- dataset generators ----------
def synth_smooth_rbf(n: int = 500, d: int = 8, ell: float = 1.0, sigma_f: float = 1.0, sigma: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    f, y = gp_sample(X, k_rbf, ell=np.array([ell], dtype=float),
                     sigma_f=sigma_f, sigma=sigma, seed=seed)
    return X, f, y

def synth_rough_ard_matern(n: int = 500, d: int = 8, relevant: int = 3, ell_relevant: float = 0.5, ell_irrelevant: float = 10.0, nu: str = "3/2", sigma_f: float = 1.0, sigma: float = 0.1, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert 0 < relevant <= d
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    ell = np.ones(d, dtype=float) * float(ell_irrelevant)
    ell[:relevant] = float(ell_relevant)
    kernel = k_matern32 if nu == "3/2" else k_matern52
    f, y = gp_sample(X, kernel, ell=ell, sigma_f=sigma_f, sigma=sigma, seed=seed)
    return X, f, y

def synth_rough_matern(n: int = 500, d: int = 8, ell: float = 1.0,
                       sigma_f: float = 1.0, sigma: float = 0.1,
                       nu: str = "3/2", seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    kernel = k_matern32 if nu == "3/2" else k_matern52
    # isotropisk längdskala: ell som skalar alla dimensioner lika
    f, y = gp_sample(X, kernel, ell=np.array([ell], dtype=float),
                     sigma_f=sigma_f, sigma=sigma, seed=seed)
    return X, f, y
# ----------  ----------  ----------  ----------

# .csv save-function
def to_csv(X: np.ndarray, y: np.ndarray, path: Path, f: np.ndarray | None = None) -> None:
    df = pd.DataFrame(X, columns=[f"X{j+1}" for j in range(X.shape[1])])
    if f is not None:
        df["f"] = f
    df["y"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def main():
    # parameters
    n        = 1000
    d        = 8
    ell      = 1.0          # isotrop lengthscale for smooth/rough
    sigma    = 0.1
    sigma_f  = 1.0
    nu       = "3/2"        # Matérn-smoothness
    # ARD-spec
    relevant = 3
    ell_rel  = 0.5
    ell_irr  = 10.0

    HERE = Path(__file__).resolve()
    PROJECT_DIR = HERE.parents[1]
    DATA_DIR = PROJECT_DIR / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)


    include_f = True       # Set true to include noise-free f
    
    # seeds (reproducability)
    base_seed = 10


    # 1) Smooth (RBF, isotrop)
    X, f, y = synth_smooth_rbf(n=n, d=d, ell=ell, sigma_f=sigma_f, sigma=sigma, seed=base_seed)
    to_csv(X, y, DATA_DIR / "rbf_smooth.csv", f if include_f else None)
    print(f"[smooth]  -> { (DATA_DIR/'rbf_smooth.csv').resolve() }")

    # 2) Rough (Matérn 3/2, isotrop)
    X, f, y = synth_rough_matern(n=n, d=d, ell=ell, sigma_f=sigma_f, sigma=sigma, nu=nu, seed=base_seed+1)
    to_csv(X, y, DATA_DIR / "matern32_rough.csv", f if include_f else None)
    print(f"[rough]   -> { (DATA_DIR/'matern32_rough.csv').resolve() }")

    # 3) ARD (Matérn 3/2, kort ℓ på relevanta, lång ℓ på irrelevanta)
    X, f, y = synth_rough_ard_matern(
        n=n, d=d, relevant=relevant, ell_relevant=ell_rel, ell_irrelevant=ell_irr,
        nu=nu, sigma_f=sigma_f, sigma=sigma, seed=base_seed+2
    )
    to_csv(X, y, DATA_DIR / "matern32_ard.csv", f if include_f else None)
    print(f"[ard]     -> { (DATA_DIR/'matern32_ard.csv').resolve() }")

    print("Done. (columns: X1..X{d}" + (", f" if include_f else "") + ", y)")

if __name__ == "__main__":
    main()
