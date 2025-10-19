from __future__ import annotations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

# functions to generate kernels

def make_gpr_iso(kernel_family: str, d: int, nu: float = 1.5, n_restarts: int = 5) -> Pipeline:
    
    if kernel_family == "RBF":
        base = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    elif kernel_family == "Matern":
        base = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=nu)
    else:
        raise ValueError("kernel_family måste vara 'RBF' eller 'Matern'")

    kernel = C(1.0, (1e-3, 1e3)) * base + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                   n_restarts_optimizer=n_restarts, random_state=0)
    pipe = Pipeline([
        ("xscaler", StandardScaler()),
        ("gpr", gpr),
    ])
    return pipe


def make_gpr_ard_rbf(d: int, n_restarts: int = 5) -> Pipeline:
    """RBF with ARD: one lengthscale per feature."""
    base = RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3))
    kernel = C(1.0, (1e-3, 1e3)) * base + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                   n_restarts_optimizer=n_restarts, random_state=0)
    pipe = Pipeline([
        ("xscaler", StandardScaler()),
        ("gpr", gpr),
    ])
    return pipe

def make_gpr_ard_matern(d: int, nu: float = 1.5, n_restarts: int = 5) -> Pipeline:
    """Matérn(v) with ARD: one lengthscale per feature."""
    base = Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3), nu=nu)
    kernel = C(1.0, (1e-3, 1e3)) * base + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                   n_restarts_optimizer=n_restarts, random_state=0)
    pipe = Pipeline([
        ("xscaler", StandardScaler()),
        ("gpr", gpr),
    ])
    return pipe