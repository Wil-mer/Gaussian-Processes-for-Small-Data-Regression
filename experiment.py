from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.pipeline import Pipeline

from .data import load_concrete, load_yacht, load_wine, load_synthetic_csv, split_iid_with_val, split_extrap, subsample_train
from .models import make_gpr_iso, make_gpr_ard_rbf, make_gpr_ard_matern
from .metrics import rmse, nll_gaussian, coverage, sharpness


def eval_model(pipe: Pipeline, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> Dict[str, float]:
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tysta opt-varningar
        pipe.fit(Xtr, ytr)
    mu, std = pipe.predict(Xte, return_std=True)
    fit_time = time.perf_counter() - t0
    return {
        "rmse": rmse(yte, mu),
        "nll": nll_gaussian(yte, mu, std),
        "cov95": coverage(yte, mu, std, z=1.96),
        "sharp": sharpness(std),
        "fit_time_s": fit_time,
    }

def make_splits(X, y, *, seed, regime, j_star=None, q=0.9):
    if regime == "iid":
        return split_iid_with_val(X, y, train=0.70, val=0.15, test=0.15, seed=seed)
    elif regime == "extra_upper":
        assert j_star is not None
        return split_extrap(X, y, j_star=j_star, tail="upper", q=q, val_frac=0.15, seed=seed)
    elif regime == "extra_lower":
        assert j_star is not None
        return split_extrap(X, y, j_star=j_star, tail="lower", q=1-q, val_frac=0.15, seed=seed)
    else:
        raise ValueError(regime)
    
def run_tests(cfg, DATA_DIR) -> pd.DataFrame:
    rows = []

    # ---------- Load datasets ----------
    paths = {
        "rbf_smooth": DATA_DIR / "rbf_smooth.csv",
        "matern32_rough": DATA_DIR / "matern32_rough.csv",
        "matern32_ard": DATA_DIR / "matern32_ard.csv",
    }
    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Saknar {name}: {p}")
        X, y = load_synthetic_csv(p)
        data[name] = (X, y)

    conc = load_concrete()
    yacht = load_yacht(DATA_DIR/"yacht_hydrodynamics.csv")
    wine = load_wine(DATA_DIR/"winequality-red.csv")
    if conc is not None:
        data["concrete"] = conc
    if yacht is not None:
        data["yacht"] = yacht
    if wine is not None:
        data["wine"] = wine

    regimes = ["iid", "extra_upper"]
    
    # ---------- A: smooth (RBF vs Matern iso) ----------
    for dataset in ["rbf_smooth"]:
        X, y = data[dataset]
        j_star = int(np.nanargmax(np.nanstd(X, axis=0))) # feature with largest std in X
        for seed in cfg.seeds:
            for regime in regimes:
                Xtr, ytr, Xva, yva, Xte, yte = make_splits(X, y, seed=seed, regime=regime, j_star=j_star, q=0.9)
                for n in cfg.n_list:
                    if n > len(ytr): 
                        continue
                    Xn, yn = subsample_train(Xtr, ytr, n, seed)
                    d = X.shape[1]
                    for fam, model_name in [("RBF", "RBF_iso"), ("Matern", "Mat32_iso")]:
                        pipe = make_gpr_iso(kernel_family=fam, d=d, nu=cfg.nu, n_restarts=cfg.restarts)
                        # Kan ignorera val här om du inte ändrat eval_model-signaturen:
                        met = eval_model(pipe, Xn, yn, Xte, yte)
                        rows.append({
                            "dataset": dataset, "regime": regime,
                            "n": n, "seed": seed,
                            "model": model_name, "kernel_family": fam,
                            "ard": False, "nu": (cfg.nu if fam == "Matern" else ""),
                            **met
                        })


    # ---------- B: rough (RBF vs Matern iso) ----------
    for dataset in ["matern32_rough"]:
        X, y = data[dataset]
        j_star = int(np.nanargmax(np.nanstd(X, axis=0)))
        d = X.shape[1]
        for seed in cfg.seeds:
            for regime in regimes:
                Xtr, ytr, Xva, yva, Xte, yte = make_splits(X, y, seed=seed, regime=regime, j_star=j_star, q=0.9)
                for n in cfg.n_list:
                    if n > len(ytr): 
                        continue
                    Xn, yn = subsample_train(Xtr, ytr, n, seed)
                    for fam, model_name in [("RBF", "RBF_iso"), ("Matern", "Mat32_iso")]:
                        pipe = make_gpr_iso(kernel_family=fam, d=d, nu=cfg.nu, n_restarts=cfg.restarts)
                        met = eval_model(pipe, Xn, yn, Xte, yte)
                        rows.append({
                            "dataset": dataset, "regime": regime,
                            "n": n, "seed": seed,
                            "model": model_name, "kernel_family": fam,
                            "ard": False, "nu": (cfg.nu if fam == "Matern" else ""),
                            **met
                    })

    # ---------- C: ARD test (Mat32 iso vs Mat32 ARD), n=200 ----------
    for dataset in ["matern32_ard"]:
        X, y = data[dataset]
        d = X.shape[1]
        n = 200
        j_star = int(np.nanargmax(np.nanstd(X, axis=0)))
        for seed in cfg.seeds:
            for regime in regimes:
                Xtr, ytr, Xva, yva, Xte, yte = make_splits(X, y, seed=seed, regime=regime, j_star=j_star, q=0.9)
                if n > len(ytr): 
                    continue
                Xn, yn = subsample_train(Xtr, ytr, n, seed)

                pipe_iso = make_gpr_iso(kernel_family="Matern", d=d, nu=cfg.nu, n_restarts=cfg.restarts)
                met_iso = eval_model(pipe_iso, Xn, yn, Xte, yte)
                rows.append({
                    "dataset": dataset, "regime": regime, "n": n, "seed": seed,
                    "model": "Mat32_iso", "kernel_family": "Matern", "ard": False, "nu": cfg.nu, **met_iso
                })

                pipe_ard = make_gpr_ard_matern(d=d, nu=cfg.nu, n_restarts=cfg.restarts)
                met_ard = eval_model(pipe_ard, Xn, yn, Xte, yte)
                rows.append({
                    "dataset": dataset, "regime": regime, "n": n, "seed": seed,
                    "model": "Mat32_ARD", "kernel_family": "Matern", "ard": True, "nu": cfg.nu, **met_ard
                })


    # ---------- D: empirical datasets ----------
    if "concrete" in data:
        X, y = data["concrete"]; d = X.shape[1]
        j_star = int(np.nanargmax(np.nanstd(X, axis=0)))
        for seed in cfg.seeds:
            for regime in regimes:
                Xtr, ytr, Xva, yva, Xte, yte = make_splits(X, y, seed=seed, regime=regime, j_star=j_star, q=0.9)
                for n in cfg.n_list:
                    if n > len(ytr): 
                        continue
                    Xn, yn = subsample_train(Xtr, ytr, n, seed)
                    candidates = [
                        ("RBF_iso","RBF",False, make_gpr_iso("RBF", d, n_restarts=cfg.restarts)),
                        ("RBF_ARD","RBF",True,  make_gpr_ard_rbf(d, n_restarts=cfg.restarts)),
                        ("Mat32_iso","Matern",False, make_gpr_iso("Matern", d, nu=cfg.nu, n_restarts=cfg.restarts)),
                        ("Mat32_ARD","Matern",True,  make_gpr_ard_matern(d, nu=cfg.nu, n_restarts=cfg.restarts)),
                    ]
                    for model_name, fam, is_ard, pipe in candidates:
                        met = eval_model(pipe, Xn, yn, Xte, yte)
                        rows.append({
                            "dataset": "concrete", "regime": regime,
                            "n": n, "seed": seed, "model": model_name,
                            "kernel_family": fam, "ard": is_ard,
                            "nu": (cfg.nu if fam == "Matern" else ""), **met
                        })
    if "yacht" in data:
        X, y = data["yacht"]; d = X.shape[1]
        j_star = int(np.nanargmax(np.nanstd(X, axis=0)))
        for seed in cfg.seeds:
            for regime in regimes:
                Xtr, ytr, Xva, yva, Xte, yte = make_splits(X, y, seed=seed, regime=regime, j_star=j_star, q=0.9)
                for n in cfg.n_list:
                    if n > len(ytr): 
                        continue
                    Xn, yn = subsample_train(Xtr, ytr, n, seed)
                    candidates = [
                        ("RBF_iso",   "RBF",    False, make_gpr_iso(kernel_family="RBF",    d=d, n_restarts=cfg.restarts)),
                        ("RBF_ARD",   "RBF",     True, make_gpr_ard_rbf(d=d, n_restarts=cfg.restarts)),
                        ("Mat32_iso", "Matern", False, make_gpr_iso(kernel_family="Matern", d=d, nu=cfg.nu, n_restarts=cfg.restarts)),
                        ("Mat32_ARD", "Matern",  True, make_gpr_ard_matern(d=d, nu=cfg.nu, n_restarts=cfg.restarts)),
                    ]
                    for model_name, fam, is_ard, pipe in candidates:
                        met = eval_model(pipe, Xn, yn, Xte, yte)
                        rows.append({
                            "dataset": "yacht","regime": regime,
                            "n": n, "seed": seed, "model": model_name,
                            "kernel_family": fam, "ard": is_ard,
                            "nu": (cfg.nu if fam == "Matern" else ""), **met
                        })
    if "wine" in data:
        X, y = data["wine"]; d = X.shape[1]
        j_star = int(np.nanargmax(np.nanstd(X, axis=0)))
        for seed in cfg.seeds:
            for regime in regimes:
                Xtr, ytr, Xva, yva, Xte, yte = make_splits(X, y, seed=seed, regime=regime, j_star=j_star, q=0.9)
                for n in cfg.n_list:
                    if n > len(ytr): 
                        continue
                    Xn, yn = subsample_train(Xtr, ytr, n, seed)
                    candidates = [
                        ("RBF_iso",   "RBF",    False, make_gpr_iso(kernel_family="RBF",    d=d, n_restarts=cfg.restarts)),
                        ("RBF_ARD",   "RBF",     True, make_gpr_ard_rbf(d=d, n_restarts=cfg.restarts)),
                        ("Mat32_iso", "Matern", False, make_gpr_iso(kernel_family="Matern", d=d, nu=cfg.nu, n_restarts=cfg.restarts)),
                        ("Mat32_ARD", "Matern",  True, make_gpr_ard_matern(d=d, nu=cfg.nu, n_restarts=cfg.restarts)),
                    ]
                    for model_name, fam, is_ard, pipe in candidates:
                        met = eval_model(pipe, Xn, yn, Xte, yte)
                        rows.append({
                            "dataset": "wine", "regime": regime,
                            "n": n, "seed": seed, "model": model_name,
                            "kernel_family": fam, "ard": is_ard,
                            "nu": (cfg.nu if fam == "Matern" else ""), **met
                    })    
    df = pd.DataFrame(rows)
    return df
