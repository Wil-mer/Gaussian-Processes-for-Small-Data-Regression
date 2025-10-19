from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- helpers -----------------
def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def aggregate_mean_se(df, metrics=("rmse","nll","cov95","sharp")):
    agg = df.groupby(["dataset","model","regime","n"], as_index=False)[list(metrics)].agg(["mean","sem"])
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    return agg.reset_index(drop=True)

def mfmt(m, s):  # mean ± SE as string
    return f"{m:.3f} ± {s:.3f}"

def plot_learning_curve(agg, dataset, metric,
                                  model_order=("RBF_iso","Mat32_iso"),
                                  regimes=("iid","extra_upper"),
                                  out_path=None, title=None):
    sub = agg[(agg["dataset"]==dataset) & (agg["model"].isin(model_order)) & (agg["regime"].isin(regimes))].copy()
    if sub.empty:
        print(f"[WARN] {dataset} saknas ({metric})"); return
    plt.figure(figsize=(7,4))
    for model in model_order:
        for reg in regimes:
            ssub = sub[(sub["model"]==model) & (sub["regime"]==reg)].sort_values("n")
            if ssub.empty: 
                continue
            x = ssub["n"].to_numpy()
            y = ssub[f"{metric}_mean"].to_numpy()
            e = ssub[f"{metric}_sem"].to_numpy()
            label = f"{model} [{reg}]"
            plt.errorbar(x, y, yerr=e, marker="o", linestyle="-", label=label, capsize=3)
    plt.xlabel("Train size n"); plt.ylabel(metric.upper())
    if title: plt.title(title)
    plt.legend(); plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=200); print(f"[OK] saved {out_path}")
    plt.close()

def write_delta_n200(df, out_csv):
    sub = df[df["n"]==200].copy()
    if sub.empty: return
    wide = (sub.pivot_table(index=["dataset","model","seed"],
                            columns="regime", values=["rmse","nll"]))
    # kräver kolumnerna iid och extra_upper
    for m in ["rmse","nll"]:
        wide[(m,"delta_extrap_minus_iid")] = wide[(m,"extra_upper")] - wide[(m,"iid")]
    # medel över seed
    delta = (wide.groupby(["dataset","model"])
                   [[("rmse","delta_extrap_minus_iid"), ("nll","delta_extrap_minus_iid")]]
                   .mean())
    delta.columns = ["rmse_delta", "nll_delta"]
    delta.reset_index().to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")


def plot_bars_n200(df, dataset, metric,
                             model_order=("Mat32_iso","Mat32_ARD"),
                             regimes=("iid","extra_upper"),
                             out_path=None, title=None):
    sub = df[(df["dataset"]==dataset) & (df["n"]==200) & (df["regime"].isin(regimes))].copy()
    if sub.empty: 
        print(f"[WARN] {dataset} @ n=200 saknas"); return
    stats = (sub.groupby(["regime","model"])[metric]
             .agg(["mean","sem"]).unstack("regime").reindex(model_order))
    if stats.empty: return
    x = np.arange(len(model_order))
    width = 0.35
    plt.figure(figsize=(6,4))
    for k, reg in enumerate(regimes):
        y = stats[("mean", reg)].to_numpy()
        e = stats[("sem",  reg)].to_numpy()
        plt.bar(x + k*width, y, width, yerr=e, capsize=3, label=reg)
    plt.xticks(x + width/2, model_order); plt.ylabel(metric.upper())
    if title: plt.title(title)
    plt.legend(); plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=200); print(f"[OK] saved {out_path}")
    plt.close()



# ----------------- main -----------------
def plot():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "results.csv",
                    help="Path till results.csv")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parents[2] / "figs",
                    help="Katalog där figurer sparas")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    df = pd.read_csv(args.csv)
    required_cols = {"dataset","model","n","seed","rmse","nll","cov95","sharp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"results.csv saknar kolumner: {sorted(missing)}")

    # Aggregera seeds → mean & SE per dataset × model × n
    agg = aggregate_mean_se(df, metrics=("rmse","nll","cov95","sharp"))

    # ----- Figurer -----
    # F1: Learning curve (RMSE) på rbf_smooth
    plot_learning_curve(agg, dataset="rbf_smooth", metric="rmse",
                        model_order=("RBF_iso","Mat32_iso"),
                        out_path=args.outdir / "F1_learning_curve_rbf_smooth_RMSE.png",
                        title="Learning curve (RMSE) – rbf_smooth")

    # F2: Learning curve (NLL) på matern32_rough
    plot_learning_curve(agg, dataset="matern32_rough", metric="nll",
                        model_order=("RBF_iso","Mat32_iso"),
                        out_path=args.outdir / "F2_learning_curve_matern32_rough_NLL.png",
                        title="Learning curve (NLL) – matern32_rough")

    # F3: ARD (n=200) – två separata staplar (RMSE, NLL) på matern32_ard
    plot_bars_n200(df, dataset="matern32_ard", metric="rmse",
                   model_order=("Mat32_iso","Mat32_ARD"),
                   out_path=args.outdir / "F3a_ARD_matern32_n200_RMSE.png",
                   title="Matérn ν=3/2 – ARD vs iso @ n=200 (RMSE)")
    plot_bars_n200(df, dataset="matern32_ard", metric="nll",
                   model_order=("Mat32_iso","Mat32_ARD"),
                   out_path=args.outdir / "F3b_ARD_matern32_n200_NLL.png",
                   title="Matérn ν=3/2 – ARD vs iso @ n=200 (NLL)")

    # Empirical 1 Concrete bars @ n=200 (RMSE, NLL)
    plot_bars_n200(df, dataset="concrete", metric="rmse",
                    model_order=("RBF_iso","Mat32_iso"),
                    out_path=args.outdir / "F_extra_concrete_n200_RMSE.png",
                    title="Concrete – RBF vs Mat32 @ n=200 (RMSE)")
    plot_bars_n200(df, dataset="concrete", metric="nll",
                    model_order=("RBF_iso","Mat32_iso"),
                    out_path=args.outdir / "F_extra_concrete_n200_NLL.png",
                    title="Concrete – RBF vs Mat32 @ n=200 (NLL)")
       
        
    # Emprical 2 Concrete Learning curve (RMSE/NLL)
    # Empirical – family (iso vs iso): RMSE + NLL
    for ds in ("concrete", "yacht", "wine"):
        plot_learning_curve(agg, dataset=ds, metric="rmse",
                            model_order=("RBF_iso","Mat32_iso"),
                            out_path=args.outdir / f"E_{ds}_family_RMSE.png",
                            title=f"Family (iso): RMSE — {ds.title()}")
        plot_learning_curve(agg, dataset=ds, metric="nll",
                            model_order=("RBF_iso","Mat32_iso"),
                            out_path=args.outdir / f"E_{ds}_family_NLL.png",
                            title=f"Family (iso): NLL — {ds.title()}")

    # Empirical – parameterization (Matérn): bars @ n=200
    for ds in ("concrete", "yacht", "wine"):
        plot_bars_n200(df, dataset=ds, metric="rmse",
                                model_order=("Mat32_iso","Mat32_ARD"),
                                out_path=args.outdir / f"E_{ds}_param_Mat32_RMSE_n200.png",
                                title=f"Parameterization (Matérn 3/2): RMSE @ n=200 — {ds.title()}")
        plot_bars_n200(df, dataset=ds, metric="nll",
                                model_order=("Mat32_iso","Mat32_ARD"),
                                out_path=args.outdir / f"E_{ds}_param_Mat32_NLL_n200.png",
                                title=f"Parameterization (Matérn 3/2): NLL @ n=200 — {ds.title()}")


    # ----- Tabell n=200 -----
    write_delta_n200(df, out_csv=args.outdir / "T1_summary_n200.csv")

