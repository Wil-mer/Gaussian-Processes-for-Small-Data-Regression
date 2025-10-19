from pathlib import Path
from typing import List
from dataclasses import dataclass

from src.experiment import run_tests
from src.plotting import plot


# paths
HERE = Path(__file__).resolve()
PROJECT_DIR = HERE.parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGS_DIR = PROJECT_DIR / "figs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "results.csv"

# config class
@dataclass
class RunConfig:
    seeds: List[int] = None
    n_list: List[int] = None
    nu: float = 1.5
    restarts: int = 5

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [1, 2, 3, 4, 5]
        if self.n_list is None:
            self.n_list = [20, 50, 100, 200]

def main():
    print(f"[INFO] Data dir:     {DATA_DIR}")
    print(f"[INFO] Results dir:  {RESULTS_DIR}")
    cfg = RunConfig()
    df = run_tests(cfg, DATA_DIR)
    df.to_csv(RESULTS_CSV, index=False)
    meta = {
        "seeds": [1, 2, 3, 4, 5],
        "n_list": [20, 50, 100, 200],
        "nu": 1.5,
        "restarts": 5,
        "datasets": sorted(df["dataset"].unique().tolist())
    }
    print(f"Experiment parameters\n")
    for key,value in meta.items():
        print(key, ":",value)
    print(f"[OK] Wrote results: {RESULTS_CSV}")
    print(df.groupby(["dataset", "model", "n"])[["rmse", "nll"]].mean().round(3))
    plot(); print(f"plots saved in: {FIGS_DIR}")

if __name__ == "__main__":
    main()
