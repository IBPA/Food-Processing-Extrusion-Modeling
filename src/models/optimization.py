"""
Extrusion Condition Optimizer
==============================
Identifies optimal extrusion temperature and residence time for each
fixed formulation to maximize a weighted combination of nutrient density
(NRF9.3) and antioxidant capacity (FRAP).

Objective (Equation 8)
----------------------
    f(T, t) = α · N̂(T,t) + (1 − α) · F̂(T,t)

where N̂ and F̂ are min–max normalized NRF9.3 and FRAP over the
feasible search space, and α = 0.5 (equal weighting, as used in paper).

Decision variables
------------------
    T ∈ [100, 200] °C   (extrusion barrel temperature)
    t ∈ [0,   120] s    (mean residence time)

Solvers
-------
    - Differential Evolution  — primary global solver (scipy)
    - Dual Annealing          — primary global solver (scipy)
    - Grid search             — landscape visualization / coarse benchmarking
    - L-BFGS-B / Nelder-Mead  — local refinement / sensitivity diagnostics
"""

import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import (
    differential_evolution,
    dual_annealing,
    minimize,
)
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T_BOUNDS = (100.0, 200.0)   # °C
t_BOUNDS = (0.0,   120.0)   # s
ALPHA    = 0.5               # equal weighting of NRF and FRAP (paper default)

BIO_COLS_MAP = {
    "DPPH": "DPPH (mmol TE/100g)",
    "FRAP": "FRAP(mmol per g)",
    "ABTS": "ABTS (mmol TE/100g)",
    "ORAC": "ORAC (mmol TE/100g)",
}
METRICS_ORDER = ["NRF", "DPPH", "FRAP", "ABTS", "ORAC"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _parse_sample_id(sid: str) -> Tuple[str, float, float]:
    """Parses 'Name(140,30)' → ('Name', 140.0, 30.0)."""
    m = re.match(r"^(.+?)\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)$", str(sid).strip())
    if m:
        return m.group(1).strip(), float(m.group(2)), float(m.group(3))
    return sid, 0.0, 0.0


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Stores the result for a single formulation–solver run."""
    formulation: str
    solver: str
    T_opt: float
    t_opt: float
    nrf_pred: float
    frap_pred: float
    objective: float
    n_seeds: int
    seed_results: list


class ExtrusionOptimizer:
    """
    Optimizes extrusion conditions (T, t) for each formulation to maximize
    the weighted NRF + FRAP objective.

    Parameters
    ----------
    bioactivity_file : str
        CSV with columns: sample_id, FRAP, DPPH, ABTS, ORAC.
        sample_id format: 'FormulationName(T,t)'
    nrf_file : str
        CSV with columns: Formulation, NRF_Before, NRF_After, NRF_Optimized.
    alpha : float
        Weight on NRF9.3 (1−alpha applied to FRAP). Default 0.5.
    T_bounds : tuple
        (T_min, T_max) in °C.
    t_bounds : tuple
        (t_min, t_max) in seconds.
    n_seeds : int
        Number of independent random seeds for global solvers (robustness check).
    """

    def __init__(
        self,
        bioactivity_file: str,
        nrf_file: str,
        alpha: float = ALPHA,
        T_bounds: Tuple[float, float] = T_BOUNDS,
        t_bounds: Tuple[float, float] = t_BOUNDS,
        n_seeds: int = 5,
    ):
        self.bioactivity_file = bioactivity_file
        self.nrf_file = nrf_file
        self.alpha = alpha
        self.T_bounds = T_bounds
        self.t_bounds = t_bounds
        self.n_seeds = n_seeds

        self.df_: Optional[pd.DataFrame] = None
        self.nrf_df_: Optional[pd.DataFrame] = None
        self._interpolators_: Dict[str, RegularGridInterpolator] = {}
        self._norm_bounds_: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Loads bioactivity grid and NRF scores; builds interpolators."""
        df = pd.read_csv(self.bioactivity_file)

        # Parse sample_id into name, T, t
        parsed = df["sample_id"].apply(
            lambda x: pd.Series(_parse_sample_id(x),
                                 index=["formulation", "T", "t"])
        )
        df = pd.concat([df, parsed], axis=1)

        for col in BIO_COLS_MAP.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["total_bioactivity"] = df[list(BIO_COLS_MAP.values())].sum(axis=1)
        self.df_ = df

        # NRF scores
        try:
            self.nrf_df_ = pd.read_csv(self.nrf_file)
        except FileNotFoundError:
            self.nrf_df_ = pd.DataFrame()
            print("[Optimizer] NRF file not found — NRF terms will be zero.")

        # Build interpolators and compute global normalization bounds
        self._build_interpolators()
        self._compute_norm_bounds()

        formulations = df["formulation"].unique()
        print(f"[Optimizer] Loaded {len(df)} condition rows "
              f"across {len(formulations)} formulations.")

    def _build_interpolators(self) -> None:
        """
        For each formulation, build a 2-D grid interpolator over (T, t)
        for FRAP and NRF so the objective can be evaluated at arbitrary points.
        """
        df = self.df_
        for form in df["formulation"].unique():
            sub = df[df["formulation"] == form].copy()
            Ts = np.sort(sub["T"].unique())
            ts = np.sort(sub["t"].unique())

            if len(Ts) < 2 or len(ts) < 2:
                continue  # Not enough grid points to interpolate

            frap_grid = np.zeros((len(Ts), len(ts)))
            nrf_grid  = np.zeros((len(Ts), len(ts)))

            for i, T in enumerate(Ts):
                for j, t in enumerate(ts):
                    row = sub[(sub["T"] == T) & (sub["t"] == t)]
                    if not row.empty:
                        frap_grid[i, j] = row[BIO_COLS_MAP["FRAP"]].values[0]

            # NRF: load from nrf_df if available; otherwise interpolate a
            # flat surface at NRF_After (a reasonable first approximation)
            nrf_val = self._get_nrf_after(form)
            nrf_grid[:] = nrf_val  # Constant surface; replace with kinetic model if available

            self._interpolators_[form] = {
                "frap": RegularGridInterpolator(
                    (Ts, ts), frap_grid,
                    method="linear", bounds_error=False, fill_value=None
                ),
                "nrf": RegularGridInterpolator(
                    (Ts, ts), nrf_grid,
                    method="linear", bounds_error=False, fill_value=None
                ),
                "T_range": (Ts.min(), Ts.max()),
                "t_range": (ts.min(), ts.max()),
            }

    def _get_nrf_after(self, formulation: str) -> float:
        if self.nrf_df_ is None or self.nrf_df_.empty:
            return 1.0
        row = self.nrf_df_[self.nrf_df_["Formulation"] == formulation]
        if not row.empty and "NRF_After" in row.columns:
            return float(row["NRF_After"].values[0])
        return 1.0

    def _compute_norm_bounds(self) -> None:
        """Compute global min–max for FRAP and NRF across the full dataset."""
        frap_vals = self.df_[BIO_COLS_MAP["FRAP"]].dropna().values
        self._norm_bounds_ = {
            "frap_min": frap_vals.min() if len(frap_vals) else 0.0,
            "frap_max": frap_vals.max() if len(frap_vals) else 1.0,
        }
        if self.nrf_df_ is not None and not self.nrf_df_.empty and "NRF_After" in self.nrf_df_.columns:
            nrf_vals = self.nrf_df_["NRF_After"].dropna().values
            self._norm_bounds_["nrf_min"] = nrf_vals.min() if len(nrf_vals) else 0.0
            self._norm_bounds_["nrf_max"] = nrf_vals.max() if len(nrf_vals) else 1.0
        else:
            self._norm_bounds_["nrf_min"] = 0.0
            self._norm_bounds_["nrf_max"] = 1.0

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _normalize(self, val: float, lo: float, hi: float) -> float:
        if hi > lo:
            return (val - lo) / (hi - lo)
        return 0.0

    def _objective(self, x: np.ndarray, formulation: str) -> float:
        """
        Negative weighted objective (minimized by scipy solvers).

        f = α · N̂(T,t) + (1−α) · F̂(T,t)

        where N̂ and F̂ are min–max normalized over the feasible space.
        """
        T, t = x
        interp = self._interpolators_.get(formulation)
        if interp is None:
            return 1.0  # Penalty

        frap = float(interp["frap"]([[T, t]])[0])
        nrf  = float(interp["nrf"]([[T, t]])[0])

        b = self._norm_bounds_
        frap_n = self._normalize(frap, b["frap_min"], b["frap_max"])
        nrf_n  = self._normalize(nrf,  b["nrf_min"],  b["nrf_max"])

        return -(self.alpha * nrf_n + (1 - self.alpha) * frap_n)

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def optimize(
        self,
        formulation: str,
        method: str = "differential_evolution",
    ) -> OptimizationResult:
        """
        Runs the selected solver for a single formulation.

        Parameters
        ----------
        formulation : str
            Must match a key in the loaded dataset.
        method : str
            One of: 'differential_evolution', 'dual_annealing',
            'lbfgsb', 'nelder_mead'

        Returns
        -------
        OptimizationResult
        """
        if self.df_ is None:
            self.load_data()

        bounds = [self.T_bounds, self.t_bounds]
        seed_results = []

        for seed in range(self.n_seeds):
            if method == "differential_evolution":
                sol = differential_evolution(
                    self._objective,
                    bounds,
                    args=(formulation,),
                    seed=seed,
                    maxiter=1000,
                    tol=1e-8,
                    workers=1,
                )
            elif method == "dual_annealing":
                sol = dual_annealing(
                    self._objective,
                    bounds,
                    args=(formulation,),
                    seed=seed,
                    maxiter=5000,
                )
            elif method in ("lbfgsb", "nelder_mead"):
                x0 = np.array([
                    np.random.default_rng(seed).uniform(*self.T_bounds),
                    np.random.default_rng(seed + 1).uniform(*self.t_bounds),
                ])
                m = "L-BFGS-B" if method == "lbfgsb" else "Nelder-Mead"
                sol = minimize(
                    self._objective, x0,
                    args=(formulation,),
                    method=m,
                    bounds=bounds if method == "lbfgsb" else None,
                )
            else:
                raise ValueError(f"Unknown method: {method}. Choose from "
                                 "differential_evolution, dual_annealing, "
                                 "lbfgsb, nelder_mead.")

            seed_results.append({
                "seed": seed,
                "T_opt": sol.x[0],
                "t_opt": sol.x[1],
                "obj":   -sol.fun,
            })

        # Select best across seeds
        best = max(seed_results, key=lambda r: r["obj"])
        T_opt, t_opt = best["T_opt"], best["t_opt"]

        # Read back predicted values at optimal point
        interp = self._interpolators_[formulation]
        frap_pred = float(interp["frap"]([[T_opt, t_opt]])[0])
        nrf_pred  = float(interp["nrf"]([[T_opt, t_opt]])[0])

        result = OptimizationResult(
            formulation=formulation,
            solver=method,
            T_opt=round(T_opt, 2),
            t_opt=round(t_opt, 2),
            nrf_pred=round(nrf_pred, 4),
            frap_pred=round(frap_pred, 4),
            objective=round(best["obj"], 4),
            n_seeds=self.n_seeds,
            seed_results=seed_results,
        )
        self._print_result(result)
        return result

    def compare_solvers(
        self,
        formulation: str,
        methods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Runs all four solvers on a single formulation and returns a
        comparison table (Table 1 in paper).

        Parameters
        ----------
        formulation : str
        methods : list, optional
            Subset of solvers to compare. Defaults to all four.
        """
        if methods is None:
            methods = ["differential_evolution", "dual_annealing", "lbfgsb", "nelder_mead"]

        rows = []
        for m in methods:
            r = self.optimize(formulation, method=m)
            rows.append({
                "Solver": m,
                "T_opt (°C)": r.T_opt,
                "t_opt (s)": r.t_opt,
                "NRF_pred": r.nrf_pred,
                "FRAP_pred": r.frap_pred,
                "Objective f": r.objective,
            })

        comparison = pd.DataFrame(rows).set_index("Solver")
        print(f"\n{'='*55}")
        print(f"  Solver Comparison — {formulation}")
        print(f"{'='*55}")
        print(comparison.to_string())
        return comparison

    def optimize_all(
        self,
        method: str = "differential_evolution",
    ) -> pd.DataFrame:
        """
        Runs optimization for every formulation in the dataset.

        Returns a summary DataFrame with optimal T, t, and predicted outcomes.
        """
        if self.df_ is None:
            self.load_data()

        rows = []
        formulations = self.df_["formulation"].unique()

        for form in formulations:
            if form not in self._interpolators_:
                continue
            r = self.optimize(form, method=method)
            rows.append({
                "Formulation": form,
                "T_opt (°C)": r.T_opt,
                "t_opt (s)": r.t_opt,
                "NRF_pred": r.nrf_pred,
                "FRAP_pred": r.frap_pred,
                "Objective f": r.objective,
                "Solver": method,
            })

        summary = pd.DataFrame(rows)
        return summary

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_landscape(
        self,
        formulation: str,
        grid_size: int = 50,
        figsize: tuple = (14, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """
        2-D contour plot of the objective landscape over (T, t) with the
        optimal point marked. Separate panels for FRAP, NRF, and combined f.
        """
        if self.df_ is None:
            self.load_data()

        interp = self._interpolators_.get(formulation)
        if interp is None:
            print(f"[plot_landscape] Formulation '{formulation}' not found.")
            return

        Ts = np.linspace(*self.T_bounds, grid_size)
        ts = np.linspace(*self.t_bounds, grid_size)
        TT, tt = np.meshgrid(Ts, ts)
        pts = np.column_stack([TT.ravel(), tt.ravel()])

        frap_grid = interp["frap"](pts).reshape(grid_size, grid_size)
        nrf_grid  = interp["nrf"](pts).reshape(grid_size, grid_size)

        b = self._norm_bounds_
        frap_n = (frap_grid - b["frap_min"]) / max(b["frap_max"] - b["frap_min"], 1e-9)
        nrf_n  = (nrf_grid  - b["nrf_min"])  / max(b["nrf_max"]  - b["nrf_min"],  1e-9)
        obj_grid = self.alpha * nrf_n + (1 - self.alpha) * frap_n

        # Find optimal point on grid
        idx = np.unravel_index(np.argmax(obj_grid), obj_grid.shape)
        T_best, t_best = Ts[idx[1]], ts[idx[0]]

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        panels = [
            (frap_grid, "FRAP (mmol Fe²⁺ eq./100 g)", "YlOrBr"),
            (nrf_grid,  "NRF9.3 Score",                "YlGn"),
            (obj_grid,  f"Combined f (α={self.alpha})", "RdYlGn"),
        ]

        for ax, (data, title, cmap) in zip(axes, panels):
            cf = ax.contourf(Ts, ts, data, levels=30, cmap=cmap)
            plt.colorbar(cf, ax=ax, shrink=0.85)
            ax.scatter([T_best], [t_best], c="red", s=120,
                       marker="*", zorder=5, label=f"Opt ({T_best:.0f}°C, {t_best:.0f}s)")
            ax.set_xlabel("Temperature (°C)", fontsize=10)
            ax.set_ylabel("Residence Time (s)", fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=8, loc="lower right")

        fig.suptitle(f"Objective Landscape — {formulation}", fontsize=13, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_condition_comparison(
        self,
        formulation: str,
        figsize: tuple = (8, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Bar chart comparing FRAP and NRF across three conditions:
        Raw (T=0, t=0), Reference (T=140°C, t=30s), and Optimal.
        """
        if self.df_ is None:
            self.load_data()

        sub = self.df_[self.df_["formulation"] == formulation]
        conditions = {
            "Raw\n(T=0, t=0)":        sub[(sub["T"] == 0) & (sub["t"] == 0)],
            "Reference\n(T=140, t=30)": sub[(sub["T"] == 140) & (sub["t"] == 30)],
            "Optimal":                  sub.loc[[sub["total_bioactivity"].idxmax()]],
        }

        labels, frap_vals, nrf_vals = [], [], []
        for label, rows in conditions.items():
            labels.append(label)
            if not rows.empty:
                frap_vals.append(rows[BIO_COLS_MAP["FRAP"]].values[0])
                nrf_vals.append(rows["NRF"].values[0] if "NRF" in rows.columns else np.nan)
            else:
                frap_vals.append(np.nan)
                nrf_vals.append(np.nan)

        x = np.arange(len(labels))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        bars1 = ax1.bar(x - width / 2, frap_vals, width,
                        label="FRAP", color="steelblue", alpha=0.85)
        bars2 = ax2.bar(x + width / 2, nrf_vals, width,
                        label="NRF9.3", color="seagreen", alpha=0.85)

        ax1.set_xlabel("Processing Condition", fontsize=11)
        ax1.set_ylabel("FRAP (mmol Fe²⁺ eq./100 g)", color="steelblue", fontsize=10)
        ax2.set_ylabel("NRF9.3 Score", color="seagreen", fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.tick_params(axis="y", colors="steelblue")
        ax2.tick_params(axis="y", colors="seagreen")
        ax1.set_title(f"FRAP & NRF9.3 by Processing Condition\n{formulation}", fontsize=12)

        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _print_result(self, r: OptimizationResult) -> None:
        print(f"\n[{r.solver}] {r.formulation}")
        print(f"  Optimal T  = {r.T_opt:.1f} °C")
        print(f"  Optimal t  = {r.t_opt:.1f} s")
        print(f"  NRF pred   = {r.nrf_pred:.4f}")
        print(f"  FRAP pred  = {r.frap_pred:.4f}")
        print(f"  Objective  = {r.objective:.4f}  (α={self.alpha})")
        print(f"  Seeds used = {r.n_seeds}")

    def summary_table(self, results: List[OptimizationResult]) -> pd.DataFrame:
        """Converts a list of OptimizationResults to a summary DataFrame."""
        return pd.DataFrame([
            {
                "Formulation": r.formulation,
                "Solver": r.solver,
                "T_opt (°C)": r.T_opt,
                "t_opt (s)": r.t_opt,
                "NRF_pred": r.nrf_pred,
                "FRAP_pred": r.frap_pred,
                "Objective f": r.objective,
            }
            for r in results
        ])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize extrusion conditions")
    parser.add_argument("--bio",    required=True, help="Bioactivity CSV path")
    parser.add_argument("--nrf",    required=True, help="NRF scores CSV path")
    parser.add_argument("--form",   default=None,  help="Single formulation name (or 'all')")
    parser.add_argument("--method", default="differential_evolution",
                        choices=["differential_evolution", "dual_annealing",
                                 "lbfgsb", "nelder_mead"])
    parser.add_argument("--alpha",  type=float, default=0.5)
    parser.add_argument("--seeds",  type=int, default=5)
    parser.add_argument("--save",   default=None)
    args = parser.parse_args()

    opt = ExtrusionOptimizer(
        bioactivity_file=args.bio,
        nrf_file=args.nrf,
        alpha=args.alpha,
        n_seeds=args.seeds,
    )
    opt.load_data()

    if args.form is None or args.form == "all":
        summary = opt.optimize_all(method=args.method)
        print(summary.to_string(index=False))
        if args.save:
            summary.to_csv(f"{args.save}/optimization_summary.csv", index=False)
    else:
        result = opt.optimize(args.form, method=args.method)
        opt.plot_landscape(
            args.form,
            save_path=f"{args.save}/landscape_{args.form}.png" if args.save else None,
        )
        opt.plot_condition_comparison(
            args.form,
            save_path=f"{args.save}/conditions_{args.form}.png" if args.save else None,
        )
        opt.compare_solvers(args.form)
