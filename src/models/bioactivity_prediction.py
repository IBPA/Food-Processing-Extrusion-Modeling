"""
Bioactivity Prediction Model (BPM)
====================================
Predicts formulation-level antioxidant capacity (FRAP) from
processing-adjusted chemical composition features using Random Forest.


Protocol
--------
- Target: log-transformed FRAP (mmol Fe2+ eq./100 g dry basis)
- Evaluation: 100 repeated random 80/20 train–test splits (formulation-level)
- Hyperparameter tuning: 3-fold GridSearchCV within each training split
- Best configuration (most frequent): n_estimators=200, max_depth=None,
  min_samples_split=10, min_samples_leaf=1, bootstrap=True
- Metrics: Pearson r (PCC), R², RMSE, MAE
"""

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Hyperparameter search grid (matches paper, Section 2.2.5)
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap":        [True, False],
}

# Best configuration selected most frequently across 100 bootstrap splits
BEST_PARAMS_DEFAULT = {
    "n_estimators":     200,
    "max_depth":        None,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "bootstrap":        True,
}


@dataclass
class BootstrapResult:
    """Stores per-split evaluation results."""
    r2_test:    list = field(default_factory=list)
    rmse_test:  list = field(default_factory=list)
    mae_test:   list = field(default_factory=list)
    pcc_test:   list = field(default_factory=list)
    r2_train:   list = field(default_factory=list)
    y_true_all: list = field(default_factory=list)
    y_pred_all: list = field(default_factory=list)
    importances: Optional[pd.Series] = None
    best_params_counter: Counter = field(default_factory=Counter)


class BioactivityPredictionModel:
    """
    Random Forest regressor for formulation-level FRAP prediction.

    Parameters
    ----------
    composition_file : str
        Path to processing-adjusted chemical composition CSV.
        Index column: 'sample_id'. Other columns are chemical features.
    frap_file : str
        Path to FRAP measurement CSV. Must contain 'sample_id' and a
        FRAP value column (last column used by default).
    n_bootstrap : int
        Number of repeated random 80/20 train–test splits (default 100).
    test_size : float
        Fraction held out per split (default 0.2).
    tune_hyperparams : bool
        If True, run GridSearchCV on every split (slow).
        If False, use BEST_PARAMS_DEFAULT directly (fast, replicates paper).
    random_state : int
        Base random seed; each split uses random_state + i.
    log_transform : bool
        Apply log10 to FRAP before modeling (default True, as in paper).
    feature_min_prevalence : float
        Drop chemicals present in fewer than this fraction of samples.
    """

    def __init__(
        self,
        composition_file: str,
        frap_file: str,
        n_bootstrap: int = 100,
        test_size: float = 0.2,
        tune_hyperparams: bool = False,
        random_state: int = 42,
        log_transform: bool = True,
        feature_min_prevalence: float = 0.10,
    ):
        self.composition_file = composition_file
        self.frap_file = frap_file
        self.n_bootstrap = n_bootstrap
        self.test_size = test_size
        self.tune_hyperparams = tune_hyperparams
        self.random_state = random_state
        self.log_transform = log_transform
        self.feature_min_prevalence = feature_min_prevalence

        self.X_: Optional[pd.DataFrame] = None
        self.y_: Optional[pd.Series] = None
        self.feature_names_: Optional[list] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Loads and aligns composition features with FRAP targets."""
        conc_df = pd.read_csv(self.composition_file)
        y_df    = pd.read_csv(self.frap_file)

        conc_df.columns = conc_df.columns.str.strip()
        y_df.columns    = y_df.columns.str.strip()

        sample_ids = y_df["sample_id"].values
        X_df = (
            conc_df
            .set_index("sample_id")
            .reindex(sample_ids)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        # Drop low-prevalence features (paper: chemicals in < prevalence% of samples)
        presence = (X_df > 0).mean(axis=0)
        X_df = X_df.loc[:, presence >= self.feature_min_prevalence]

        # FRAP target
        frap_col = y_df.columns[-1]
        y = y_df[frap_col].astype(float)

        if self.log_transform:
            y = np.log10(y.clip(lower=1e-9))

        self.X_ = X_df
        self.y_ = y
        self.feature_names_ = list(X_df.columns)

        print(f"[BPM] Loaded {len(sample_ids)} samples, {X_df.shape[1]} features "
              f"(after prevalence filter ≥ {self.feature_min_prevalence:.0%})")

    # ------------------------------------------------------------------
    # Bootstrap evaluation loop
    # ------------------------------------------------------------------

    def run(self) -> BootstrapResult:
        """
        Runs the full 100-iteration bootstrap evaluation loop.

        Returns
        -------
        BootstrapResult
            Aggregated per-split metrics, pooled predictions, and
            mean feature importances.
        """
        if self.X_ is None:
            self.load_data()

        X, y = self.X_.values, self.y_.values
        result = BootstrapResult()
        all_importances = np.zeros(X.shape[1])

        print(f"[BPM] Running {self.n_bootstrap} bootstrap splits "
              f"({'with' if self.tune_hyperparams else 'without'} GridSearchCV) ...")

        for i in range(self.n_bootstrap):
            seed = self.random_state + i
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=self.test_size, random_state=seed
            )

            # Scale (preserves RF performance; needed if other models compared)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Hyperparameter selection
            if self.tune_hyperparams:
                gs = GridSearchCV(
                    RandomForestRegressor(random_state=seed),
                    PARAM_GRID,
                    cv=3,
                    n_jobs=-1,
                    scoring="r2",
                )
                gs.fit(X_tr_s, y_tr)
                params = gs.best_params_
            else:
                params = {**BEST_PARAMS_DEFAULT}

            result.best_params_counter[str(params)] += 1

            # Fit
            model = RandomForestRegressor(**params, random_state=seed)
            model.fit(X_tr_s, y_tr)

            # Predict
            y_pred_te = model.predict(X_te_s)
            y_pred_tr = model.predict(X_tr_s)

            # Metrics
            result.r2_test.append(r2_score(y_te, y_pred_te))
            result.rmse_test.append(np.sqrt(mean_squared_error(y_te, y_pred_te)))
            result.mae_test.append(mean_absolute_error(y_te, y_pred_te))
            result.pcc_test.append(pearsonr(y_te, y_pred_te)[0])
            result.r2_train.append(r2_score(y_tr, y_pred_tr))

            result.y_true_all.extend(y_te.tolist())
            result.y_pred_all.extend(y_pred_te.tolist())

            all_importances += model.feature_importances_

        # Average importances
        mean_imp = all_importances / self.n_bootstrap
        result.importances = pd.Series(
            mean_imp, index=self.feature_names_
        ).sort_values(ascending=False)

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Summary reporting
    # ------------------------------------------------------------------

    def _print_summary(self, result: BootstrapResult) -> None:
        """Prints mean ± SD for all evaluation metrics."""
        overfitting = np.array(result.r2_train) - np.array(result.r2_test)
        print("\n" + "=" * 55)
        print("  BPM — Bootstrap Evaluation Summary (n = {})".format(self.n_bootstrap))
        print("=" * 55)
        metrics = {
            "R²        ": result.r2_test,
            "RMSE      ": result.rmse_test,
            "MAE       ": result.mae_test,
            "PCC (r)   ": result.pcc_test,
            "Train R²  ": result.r2_train,
            "Overfit ΔR²": overfitting,
        }
        for name, vals in metrics.items():
            arr = np.array(vals)
            print(f"  {name}:  {arr.mean():.4f} ± {arr.std():.4f}")
        print("=" * 55)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_predicted_vs_actual(
        self,
        result: BootstrapResult,
        figsize: tuple = (6, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Scatter plot of pooled predicted vs. actual FRAP values across all
        100 held-out test sets (replicates Figure S3 in the paper).
        """
        y_true = np.array(result.y_true_all)
        y_pred = np.array(result.y_pred_all)

        r2   = r2_score(y_true, y_pred)
        pcc  = pearsonr(y_true, y_pred)[0]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="steelblue",
                   facecolors="lightsteelblue", s=40, linewidths=0.6, label="Predictions")

        lims = [min(y_true.min(), y_pred.min()) - 0.1,
                max(y_true.max(), y_pred.max()) + 0.1]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="1:1 line")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        label = f"R² = {r2:.3f}\nPCC = {pcc:.3f}\nRMSE = {rmse:.3f}"
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                verticalalignment="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

        title_suffix = "(log₁₀ FRAP)" if self.log_transform else "(FRAP)"
        ax.set_xlabel(f"Actual {title_suffix}", fontsize=12)
        ax.set_ylabel(f"Predicted {title_suffix}", fontsize=12)
        ax.set_title("BPM: Predicted vs. Actual FRAP\n"
                     f"Pooled across {self.n_bootstrap} bootstrap splits", fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_feature_importance(
        self,
        result: BootstrapResult,
        top_n: int = 20,
        figsize: tuple = (9, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Horizontal bar chart of top-N chemical features by mean Gini importance
        (averaged over all trees and bootstrap iterations).
        """
        top = result.importances.head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        colors = sns.color_palette("Blues_d", top_n)[::-1]
        ax.barh(top.index[::-1], top.values[::-1], color=colors, edgecolor="none")
        ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11)
        ax.set_ylabel("Chemical Feature", fontsize=11)
        ax.set_title(f"Top {top_n} Most Important Chemical Features\n"
                     f"(mean across {self.n_bootstrap} bootstrap trees)", fontsize=12)
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_metric_distributions(
        self,
        result: BootstrapResult,
        figsize: tuple = (12, 4),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Violin plots of R², RMSE, and PCC distributions across 100 bootstrap splits.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        data_map = {
            "R²": result.r2_test,
            "RMSE": result.rmse_test,
            "PCC (Pearson r)": result.pcc_test,
        }

        for ax, (name, vals) in zip(axes, data_map.items()):
            arr = np.array(vals)
            ax.violinplot(arr, positions=[0], showmedians=True)
            ax.scatter([0], [arr.mean()], color="red", zorder=5, s=40, label="Mean")
            ax.set_title(name, fontsize=12)
            ax.set_xticks([])
            ax.set_ylabel(name, fontsize=10)
            ax.text(0.5, 0.05, f"{arr.mean():.3f} ± {arr.std():.3f}",
                    ha="center", va="bottom", transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        fig.suptitle(f"Metric Distributions across {self.n_bootstrap} Bootstrap Splits",
                     fontsize=13, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BPM bootstrap evaluation")
    parser.add_argument("--comp",  required=True, help="Path to composition CSV")
    parser.add_argument("--frap",  required=True, help="Path to FRAP values CSV")
    parser.add_argument("--n",     type=int, default=100, help="Bootstrap iterations")
    parser.add_argument("--tune",  action="store_true", help="Run GridSearchCV per split")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--save",  default=None, help="Output directory for figures")
    args = parser.parse_args()

    bpm = BioactivityPredictionModel(
        composition_file=args.comp,
        frap_file=args.frap,
        n_bootstrap=args.n,
        tune_hyperparams=args.tune,
        random_state=args.seed,
    )
    results = bpm.run()

    save = args.save
    bpm.plot_predicted_vs_actual(results,  save_path=f"{save}/pred_vs_actual.png"  if save else None)
    bpm.plot_feature_importance(results,   save_path=f"{save}/feature_importance.png" if save else None)
    bpm.plot_metric_distributions(results, save_path=f"{save}/metric_distributions.png" if save else None)
