# Computational Framework for Cereal Extrusion Optimization

**A computational framework connecting formulation, processing, nutrient density, and antioxidant capacity: a cereal extrusion study**

Pranav Gupta¹², Keer Ni¹², Xu Zhou¹², & Ilias Tagkopoulos¹²

¹Department of Computer Science & Genome Center, University of California, Davis  
²USDA/NSF AI Institute for Next Generation Food Systems (AIFS)

---

## Overview

This repository provides the computational framework described in the paper. The pipeline integrates mechanistic kinetic models of thermal processing with supervised machine learning to predict formulation-level antioxidant capacity, and uses global optimization to identify extrusion conditions that maximize both nutrient density and antioxidant activity.

---

## Scientific Background

### Problem

Cereal extrusion is a high-temperature, short-time process that simultaneously modifies nutrient content and bioactive compound profiles. No prior computational framework jointly models the effects of formulation composition, extrusion temperature, and residence time on both nutrient density (NRF9.3) and antioxidant capacity (FRAP, DPPH, ABTS, ORAC).

### Approach

1. **Kinetic models** propagate initial chemical composition through thermal degradation and phenolic release reactions under defined extrusion conditions (temperature: 100–200 °C; residence time: 0–120 s), yielding a processing-adjusted feature vector per formulation–condition pair.

2. **Bioactivity Prediction Model (BPM)** — a Random Forest regressor trained on the processing-adjusted chemical concentrations to predict FRAP (primary endpoint). DPPH, ABTS, and ORAC serve as complementary endpoints.

3. **Optimization** — given a fixed formulation, Differential Evolution and Dual Annealing search the (temperature, residence time) space to maximize a weighted combination of NRF9.3 and predicted FRAP.

### Dataset: Bioactivity Food Link (BFL)

The BFL dataset links food chemical profiles to in vitro antioxidant measurements:

- **n = 275** food samples with one or more of: FRAP, DPPH, ABTS, ORAC
- **189 foods** from [FoodAtlas](https://www.foodatlas.ai/) + 5 additional ingredients from literature
- FRAP primary target: standardized to mmol Fe²⁺ equivalents per 100 g (dry basis), log-transformed
- Stratification by unique extrusion conditions yields **30 extrusion-condition-specific samples** for supervised modeling

---

## Models

### 1. Bioactivity Prediction Model (BPM)

**Algorithm:** Random Forest Regressor (scikit-learn)

**Why Random Forest?** Selected for its ability to capture nonlinear relationships and interactions among compositional features without parametric assumptions, robustness to small sample sizes, tolerance of correlated features, and interpretable feature importance rankings. Evaluated against 9 alternative algorithms (Gradient Boosting, AdaBoost, XGBoost, SVR, LightGBM, ElasticNet, Lasso, Ridge, MLP); Random Forest achieved the highest normalized score across all six evaluation metrics.

**Input features:** Processing-adjusted chemical concentrations per formulation (post-extrusion, aggregated across all ingredients)

**Target variable:** Log-transformed FRAP (mmol Fe²⁺ eq./100 g)

**Evaluation protocol:**
- 100 iterations of repeated random 80/20 train–test splits
- Splitting at the formulation level (no condition-level leakage)
- Hyperparameters tuned via 3-fold cross-validated grid search within each training split

**Hyperparameter grid:**

| Parameter | Values searched |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | None, 10, 20, 30 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `bootstrap` | True, False |

**Best configuration (most frequent across 100 splits):**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | None |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 1 |
| `bootstrap` | True |

**Performance (100 held-out test sets):**

| Metric | Value |
|---|---|
| R² | 0.75 |
| PCC | 0.87 |

**Feature importance:** Mean decrease in impurity (Gini importance) averaged across all trees. Features with consistently low importance across bootstrap runs are excluded in a single refinement pass.

### Expected input data for Bioactivity model inputs

#### 1. Concentration matrix CSV
A CSV with:

- one row per `sample_id`
- one column named `sample_id`
- remaining columns = processing-adjusted chemical concentration features

Example:

| sample_id | ferulic_acid | catechin | vanillic_acid |
|---|---:|---:|---:|
| FormulationA(140,30) | 0.12 | 0.05 | 0.03 |
| FormulationA(160,45) | 0.10 | 0.04 | 0.02 |

#### 2. Target CSV
A CSV with:

- one column named `sample_id`
- one FRAP target column

Example:

| sample_id | FRAP |
|---|---:|
| FormulationA(140,30) | 1.78 |
| FormulationA(160,45) | 1.95 |
---

### 2. Extrusion Condition Optimizer

**Decision variables:** Extrusion temperature *T* ∈ [100, 200] °C and mean residence time *t* ∈ [0, 120] s

**Objective:**

$$\max_{T, t} \; f(T, t) = \alpha \cdot \hat{N}(T,t) + (1-\alpha) \cdot \hat{F}(T,t)$$

where $\hat{N}$ and $\hat{F}$ are min–max normalized NRF9.3 and predicted FRAP over the feasible search space, and $\alpha = 0.5$ (equal weighting, as used in the paper).

**Solvers:**

| Solver | Role |
|---|---|
| Differential Evolution | Primary global solver; gradient-free, population-based |
| Dual Annealing (Simulated Annealing) | Primary global solver; less sensitive to initialization |
| Grid search | Visualization and coarse benchmarking |
| Local solvers (L-BFGS-B, Nelder-Mead) | Diagnostic — sensitivity to initialization |

**Robustness:** Multiple random seeds; solutions verified by local refinement and neighborhood sensitivity analysis.

### Expected input data for Optimization inputs

Two long-format CSVs are expected:

#### 1. FRAP prediction surface CSV
Must contain columns representing:

- formulation identifier
- temperature
- residence time
- FRAP value

#### 2. NRF surface CSV
Must contain columns representing:

- formulation identifier
- temperature
- residence time
- NRF value


---

## Repository Structure

```
cereal-extrusion-framework/
├── README.md
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── bioactivity_prediction.py   # BPM: Random Forest + bootstrap evaluation
    └── optimization.py             # Global optimization of extrusion conditions

```

---

## Installation

```bash
git clone https://github.com/tagkopouloslab/food-processing-extrusion-modeling.git
cd Food-Processing-Extrusion-Modeling
pip install -r requirements.txt
```

---

## Usage

### Bioactivity Prediction

```python
from models.bioactivity_prediction import BioactivityPredictionModel

bpm = BioactivityPredictionModel(
    composition_file="data/x_top_fp_chem_food_atDifferentTemp_df.csv",
    frap_file="data/fp_LIT_frap_Values_matches.csv",
    n_bootstrap=100,
    test_size=0.2,
    random_state=42
)

results = bpm.run()
bpm.plot_predicted_vs_actual(results)
bpm.plot_feature_importance(results, top_n=20)
```

### Optimization

```python
from models.optimization import ExtrusionOptimizer

optimizer = ExtrusionOptimizer(
    bioactivity_file="data/fp_formulation_bioactivity_values_100_200C_0_120s.csv",
    nrf_file="data/fp_formulation_scores_weighted_opt_NRF_100_200C_0_120s.csv",
    alpha=0.5
)

# Optimize a specific formulation
result = optimizer.optimize(formulation="FP_Formulation_1", method="differential_evolution")
print(result)

# Compare all solvers
optimizer.compare_solvers(formulation="FP_Formulation_1")

# Visualize the objective landscape
optimizer.plot_landscape(formulation="FP_Formulation_1")
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{gupta2025cereal,
  title={A computational framework connecting formulation, processing, nutrient density, and antioxidant capacity: a cereal extrusion study},
  author={Gupta, Pranav and Ni, Keer and Zhou, Xu and Tagkopoulos, Ilias},
  journal={},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work was supported by the USDA/NSF AI Institute for Next Generation Food Systems (AIFS) and the University of California, Davis.
