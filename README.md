# Computational Framework for Cereal Extrusion Optimization

**A computational framework connecting formulation, processing, nutrient density, and antioxidant capacity: a cereal extrusion study**

Keer Ni¹²†, Xu Zhou¹²†, Pranav Gupta¹², & Ilias Tagkopoulos¹²

†These authors contributed equally to this work

¹Department of Computer Science & Genome Center, University of California, Davis
²USDA/NSF AI Institute for Next Generation Food Systems (AIFS)

---

## Overview

This repository provides the full computational framework described in the paper, including the data collection pipelines, mechanistic kinetic model, machine learning predictor, and extrusion condition optimizer.

The framework addresses three components:

1. **Data collection** — Two LLM-assisted automated pipelines collected (i) 12 nutritional values per ingredient from commercial supplier PDFs via an n8n workflow, and (ii) bioactive compound concentrations for 28 ingredients from ~2,000 peer-reviewed papers using a structured ChatGPT extraction protocol.

2. **Bioactivity Prediction Model (BPM)** — A Random Forest regressor trained on processing-adjusted chemical compositions to predict formulation-level antioxidant capacity (FRAP). Evaluated across 100 repeated 80/20 train–test splits; R² = 0.75, PCC = 0.87.

3. **Extrusion Condition Optimizer** — Differential Evolution and Dual Annealing search the (temperature, residence time) space to maximize a weighted combination of NRF9.3 and predicted FRAP, improving the combined score by a mean of 31% relative to unprocessed formulations.

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

## Data Collection Pipeline

Two parallel LLM-assisted extraction workflows were used to build the ingredient database.

### 1. Nutrient data (supplier PDFs → NRF9.3 inputs)

Nutritional composition data for 28 ingredients were collected from TraceGains ([tracegains.com](https://tracegains.com/)) supplier specification PDFs using an automated n8n workflow. For each PDF:

1. A schedule trigger (every 2 min) detected new PDFs in a designated Google Drive input folder.
2. Mistral AI extracted raw text from the PDF.
3. GPT-5.1 extracted 13 nutritional values and 7 product metadata fields using a structured schema and a nutrition-aware system prompt.
4. Extracted records were written to a Google Sheet; processed files were archived and renamed as `{ingredient_category} - {product_id}`.

The full importable n8n workflow is provided in `data-pipeline/n8n/`. Credential IDs and private Google Drive/Sheets IDs have been replaced with `YOUR_*` placeholders — substitute your own values before importing. The exact extraction schema and system prompt are documented in `data-pipeline/prompts/nutrient_extraction_prompts.md`.

### 2. Bioactive compound data (literature PDFs → antioxidant model inputs)

Phenolic acid and flavonoid concentrations were compiled from ~2,000 peer-reviewed papers identified through structured Google Scholar searches. Each paper was processed by uploading its PDF to a dedicated ChatGPT Project (GPT-5.2) with a table-by-table extraction protocol:

1. The model assessed each table for relevance (does it contain bioactive compound concentrations for a target ingredient?).
2. Relevant tables were extracted into flat JSON records preserving exact values, units, and abbreviation resolutions.
3. Extraction was constrained to the 28 target ingredients using a companion `ingredients.txt` scope file.
4. Total measurements (TPC, TFC) were included; nutrients and assay endpoints (FRAP, DPPH) were excluded.

The exact system prompt and per-paper user prompt are documented in `data-pipeline/prompts/bioactive_extraction_prompts.md`.

**Extraction quality:** A random 10% sample of source PDFs (~200 records) was manually audited. All audited records meeting completeness criteria achieved 100% agreement across extracted values. 17.4% of records were removed during review due to incomplete coverage (≥60% missing required fields), yielding a curated database of 503 records.

---

## Models

### 1. Bioactivity Prediction Model (BPM)

**Algorithm:** Random Forest Regressor (scikit-learn)

**Why Random Forest?** Selected for its ability to capture nonlinear relationships among compositional features without parametric assumptions, robustness to small sample sizes, and interpretable feature importance rankings. Evaluated against 9 alternative algorithms (Gradient Boosting, AdaBoost, XGBoost, SVR, LightGBM, ElasticNet, Lasso, Ridge, MLP); Random Forest achieved the highest normalized score across all six evaluation metrics.

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

### Expected input data for BPM

#### 1. Composition matrix CSV

A CSV with one row per `sample_id` (format: `FormulationName(Temperature,ResidenceTime)`) and one column per processing-adjusted chemical concentration feature.

| sample_id | ferulic_acid | catechin | vanillic_acid |
|---|---:|---:|---:|
| FormulationA(140,30) | 0.12 | 0.05 | 0.03 |
| FormulationA(160,45) | 0.10 | 0.04 | 0.02 |

#### 2. FRAP target CSV

A CSV with `sample_id` and the FRAP measurement column.

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
| Dual Annealing | Primary global solver; less sensitive to initialization |
| Grid search | Visualization and coarse benchmarking |
| Local solvers (L-BFGS-B, Nelder-Mead) | Diagnostic — sensitivity to initialization |

**Robustness:** Multiple random seeds; solutions verified by local refinement and neighborhood sensitivity analysis.

### Expected input data for Optimizer

#### 1. FRAP prediction surface CSV

Long-format CSV with columns for formulation identifier, temperature, residence time, and FRAP value. `sample_id` encodes temperature and time as `FormulationName(T,t)`.

#### 2. NRF surface CSV

Long-format CSV with columns for formulation identifier, temperature, residence time, and NRF9.3 score.

---

## Repository Structure

```
cereal-extrusion-framework/
├── README.md
├── requirements.txt
├── data-pipeline/
│   ├── n8n/
│   │   └── GoogleDrive-Mistral-GeminiChat-GoogleSheets_v3.json  # n8n workflow (credentials redacted)
│   └── prompts/
│       ├── nutrient_extraction_prompts.md   # GPT-5.1 system prompt + field schema (supplier PDFs)
│       └── bioactive_extraction_prompts.md  # GPT-5.2 prompts for literature bioactive extraction
└── src/
    └── models/
        ├── __init__.py
        ├── bioactivity_prediction.py   # BPM: Random Forest + bootstrap evaluation
        └── optimization.py             # Global optimization of extrusion conditions
```

---

## Installation

```bash
git clone https://github.com/tagkopouloslab/food-processing-extrusion-modeling.git
cd food-processing-extrusion-modeling
pip install -r requirements.txt
```

---

## Usage

### Bioactivity Prediction

```python
from src.models.bioactivity_prediction import BioactivityPredictionModel

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

Or from the command line:

```bash
python -m src.models.bioactivity_prediction \
    --comp data/x_top_fp_chem_food_atDifferentTemp_df.csv \
    --frap data/fp_LIT_frap_Values_matches.csv \
    --n 100 --save figures/
```

### Optimization

```python
from src.models.optimization import ExtrusionOptimizer

optimizer = ExtrusionOptimizer(
    bioactivity_file="data/fp_formulation_bioactivity_values_100_200C_0_120s.csv",
    nrf_file="data/fp_formulation_scores_weighted_opt_NRF_100_200C_0_120s.csv",
    alpha=0.5
)

# Optimize a specific formulation
result = optimizer.optimize(formulation="FP_Formulation_1", method="differential_evolution")
print(result)

# Compare all solvers for one formulation
optimizer.compare_solvers(formulation="FP_Formulation_1")

# Optimize all formulations and get summary table
summary = optimizer.optimize_all(method="differential_evolution")

# Visualize the objective landscape
optimizer.plot_landscape(formulation="FP_Formulation_1")
```

Or from the command line:

```bash
python -m src.models.optimization \
    --bio data/fp_formulation_bioactivity_values_100_200C_0_120s.csv \
    --nrf data/fp_formulation_scores_weighted_opt_NRF_100_200C_0_120s.csv \
    --form all --method differential_evolution --save figures/
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{ni2025cereal,
  title={A computational framework connecting formulation, processing, nutrient density, and antioxidant capacity: a cereal extrusion study},
  author={Ni, Keer and Zhou, Xu and Gupta, Pranav and Tagkopoulos, Ilias},
  journal={npj Science of Food},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work was supported by the USDA/NSF AI Institute for Next Generation Food Systems (AIFS) and the University of California, Davis.
