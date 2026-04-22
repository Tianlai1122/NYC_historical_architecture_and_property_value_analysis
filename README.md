# NYC Historical Architecture & Property Value Analysis

A Streamlit app that asks a single question:
> **Does a building's architectural heritage create measurable market value in Manhattan?**

Most real-estate pricing models stop at structural features (square footage, lot size, floors). We add **architect prestige, construction era, facade material, architectural style, landmark status, and historic-district membership**, then measure whether those features contribute to sale price on top of the baseline.

The short answer turned out to be more interesting than expected — the heritage premium exists but is largely already baked into the city's tax assessment. Full analytical write-up: [FINDINGS.md](FINDINGS.md).

---

## The App (6 Pages)

| # | Page | What's there |
|---|---|---|
| 1 | **Business Case & Data** | Research question, data merge strategy, dataset preview, summary KPIs. |
| 2 | **Visualizations & Maps** | 3D pydeck property map, era / style / material price comparisons, **Architect Value Leaderboard** with heritage premium ranking, **Era → Style → Price Tier Sankey**, correlation heatmap. |
| 3 | **Prediction Models** | 8 regression models trained side-by-side on Baseline vs Heritage feature sets. Live mini-leaderboard, Manhattan trivia carousel during training, **residual map** + top 10 over/under-priced properties. |
| 4 | **Feature Importance** | Per-model importance bar charts, full SHAP summary plot, individual SHAP waterfall for any test sample. Friendly column names throughout. |
| 5 | **Hyperparameter Tuning** | Grid search across configurable parameter ranges. Optional Weights & Biases integration with in-app API-key input. |
| 6 | **Property Valuator** *(extra)* | Pick any property, see Baseline vs Heritage predictions, 80% confidence interval via LightGBM quantile regression, SHAP waterfall, and a Wikipedia card pulling photos for famous buildings. |

---

## Models (8 total)

| Family | Models |
|---|---|
| Linear | Linear Regression, Ridge, Lasso, Elastic Net |
| Tree | Decision Tree, Random Forest |
| Gradient Boosting | sklearn Gradient Boosting, **LightGBM**, **CatBoost** |

Best test R² on log-price (full feature set, 80/20 split): **LightGBM 0.618 / CatBoost 0.616**.

---

## Data

Three NYC open datasets merged on **BBL (Borough-Block-Lot)**:

| Dataset | Provides |
|---|---|
| NYC Rolling Sales | Sale price, sale date, building class |
| MapPLUTO | Building dimensions, zoning, GPS, tax assessment, FAR |
| Landmarks Database | Architect, style, material, historic district, alteration year |

After merging and filtering for sales ≥ $100K: **2,864 properties × 62 columns**.

### Engineered features

| Feature | Notes |
|---|---|
| `building_age` | 2026 minus construction year |
| `construction_era` | Bucketed: Pre-1850, 1850–1899, 1900–1919, 1920–1939 (Art Deco), 1940–1969 (Mid-Century), 1970+ |
| `architect_prestige_score` | Frequency-weighted count of buildings per architect in the dataset |
| `architect_building_count` | Raw portfolio size |
| `rare_style_score` / `style_frequency` | Style rarity, both raw and inverse-frequency weighted |
| `years_since_alteration` | Recency of last renovation (sentinel for never-altered) |
| `is_landmark` / `in_historic_district` | Binary flags from the Landmark database |

Feature schema and prep logic live in `prepare_data.py` and the top of `app.py`.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>.

To rebuild the merged dataset from the three source CSVs (optional, source files not included in repo):
```bash
python3 prepare_data.py
```

### Weights & Biases (optional)

Page 5 supports W&B experiment tracking. Either:
- Run `wandb login` once in your terminal, **or**
- Paste your API key into the input field on the Hyperparameter Tuning page.

Project name on W&B: `Manhattan-heritage-property-analysis-app`.

---

## Tech Stack

- **App framework:** Streamlit (multi-page sidebar navigation, glassy custom CSS)
- **Visualization:** Plotly (interactive charts, Sankey, mapbox-free maps), pydeck (3D ColumnLayer), Matplotlib + Seaborn
- **ML:** scikit-learn, LightGBM (regular + quantile), CatBoost
- **Explainability:** SHAP (TreeExplainer + LinearExplainer, summary plot + waterfall)
- **Experiment tracking:** Weights & Biases (optional)
- **External data:** Wikipedia API (auto-fetches building photos and intros for the Property Valuator)

---

## Project Structure

```
├── app.py                              # Streamlit application (6 pages)
├── prepare_data.py                     # Data merge & feature engineering pipeline
├── Manhattan_Heritage_Analysis.csv     # Final merged dataset (2,864 rows × 62 cols)
├── requirements.txt                    # Python dependencies (incl. lightgbm, catboost)
├── README.md                           # This file
├── FINDINGS.md                         # Analytical insights & talking points
└── .gitignore                          # Excludes large source CSVs
```

---

## Themes

The sidebar includes three modern themes:
- **Porcelain** — Apple light mode (default), system white + SF blue
- **Graphite** — Apple dark mode, deep black + electric blue/purple gradient
- **Aurora** — Linear-app vibe, midnight navy + cyan/violet

---

## Course context

Final project for a Data Science course. The 5 graded pages (Business Case, Visualizations, Prediction with ≥ 5 models, Feature Importance, Hyperparameter Tuning with W&B) are all present; the Property Valuator (Page 6), Architect Leaderboard, Sankey, residual map, and live training UI are extras built on top of the base requirements.
