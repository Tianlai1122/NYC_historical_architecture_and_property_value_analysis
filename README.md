# NYC Historical Architecture and Property Value Analysis

## What This Project Does

This is a Streamlit web application that answers one question: **does a building's architectural heritage affect its market price in Manhattan?**

Most real estate pricing models only use structural features like square footage, lot size, and number of floors. We go further by adding architectural and preservation characteristics — construction era, architect prestige, facade material, architectural style, historic district status — and measuring whether those features create a detectable "preservation premium."

## How It Works

We merged three NYC open datasets into one unified dataset of **2,864 historic properties** in Manhattan:

| Dataset | What It Provides |
|---------|-----------------|
| NYC Rolling Sales | Sale prices, transaction dates |
| MapPLUTO | Building dimensions, zoning, GPS coordinates, tax assessments |
| Landmark Database | Architect, architectural style, facade material, historic district, alteration history |

All three datasets are joined on **BBL (Borough-Block-Lot)**, NYC's unique property identifier.

## The App (5 Pages)

**Page 1 — Business Case & Data**
Overview of the research question, data sources, merge strategy, and a preview of the final dataset.

**Page 2 — Visualizations & Maps**
Interactive Plotly maps showing property locations colored by price, building age, or architect prestige. Charts comparing price across architectural styles, facade materials, construction eras, and neighborhoods.

**Page 3 — Prediction Models**
Six regression models (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting) trained on two feature sets side by side:
- **Baseline:** structural features only (sqft, floors, lot size, assessments)
- **Heritage-Enhanced:** structural + architectural features (style, era, architect prestige, landmark status)

The page shows which models improve when heritage features are added.

**Page 4 — Feature Importance (SHAP)**
SHAP explainability analysis showing which features drive predictions. Features are color-coded as Baseline vs Heritage so you can see how much architectural character contributes to the model.

**Page 5 — Hyperparameter Tuning**
Grid search with configurable parameter ranges. Optional integration with Weights & Biases for experiment tracking.

## Quick Start

```bash
pip install -r requirements.txt
python3 -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

To rebuild the dataset from the three source files (optional):
```bash
python3 prepare_data.py
```

## Tech Stack

- **App:** Streamlit
- **Visualization:** Plotly (interactive maps & charts), Matplotlib, Seaborn
- **ML:** scikit-learn (6 regression models)
- **Explainability:** SHAP
- **Experiment Tracking:** Weights & Biases (optional)
- **Data:** pandas, numpy

## Project Structure

```
├── app.py                              # Streamlit application (5 pages)
├── prepare_data.py                     # Data merge & feature engineering pipeline
├── Manhattan_Heritage_Analysis.csv     # Final merged dataset (2,864 rows × 62 cols)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Excludes large source CSVs
└── README.md                           # This file
```

## Key Engineered Features

| Feature | Description |
|---------|-------------|
| `building_age` | 2026 minus construction year |
| `construction_era` | Binned into Pre-1850, 1850–1899, 1900–1919, Art Deco, Mid-Century, 1970+ |
| `architect_prestige_score` | Based on how many buildings in the dataset share the same architect |
| `rare_style_score` | Inverse frequency of architectural style (rarer style = higher score) |
| `preservation_level` | Combined landmark + historic district + alteration status |
| `is_landmark` / `in_historic_district` | Binary flags from the Landmark database |

## Theme Options

The sidebar includes a theme switcher with three visual styles: Classic Dark, Light Minimal, and Warm Tone.
