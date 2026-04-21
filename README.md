# 🏛️ Manhattan Heritage Property Valuation

**Financial Premium of Historic Preservation in Manhattan**  
A Streamlit app that quantifies how architectural character and historic preservation status affect NYC property prices.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the merged dataset (only needed once)
python3 prepare_data.py

# 3. Run the app
python3 -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📦 Project Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application (5 pages) |
| `prepare_data.py` | Data pipeline: merge & feature engineering |
| `Manhattan_Heritage_Analysis.csv` | Final merged dataset (2,864 rows × 62 cols) |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |

---

## 🏗️ Core Concept: Baseline vs Heritage Models

Traditional real-estate models use:
> Lot size, floors, FAR, zoning, assessment value

This app adds **architeural & preservation features**:
> Construction era · Architectural style · Facade material ·  
> Architect prestige score · Landmark status · Historic district membership ·  
> Alteration history · Rare style score

We compare both sets directly, showing the **Heritage Uplift** in R² across 6 models.

---

## 📄 App Pages

### 1️⃣ Business Case & Data
- Problem framing: baseline vs heritage predictor
- Five research questions
- Data source descriptions & merge strategy
- Data preview & column types

### 2️⃣ Map & Visualizations
- 🗺️ Interactive Plotly mapbox — color/size properties by any variable
- Construction era geography across Manhattan
- Price bubble map by neighborhood
- Architectural style premium charts
- Facade material & architect premiums
- Price by construction era (boxplot)
- Heritage feature correlation heatmap

### 3️⃣ Prediction Models
- 6 models: Linear · Ridge · Lasso · Decision Tree · Random Forest · Gradient Boosting
- **Dual training**: Baseline features only vs All features (baseline + heritage)
- Side-by-side R² comparison table
- Heritage Uplift bar chart
- Actual vs Predicted scatter

### 4️⃣ Feature Explainability (SHAP)
- Feature importance / coefficients colored by type (baseline vs heritage)
- Heritage contribution % of total model importance
- SHAP summary plot (global)
- SHAP waterfall plot (individual predictions)

### 5️⃣ Hyperparameter Tuning (W&B)
- Grid-search any model with configurable parameter ranges
- Real-time progress bar
- Results table with highlight of best config
- Tuning curve plots
- Optional Weights & Biases logging

---

## 🎨 Themes Available
| Theme | Style |
|-------|-------|
| 🌑 Midnight Purple | Dark background, purple gradients |
| 🪟 Glassmorphism | Translucent light panels |
| ⚡ Cyberpunk Neon | Black bg, neon red/cyan |
| 🔷 Royal Blue | Clean corporate navy |

---

## 📡 Weights & Biases Setup

```bash
pip install wandb
wandb login          # enter your API key from wandb.ai/settings
# Then in the app: check "Log to W&B" on page 5
```

---

## 🗂️ Data Sources

| Dataset | Source | Records (Manhattan) |
|---------|--------|---------------------|
| NYC Rolling Sales | NYC Open Data | 18,817 |
| MapPLUTO | NYC Dept of City Planning | 42,600 |
| Landmark & Historic District DB | NYC LPC | 14,610 |

**Merged on BBL** (Borough-Block-Lot) → **2,864 matched records**, 62 features.

---

## 📤 Upload to GitHub

```bash
cd "/Users/tianlaizhang/Downloads/DS Final"
git init
git add .
git commit -m "Initial commit: Manhattan Heritage Valuation App"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/manhattan-heritage.git
git push -u origin main
```
