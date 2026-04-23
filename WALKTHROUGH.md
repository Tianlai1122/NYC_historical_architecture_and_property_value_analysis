# Project Walkthrough — Manhattan Heritage Valuation

A page-by-page, chart-by-chart guide to what's in the app, what to look at on each visualization, and what the numbers mean. Written so you can read this once and present the project end-to-end without re-reading the code.

If you only have 90 seconds, jump to the **TL;DR** at the bottom.

---

## Project in one sentence

> We built a Streamlit app that asks **"Does Manhattan property heritage actually move sale price?"** by training nine regression models on a custom-merged 2,864-row dataset of NYC sales × MapPLUTO × Landmarks, comparing a structural-only baseline against a heritage-enhanced version, and surfacing the result through interactive maps, SHAP explanations, hyperparameter tuning, and a per-property valuator.

---

## The dataset

We merge **three NYC open datasets** on `BBL` (Borough-Block-Lot, NYC's universal property key):

| Source | What it gives | Records |
|---|---|---|
| NYC Rolling Sales (2025) | sale price, sale date, building class | 18,817 |
| MapPLUTO | sqft, floors, GPS, tax assessment, zoning, FAR caps | 42,600 |
| LPC Landmarks DB | architect, style, material, historic district, alteration year | 14,610 |

After dropping non-market transactions (sales < $100K = donations, internal transfers, etc.) we end up with **2,864 properties × 62 columns**. Every row in the merged set is a Manhattan property that was both sold in 2025 and is in the Landmark Database.

Engineered features (in `prepare_data.py`):
- `building_age` (2026 − YearBuilt)
- `construction_era` bucketed into 6 tiers (Pre-1850 / 1850–1899 / 1900–1919 / 1920–1939 Art Deco / 1940–1969 Mid-Century / 1970+)
- `architect_prestige_score` and `architect_building_count` (frequency-weighted; portfolio size)
- `rare_style_score` and `style_frequency` (style rarity, raw + inverse weighted)
- `years_since_alteration` with sentinel for never-altered
- `is_landmark`, `in_historic_district` binary flags

---

## Page 1 — Business Case & Data

### What's there
- The research question stated explicitly with the predictive-vs-causal limitation called out (we do not claim landmark status *causes* premium, only that it *predicts* price).
- Two side-by-side panels listing what's in the **Baseline** model (size, lot, zoning, assessment, FAR family) versus what's added in the **Heritage-Enhanced** model (era, age, architect, style, material, landmark flags, alteration history).
- A "Why this matters" block.
- An info banner restating the predictive-not-causal limitation.
- Three data-source tiles with record counts and a description of the BBL merge key.
- 4 KPI cards: Final merged records · Engineered features · Unique architects · With GPS coordinates.
- A 20-row data preview.
- An expander with column types and missing-value percentages for all 62 columns.

### What to highlight in a presentation
- We're upfront that this is **prediction, not causal inference** — that's a smart, defensive framing for a data science course.
- The data is custom-built. None of the three source datasets has every field we need; the BBL join is what makes the project possible.
- The expander shows that some heritage fields are sparse (`landmark_orig`, `style_secondary`) while structural ones are dense — useful to acknowledge before the model section.

---

## Page 2 — Visualizations & Maps

### Section A — 3D Property Map (pydeck)

**What it shows.** Every property as a 3D column. User picks two encodings:
- **Color** = sale_price / price_per_sqft / building_age / architect_prestige / num_floors / assess_total
- **Height** = sale_price / sqft / floors / age / price_per_sqft

A custom HTML legend below the map shows the actual color gradient with min/max numeric labels and an "↕ Height = …" annotation.

**Observation points.**
- Default (color = sale price, height = sqft) makes Midtown East and the Upper East Side glow as the densest cluster of high-priced trophy buildings. SoHo lights up for `architect_prestige_score`.
- Switching height to `building_age` reveals a different geography: Greenwich Village and the Lower East Side become tall, Midtown is shorter — the historic core of the city is older.
- Hovering any column shows a tooltip with the building name, formatted price, style, architect, era, and sqft.

**Why it matters.** This is the "wow" view that replaces the static 2-D map most class projects use. The pydeck ColumnLayer is GPU-rendered and stays smooth at 2,800+ points.

### Section B — Construction Era Map

**What it shows.** A flat scatter map colored by the 6 construction eras. Each era has its own color (`Pre-1850` red → `1970+` light cyan).

**Observation points.**
- Pre-1850 buildings cluster downtown (Financial District, South Street Seaport).
- Art Deco (1920–1939) is concentrated in Midtown — that's the Empire State / Chrysler / Rockefeller era.
- Post-war buildings push north into the Upper East and Upper West Sides.

This is the geographic story of how Manhattan was built outward in time.

### Section C — Price Analysis (era / style / material panels)

**Observation points.**
- **Median price by era** is U-shaped: very old (pre-1850 surviving stock) and post-war (1970+ luxury) sit at the top, with the middle eras lower. Suggests scarcity of pre-Civil-War buildings drives one premium and modern construction quality drives another.
- **Top architectural styles by median sqft price** put Beaux-Arts, Greek Revival, and Italian Renaissance Revival near the top. Validates the qualitative story that early 20th-century European-influenced styles command premium today.
- **Material panel** (limestone vs brick vs brownstone) shows limestone leads — limestone facades are typical of Beaux-Arts and Italianate buildings, so this is consistent.

### Section D — Architect Value Leaderboard

**What it shows.** A ranked table of architects whose portfolios sell above the city's median price-per-sqft. Filtered to architects with **≥ 5 buildings** and sales **≥ $100K, ≥ $50/sqft** to remove non-market noise.

**Observation points.**
- The top of the list surfaces firms like McKim, Mead & White and C. P. H. Gilbert — exactly the names you'd expect from architectural-history class.
- The leaderboard also shows the "Heritage Premium" column (the architect's median price/sqft minus the city baseline) which is the cleanest single-number expression of brand value.

### Section E — Era → Style → Price Tier Sankey

**What it shows.** A three-stage Sankey: Era nodes flow into Style nodes flow into Price-Tier nodes (Low / Mid / High / Premium quartiles).

**Observation points.**
- Beaux-Arts flows almost entirely from 1900–1919 → Premium tier. That's a clean signal.
- Mid-Century Modern (1940–1969) splits across Mid and High tiers — those buildings are not premium-priced.
- Post-1970 Contemporary feeds into all four tiers, showing modern construction has a wider price spread.

### Section F — Monthly Sale Count + Median Price (era small multiples)

**What it shows.** Six mini-charts (3×2 grid), one per construction era, with monthly bars for transaction count and an overlaid line for median sale price on a secondary y-axis.

**Observation points.**
- Volume in older eras is steady but thin; volume in modern eras is higher with more month-to-month variance.
- The price line is wobbly within a single year (only 2025 of data) but useful for showing the "thin tape" caveat — there is *not* enough data to talk about price appreciation per architect.

### Section G — Correlation Heatmap

**What it shows.** Pearson correlations among numeric features.

**Observation points.**
- `assess_total` is the single strongest correlate of `sale_price` (≈ 0.7). This is why heritage features struggle to add R² uplift in the modeling section — the city's tax assessment has already absorbed heritage.
- Within heritage features, `architect_prestige_score` correlates with `is_landmark` and `in_historic_district`, so they encode overlapping signal.

---

## Page 3 — Prediction Models

This is the page that directly answers the research question.

### What it does
- Trains the selected models (default: all 9) twice each — once on the **21-feature Baseline** and once on the **33-feature Heritage-Enhanced** set — with an 80/20 split (configurable).
- During training, shows a real-time progress bar, a status line ("Training 4/9: Random Forest"), a Manhattan trivia carousel that rotates per model, and a **live leaderboard bar chart** that re-ranks after every model finishes.
- When complete, fires `st.balloons()` and reports total elapsed time + slowest model.

### Results table (the headline)
Two side-by-side dataframes (Baseline vs Heritage-Enhanced) with friendly column headers:
- **R² (Train)**, **R² (Test)** — best test value highlighted blue
- **MAE ($)**, **RMSE ($)** — lowest highlighted green; reported in dollars (after `expm1` of the log-space prediction) so they're interpretable directly
- **Time (s)** — per-model wall-clock

A caption explains why linear models report inflated dollar errors (small log error × large nominal price = big dollar miss after exponentiation).

### Heritage Feature Uplift chart
Grouped bar chart: Baseline R² vs Heritage R² per model. The **delta is the answer** to the research question.

### Observation points
- **Best test R² ≈ 0.61–0.62** for LightGBM and CatBoost on the heritage set — virtually tied. Random Forest and Gradient Boosting trail at ~0.58–0.60.
- Linear models cap at R² ≈ 0.19. They cannot capture the non-linear interactions in this data.
- The **heritage uplift is small (~1–2%)** on the expanded baseline. This is the "vanishing premium" finding — see **FINDINGS.md §1**. The premium isn't gone; it's hidden inside `assess_total`.
- The dollar MAE for the best models is around **$2.7M–$3.0M** — sounds large until you remember Manhattan prices span 4 orders of magnitude (from $100K studios to $200M trophy assets). A log-RMSE of ~0.6 means typical predictions are within ~1.8× of the truth.

### Bonus: Residual Map + Top 10 Over/Under-Priced
After training, the page renders:
- A **residual map** (red = sold for more than predicted, green = sold for less) using a diverging colorscale with a clearly labeled colorbar.
- A **Top 10 over-priced** and **Top 10 under-priced** dataframe with building name, address, predicted vs actual, dollar gap, neighborhood. This is the page's most "tell-a-story" moment because it identifies specific properties the model thinks were mispriced.

---

## Page 4 — Feature Importance & Explainability

Requires that Page 3 trained at least one model (cached in session state).

### Feature Importance section
- Horizontal bar chart of importance values, colored by Baseline (steelblue) vs Heritage (theme accent) so you can immediately see how much of the model's intelligence is coming from heritage variables.
- An info banner reports **"Heritage features account for X% of total importance"** — typically ~25–35% for tree models.
- For linear models, also shows a **Coefficient Direction** chart with a vertical line at 0.

### SHAP section
- **SHAP summary plot** (matplotlib) on a 250-sample subset of the test set. This is the rich visualization showing both feature importance and direction (which feature values push price up vs down).
- **Individual prediction breakdown**: pick any test sample with a slider, see the predicted price in dollars, and a SHAP waterfall plot showing exactly which features added or subtracted from the base prediction.

### Observation points
- For tree models, the top SHAP features are usually `assess_total`, `gross_sqft`, `architect_prestige_score`, and one of the FAR features. Heritage features (era, style, age) appear in the middle of the bar.
- The waterfall plot is the "CEO mode" visualization — even non-technical viewers can follow which features lifted or pushed down the prediction for a specific building.

---

## Page 5 — Hyperparameter Tuning

### Layout
- Top expander: "What is hyperparameter tuning?" — beginner intro.
- Model selector with a tooltip explaining what each family tunes.
- Test-size slider with tooltip.
- For each model, a **per-model tip caption** ("Tip: Alpha controls regularization strength…") so newcomers understand what knobs to turn without leaving the page.
- The grid of multiselects expands based on the chosen model.
- Total-experiments counter.
- Optional W&B section with API key input, project/entity fields, and a checkbox tooltip.

### Run experience
- Live progress bar updates per experiment.
- Each run optionally logs to Weights & Biases under project `Manhattan-heritage-property-analysis-app`.
- W&B errors are surfaced as warnings (first 2 runs), and fail-safe (training continues even if W&B is down).

### Results
- Sortable dataframe with friendly headers (R² (Train), R² (Test), MAE ($), RMSE ($)) and a one-line caption.
- Best test R² cell highlighted blue, lowest dollar errors highlighted green.
- "Best:" success banner with the winning hyperparameters.
- A line or bar chart of R² vs the first hyperparameter dimension to spot trends.

### Observation points
- Random Forest peaks around max_depth=10, n_estimators=200 — past that you're spending compute for no R² gain.
- LightGBM is best with smaller learning_rate (0.05) and more estimators (400–800), classic boosting tradeoff.
- Lasso/Elastic Net performance is flat across alpha values — the linear ceiling is the bottleneck, not the regularization.

---

## Page 6 — Property Valuator (extra, beyond the rubric)

### What it does
Pick any property from a search box (autocomplete on building name + address). The page then:

1. Shows the property's photo, address, and metadata.
2. Predicts the sale price using **Baseline** and **Heritage-Enhanced** trained models side by side.
3. Shows an **80% prediction interval** (10th and 90th percentile) computed from LightGBM quantile regression — much more honest than a single point prediction.
4. Compares the interval to the actual sale price (if known) and labels it as "inside the band", "above P90 by $X", or "below P10 by $X".
5. Renders a SHAP waterfall plot for that specific property's heritage prediction.
6. Auto-fetches a **Wikipedia card** for famous buildings (e.g. Flatiron, Plaza Hotel) — pulls the page intro and the lead photo via the Wikipedia REST API.

### Observation points / talking points
- The 80% interval typically spans roughly 1.5×–2× wide (e.g. $2.4M–$4.6M for a $3.2M expected price). That width *is* the honest uncertainty.
- For famous buildings, the Wikipedia card gives the demo a "wow this knows what it's looking at" moment.
- The Baseline-vs-Heritage delta on a single property is often more striking than the aggregate test-set delta — for an architect-rich property the heritage prediction can be 15–25% higher.

---

## Theme system

Three Apple-quality themes selectable from the sidebar:
- **Porcelain** — Apple light mode. Pure white background, SF-blue accent. Default.
- **Graphite** — Apple dark mode. Deep black, electric blue → purple gradient.
- **Aurora** — Linear-app vibe. Midnight navy, cyan → violet.

The theme drives Plotly template, matplotlib style, pydeck basemap (carto-positron / carto-darkmatter), Streamlit CSS, KPI card glass effect, button styling, and the gradient page title. All in one dictionary, one CSS injection.

---

## What we measured (numerical results recap)

| Model | Heritage R² | Heritage MAE ($) | Notes |
|---|---|---|---|
| LightGBM | **0.610** | ≈ $2.84M | Best-or-tied. ~1s training. |
| CatBoost | 0.592 | ≈ $2.92M | Tied at the top, no scaling needed. |
| Random Forest | 0.588 | ≈ $2.78M | Robust; smallest dollar MAE. |
| Gradient Boosting | 0.569 | ≈ $2.99M | Solid but slower than LightGBM. |
| Decision Tree | 0.450 | ≈ $3.14M | Single tree under-fits. |
| Linear / Ridge / Lasso / ElasticNet | 0.19–0.20 | $11M–$18M | Linear ceiling is the bottleneck. |

(MAE values are in raw dollars after `expm1`. Linear models look terrible in dollars because their log-residuals exponentiate badly.)

**Heritage uplift on test R² ≈ 1–2%** with the expanded baseline; ~5–8% with a thinner baseline. The full story is in **FINDINGS.md §1**.

---

## Project achievements (what to be proud of)

1. **Built a real custom dataset.** Three NYC datasets merged on BBL, not a Kaggle download. Anyone can rebuild from `prepare_data.py`.
2. **9 models, not just the rubric's 5.** Linear / Ridge / Lasso / ElasticNet / Decision Tree / Random Forest / Gradient Boosting / LightGBM / CatBoost — covers linear, single-tree, bagging, and two boosting families.
3. **Honest baseline.** We kept `assess_total` in the baseline even though it weakens the heritage uplift. We chose to surface the econometric insight ("the city has already priced heritage in") rather than chase a flattering uplift number.
4. **MAE in dollars, not log-units.** Every metric users see is in real money. A model report that says "$2.8M MAE" is interpretable; one that says "0.49 log-RMSE" is not.
5. **Live training UX.** Progress bar + per-model status + Manhattan trivia carousel + live leaderboard re-ranking after each model. No silent spinner.
6. **Friendly column names everywhere.** `gross_sqft` shows up as "Gross Floor Area (gross_sqft)" in dropdowns and chart labels. Reduces cognitive load for non-technical viewers.
7. **Per-model beginner tips on the tuning page** + tooltips on every multiselect — newcomers can learn what each knob does without opening a textbook.
8. **3D pydeck map with a real legend.** Most class projects show a flat scatter map. Ours has elevation, color, picked basemap per theme, custom HTML tooltips, and a colorbar with min/max numeric labels.
9. **80% prediction intervals** via LightGBM quantile regression on Page 6. Much more defensible than a single point estimate.
10. **Wikipedia integration.** The Property Valuator pulls real building photos and intros for famous landmarks. Free, cached, fail-safe.
11. **W&B integration done right.** Optional, in-app API key input, errors surface as warnings, training continues if W&B is down.
12. **SHAP both globally (summary) and locally (waterfall).** Tree explainer for tree models, linear explainer for linear models, automatic dispatch.

---

## Things to mention as honest limitations

- **Single year of sales data.** Can't model price *appreciation* per architect — only price *levels*.
- **`is_landmark` is constant in the merged set** because the merge filter intersects with the Landmark DB (every row is already a landmark). Documented in FINDINGS.md §5; we keep it in the schema for clarity.
- **No interior data.** Bathroom count, ceiling height, view tier, renovation cost would close most of the remaining R² gap. These are private to Zillow / Compass / StreetEasy.
- **The trophy-asset ceiling.** R² actually drops on the highest-prestige subset because trophy buildings are bought on idiosyncratic factors (taste, scarcity, story) that no structural data can capture (FINDINGS.md §2).

---

## TL;DR

> We answer "does heritage matter?" by training nine regression models on a custom-merged 2,864-property Manhattan dataset, comparing structural-only vs heritage-enhanced versions. Best models (LightGBM, CatBoost) reach R² ≈ 0.61. The headline finding is that the heritage premium is real but mostly already absorbed by the city's tax assessment, which is itself a downstream function of landmark status. The app surfaces this through 6 pages: a research-question landing page, an interactive 3D map + Sankey + architect leaderboard, a live-training model comparison, SHAP-based feature importance, hyperparameter tuning with W&B, and a per-property valuator with 80% prediction intervals and Wikipedia auto-lookup. The whole thing runs in 3 seconds end-to-end and is themed in three modern looks.
