# Findings & Analytical Notes

Working notes from analyzing the **Manhattan Heritage Property Valuation** dataset (2,864 historic landmarked properties merged from NYC Rolling Sales + MapPLUTO + Landmarks Database). These findings emerged from exploratory work and shape the story we tell on Page 1 / Page 3 / the presentation.

---

## 1. The "Vanishing Heritage Premium"

### What we observed
- With a **minimal baseline** (12 structural features: sqft, floors, lot dimensions, assessment values), adding heritage features lifted log-price R² by **~5–8%**.
- After expanding the baseline to **21 features** (adding tax exemptions, unit mix, zoning class, building class, sale month, FAR family), the heritage uplift collapsed to **~1–2%**.

### Why it matters
This is **not** a bug or a sign that heritage doesn't matter. It's an econometric insight:
> The city's tax assessment system has already priced heritage in. Landmark designation, historic-district membership, and architectural prestige bleed into `assess_total` because the assessors consider them. Once the model has access to assessment value, the *marginal* contribution of explicit heritage features shrinks because the same information is already there, just laundered through tax data.

### Talking point
> *"We don't see a large heritage R² uplift because the city has already captured it implicitly through tax assessment. The premium is real — it's just being absorbed by `assess_total`, which itself is a downstream function of landmark status."*

---

## 2. The Trophy-Asset Ceiling

### What we tested
We sliced the dataset by architect prestige to see if "famous-architect properties" predict more accurately than the average:

| Subset | n | R² (log-price) | MAE (USD) |
|---|---|---|---|
| Full dataset | 2,834 | **0.57** | $2.99 M |
| Known architect | 2,834 | 0.57 | $2.99 M |
| Top-50% prestige | 1,425 | 0.49 | $2.32 M |
| Architect portfolio ≥ 50 buildings | 1,235 | 0.51 | $2.65 M |

### Why it matters
- The full dataset is *already* 100% architect-known (every row in the Landmarks DB has a credited architect), so "having an architect" is not a discriminating feature.
- Subsetting to **higher-prestige** architects actually **lowers R²**. Trophy assets are bought on idiosyncratic factors (taste, scarcity, market sentiment, view, story) that structural data cannot capture.
- MAE drops in absolute dollars on the high-prestige subset because the price distribution is tighter at the high end — but explanatory power decreases.

### Talking point
> *"Structural models work best in the middle market. They hit a ceiling on trophy luxury, where price is a function of buyer psychology more than building specs."*

---

## 3. Best Models on This Dataset

After expanding features, comparing all 8 models (log-price target, 80/20 split, random_state=42):

| Model | Heritage R² | Notes |
|---|---|---|
| **LightGBM** | **0.618** | Best — fast and accurate |
| **CatBoost** | **0.616** | Tied for best — strong out-of-the-box |
| Gradient Boosting | 0.591 | Sklearn baseline boosting |
| Random Forest | 0.596 | Robust; close to GBR |
| Decision Tree | 0.354 | Single tree under-fits |
| Linear Regression | 0.194 | Linear models cap out at ~0.20 |
| Ridge / Lasso / Elastic Net | 0.16–0.20 | Regularization didn't help much |

**Takeaway:** Tree-boosting is the right family for this problem. LightGBM and CatBoost are essentially tied — both belong in the deck because they represent two different boosting families (gradient-based vs. ordered boosting).

---

## 4. Architect Value Leaderboard (Page 2)

When we rank architects by their portfolio's median price-per-sqft premium over the city baseline (filtering for ≥ 5 buildings and ≥ $50/sqft sales to drop non-market transfers), the leaderboard surfaces a distinct group of "premium-portfolio" architects whose buildings consistently sell above the Manhattan median. This validates the qualitative story: certain firms (e.g., McKim, Mead & White; C. P. H. Gilbert) attach durable value to their work.

---

## 5. Data Anomalies & Pre-processing Decisions

| Issue | Resolution |
|---|---|
| `is_landmark` is all 0, `in_historic_district` is all 1 | The merge filter intersects with the Landmarks DB, which is itself a list of historic-district / landmark properties. Both flags collapse to constants in the merged subset. We keep them in the schema for clarity but they carry no signal here. |
| `preservation_level` has a single value (`"Historic District"`) | Same root cause. Dropped from feature set. |
| `sale_year` is 100% 2025 | Single year of rolling-sales data. Dropped; only `sale_month` is informative. |
| `price_per_sqft` has a long left tail (25th percentile = $7) | Internal transfers, easements, and tax-only transactions pollute the bottom. We require ≥ $100K sale price AND (for the architect leaderboard) ≥ $50/sqft. |
| `years_since_alteration` set to 2026 for never-altered buildings | Effectively a "never altered" sentinel. We keep it as-is — the model can split on it. |
| `residential_units` / `commercial_units` / `total_units` have ~88% NaN | Treated as "no units recorded → 0" rather than median imputation, since missing usually means non-residential or commercial-only. |

---

## 6. Modeling Choices That Mattered

- **Log-price target.** `np.log1p(sale_price)` instead of raw price. Manhattan prices span 4 orders of magnitude; log-transform makes residuals roughly Gaussian and stops the optimizer from chasing a few $200M outliers.
- **Display in dollars.** All UI metrics convert back via `np.expm1` so users see "$3.2M predicted" rather than "log-price 14.98."
- **80% prediction interval** on Page 6 (Property Valuator) uses LightGBM quantile regression at the 10th and 90th percentiles. This is more honest than a single point estimate.
- **`assess_total` is in BASELINE** even though it's arguably a forward-leaking feature. We kept it because (a) it's publicly available before any sale and (b) removing it makes the heritage uplift artificially large. Honest baseline beats flattering uplift.

---

## 7. What We Would Add With More Data

- **Interior specs.** Bathroom count, kitchen condition, ceiling height, view tier, and renovation cost would close most of the remaining R² gap. Public datasets don't expose these — Zillow, Compass, and StreetEasy have them and that's why their AVMs hit R² > 0.85.
- **Macro features.** Mortgage rates at sale time, neighborhood crime/school index, walkability score.
- **Multi-year sales.** A panel covering 5–10 years would let us model price *appreciation* per architect, not just price levels.
