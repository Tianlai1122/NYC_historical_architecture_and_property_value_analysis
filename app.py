"""
Manhattan Heritage Property Valuation
Financial Premium of Historic Preservation in Manhattan

Central Question: Does architectural heritage create measurable market value?
"""

import streamlit as st
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manhattan Heritage Valuation",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME SYSTEM
# ─────────────────────────────────────────────────────────────
THEMES = {
    "Classic Dark": {
        "header_grad": "linear-gradient(135deg, #d4a5ff 0%, #f0c6ff 100%)",
        "app_bg": "#0e1117",
        "sidebar_bg": "#161b22",
        "text": "#e6e6e6",
        "accent": "#8b5cf6",
        "card_bg": "rgba(139,92,246,0.12)",
        "card_border": "rgba(139,92,246,0.25)",
        "plotly": "plotly_dark",
        "mpl": "dark_background",
        "map_style": "carto-darkmatter",
    },
    "Light Minimal": {
        "header_grad": "linear-gradient(135deg, #1a5276 0%, #2e86c1 100%)",
        "app_bg": "#ffffff",
        "sidebar_bg": "#f7f8fa",
        "text": "#1a1a2e",
        "accent": "#1a5276",
        "card_bg": "rgba(26,82,118,0.06)",
        "card_border": "rgba(26,82,118,0.15)",
        "plotly": "plotly_white",
        "mpl": "seaborn-v0_8-whitegrid",
        "map_style": "carto-positron",
    },
    "Warm Tone": {
        "header_grad": "linear-gradient(135deg, #c0392b 0%, #e74c3c 100%)",
        "app_bg": "#1a1a1a",
        "sidebar_bg": "#111111",
        "text": "#e8e8e8",
        "accent": "#e74c3c",
        "card_bg": "rgba(231,76,60,0.10)",
        "card_border": "rgba(231,76,60,0.25)",
        "plotly": "plotly_dark",
        "mpl": "dark_background",
        "map_style": "carto-darkmatter",
    },
}

# Sidebar
st.sidebar.markdown("### Manhattan Heritage Valuation")
theme_name = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=1)
T = THEMES[theme_name]

try:
    plt.style.use(T["mpl"])
except Exception:
    plt.style.use("default")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "1 — Business Case & Data",
        "2 — Visualizations & Maps",
        "3 — Prediction Models",
        "4 — Feature Importance",
        "5 — Hyperparameter Tuning",
    ],
)

# Inject CSS
st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{background: {T["app_bg"]} !important;}}
  [data-testid="stSidebar"] {{background: {T["sidebar_bg"]} !important;}}
  .stMarkdown p, .stMarkdown li {{color: {T["text"]} !important;}}
  h1,h2,h3,h4 {{color: {T["text"]} !important;}}

  .page-title {{
    font-size: 2.4rem; font-weight: 800;
    background: {T["header_grad"]};
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
  }}
  .page-subtitle {{
    font-size: 1.05rem; color: {T["text"]}; opacity: 0.65;
    margin-bottom: 1.5rem; font-weight: 300;
  }}
  .card {{
    background: {T["card_bg"]};
    border: 1px solid {T["card_border"]};
    border-radius: 10px; padding: 1.2rem;
    margin-bottom: 0.8rem;
  }}
  .card .val {{font-size: 1.6rem; font-weight: 700; margin: 0; color: {T["text"]};}}
  .card .lbl {{font-size: 0.8rem; opacity: 0.6; margin: 0; color: {T["text"]};}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    d = pd.read_csv("Manhattan_Heritage_Analysis.csv", low_memory=False)
    d["sale_date"] = pd.to_datetime(d["sale_date"], errors="coerce")

    # Clean "0" placeholders in text columns — these are not real values
    zero_cols = ["building_name", "landmark_orig", "landmark_new",
                 "style_secondary", "material_secondary"]
    for col in zero_cols:
        if col in d.columns:
            d[col] = d[col].astype(str).replace("0", "")

    # Create a clean display label: use building name if available, else address
    d["display_name"] = d["building_name"].where(d["building_name"].str.len() > 0, d["address"])

    return d

df = load_data()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Dataset:** {len(df):,} rows, {len(df.columns)} cols  \n"
    f"**Median Price:** ${df['sale_price'].median():,.0f}  \n"
    f"**Architects:** {df['architect'].nunique()}"
)

# ─────────────────────────────────────────────────────────────
# FEATURE SETS
# ─────────────────────────────────────────────────────────────
BASELINE = [
    "gross_sqft", "land_sqft", "num_floors", "lot_area", "lot_depth",
    "lot_frontage", "building_depth", "building_frontage",
    "assess_total", "assess_land", "built_far", "resid_far",
]
HERITAGE = [
    "building_age", "architect_prestige_score", "rare_style_score",
    "is_landmark", "in_historic_district", "is_altered",
    "construction_era_encoded", "material_primary_encoded", "style_primary_encoded",
]

@st.cache_data
def prepare_features(dataframe):
    # Filter non-market transactions (donations, estate transfers, internal transfers)
    d = dataframe[dataframe["sale_price"] >= 100_000].copy()
    d["log_price"] = np.log1p(d["sale_price"])
    for cat in ["construction_era", "material_primary", "style_primary"]:
        if cat in d.columns:
            le = LabelEncoder()
            d[f"{cat}_encoded"] = le.fit_transform(d[cat].fillna("Unknown").astype(str))
    all_f = BASELINE + HERITAGE
    avail = [f for f in all_f if f in d.columns]
    d[avail] = d[avail].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=avail, thresh=len(avail) - 4)
    for c in avail:
        d[c] = d[c].fillna(d[c].median())
    return d, [f for f in BASELINE if f in avail], [f for f in HERITAGE if f in avail]

mdf, FB, FH = prepare_features(df)
FALL = FB + FH


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def title(text, sub=""):
    st.markdown(f'<p class="page-title">{text}</p>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<p class="page-subtitle">{sub}</p>', unsafe_allow_html=True)

def cards(items):
    cols = st.columns(len(items))
    for col, (lbl, val) in zip(cols, items):
        col.markdown(
            f'<div class="card"><p class="val">{val}</p><p class="lbl">{lbl}</p></div>',
            unsafe_allow_html=True,
        )

def train_eval(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    return {
        "R2_train": r2_score(ytr, model.predict(Xtr)),
        "R2_test": r2_score(yte, model.predict(Xte)),
        "MAE": mean_absolute_error(yte, model.predict(Xte)),
        "RMSE": np.sqrt(mean_squared_error(yte, model.predict(Xte))),
        "preds": model.predict(Xte),
        "model": model,
    }


# =================================================================
# PAGE 1 — BUSINESS CASE & DATA
# =================================================================
def page1():
    title("Manhattan Heritage Valuation",
          "How much does a building's architectural character affect its market price?")

    st.markdown("---")

    # ── Problem framing ──
    st.markdown("### The Research Question")
    st.markdown("""
Most real-estate pricing models rely on **structural features** — square footage,
lot size, number of floors, zoning, and assessment values. These are important,
but they ignore *why certain buildings in Manhattan command extraordinary premiums*.

**Our approach:** We add a second layer of **architectural and preservation features**
— construction era, architect prestige, facade material, architectural style, landmark
status, and historic district membership — to see if these create a measurable
**"preservation premium"** in sale prices.
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Baseline Model (Structural)")
        st.markdown("""
- Lot & building dimensions
- Floor-to-area ratio (FAR)
- Tax assessment values
- Gross square footage

*Standard variables found in any real-estate dataset.*
        """)
    with c2:
        st.markdown("#### Heritage-Enhanced Model (Ours)")
        st.markdown("""
- Architectural style & facade material
- Architect prestige score
- Construction era (Pre-1850 → Modern)
- Landmark & historic district status
- Alteration history

*Preservation-specific variables that capture aesthetic & historic value.*
        """)

    st.markdown("---")

    # ── Data sources ──
    st.markdown("### Data Sources")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**NYC Rolling Sales**")
        st.caption("Sale prices, dates, building class, square footage.")
        st.metric("Records", "18,817")
    with d2:
        st.markdown("**MapPLUTO**")
        st.caption("Structural features, zoning, GPS coordinates, assessments.")
        st.metric("Records", "42,600")
    with d3:
        st.markdown("**Landmark Database**")
        st.caption("Architect, style, material, historic district, alteration history.")
        st.metric("Records", "14,610")

    st.markdown("""
All datasets merged on **BBL (Borough-Block-Lot)**, NYC's unique property identifier.
After filtering out \\$0 non-market transactions:
    """)
    cards([
        ("Final merged records", f"{len(df):,}"),
        ("Engineered features", f"{len(df.columns)}"),
        ("Unique architects", f"{df['architect'].nunique()}"),
        ("With GPS coordinates", f"{df.dropna(subset=['latitude']).shape[0]:,}"),
    ])

    st.markdown("---")

    # ── Data preview ──
    st.markdown("### Data Preview")
    preview_cols = [
        "BBL", "sale_price", "gross_sqft", "num_floors", "building_age",
        "construction_era", "style_primary", "material_primary",
        "architect", "in_historic_district", "neighborhood",
    ]
    avail_cols = [c for c in preview_cols if c in df.columns]
    st.dataframe(df[avail_cols].head(20), use_container_width=True, height=380)

    with st.expander("Column types & missing values"):
        info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notna().sum().values,
            "Missing %": (df.isna().mean() * 100).round(1).astype(str) + "%",
        })
        st.dataframe(info, use_container_width=True, height=400)


# =================================================================
# PAGE 2 — VISUALIZATIONS & MAPS
# =================================================================
def page2():
    title("Visualizations & Maps",
          "Exploring how architectural heritage relates to property value across Manhattan")

    viz = df[
        (df["sale_price"] < df["sale_price"].quantile(0.97))
        & (df["price_per_sqft"] < df["price_per_sqft"].quantile(0.97))
        & df["latitude"].notna()
    ].copy()

    section = st.radio(
        "Section",
        ["Interactive Map", "Price Analysis", "Architectural Premiums", "Era & Correlation"],
        horizontal=True,
    )

    # ── MAP ──
    if section == "Interactive Map":
        st.markdown("### Property Map")
        st.caption("Each dot is a historic property. Choose what variable controls color and size.")

        c1, c2 = st.columns(2)
        with c1:
            color_var = st.selectbox("Color by", [
                "sale_price", "price_per_sqft", "building_age",
                "architect_prestige_score", "num_floors", "assess_total",
            ])
        with c2:
            size_var = st.selectbox("Size by", [
                "gross_sqft", "sale_price", "num_floors", "building_age",
            ])

        map_d = viz.dropna(subset=[color_var, size_var]).copy()
        smax = map_d[size_var].quantile(0.95)
        map_d["_sz"] = (map_d[size_var].clip(0, smax) / smax * 16 + 3)

        fig = px.scatter_mapbox(
            map_d, lat="latitude", lon="longitude",
            color=color_var, size="_sz",
            hover_name="display_name",
            hover_data={
                "sale_price": ":,.0f", "construction_era": True,
                "style_primary": True, "architect": True,
                "historic_district": True,
                "latitude": False, "longitude": False, "_sz": False,
            },
            color_continuous_scale="Plasma",
            zoom=12, center={"lat": 40.754, "lon": -73.987},
            height=580, template=T["plotly"],
        )
        fig.update_layout(
            mapbox_style=T["map_style"],
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Construction Era Map")
        st.caption("See how different architectural periods cluster geographically.")
        era_colors = {
            "Pre-1850": "#e63946", "1850–1899": "#f77f00", "1900–1919": "#fcbf49",
            "1920–1939 (Art Deco)": "#2ec4b6", "1940–1969 (Mid-Century)": "#457b9d",
            "1970+": "#a8dadc",
        }
        era_d = viz[viz["construction_era"].isin(era_colors.keys())]
        fig2 = px.scatter_mapbox(
            era_d, lat="latitude", lon="longitude",
            color="construction_era", color_discrete_map=era_colors,
            hover_name="display_name",
            hover_data={"sale_price": ":,.0f", "construction_era": True,
                        "latitude": False, "longitude": False},
            zoom=12, center={"lat": 40.754, "lon": -73.987},
            height=540, template=T["plotly"],
        )
        fig2.update_layout(
            mapbox_style=T["map_style"],
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── PRICE ──
    elif section == "Price Analysis":
        st.markdown("### Price Distributions")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(viz, x="sale_price", nbins=50,
                               title="Sale Price Distribution",
                               color_discrete_sequence=[T["accent"]],
                               template=T["plotly"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(viz.dropna(subset=["price_per_sqft"]),
                               x="price_per_sqft", nbins=50,
                               title="Price per SqFt Distribution",
                               color_discrete_sequence=["#e74c3c"],
                               template=T["plotly"])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Neighborhood Price Ranking (Top 20)")
        hood = (viz.groupby("neighborhood")["sale_price"]
                .median().sort_values(ascending=False).head(20).reset_index())
        fig = px.bar(hood, x="sale_price", y="neighborhood", orientation="h",
                     color="sale_price", color_continuous_scale="Plasma",
                     template=T["plotly"], height=550,
                     labels={"sale_price": "Median Sale Price ($)", "neighborhood": ""})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Price by Historic District")
        hd = viz[viz["historic_district"].str.len() > 0]
        fig = px.box(hd, x="historic_district", y="price_per_sqft",
                     template=T["plotly"], height=480,
                     labels={"historic_district": "", "price_per_sqft": "$/SqFt"})
        fig.update_xaxes(tickangle=35)
        st.plotly_chart(fig, use_container_width=True)

    # ── ARCHITECTURAL ──
    elif section == "Architectural Premiums":
        st.markdown("### Architectural Style vs Price")
        sp = (viz.groupby("style_primary")["price_per_sqft"]
              .agg(["median", "count"]).query("count>=5")
              .sort_values("median", ascending=False).head(20).reset_index())
        fig = px.bar(sp, x="median", y="style_primary", orientation="h",
                     color="median", color_continuous_scale="Viridis",
                     hover_data={"count": True}, template=T["plotly"], height=560,
                     labels={"median": "Median $/SqFt", "style_primary": ""})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Facade Material")
            mp = (viz.groupby("material_primary")["price_per_sqft"]
                  .agg(["median", "count"]).query("count>=5")
                  .sort_values("median", ascending=False).head(12).reset_index())
            fig = px.bar(mp, x="median", y="material_primary", orientation="h",
                         color="median", color_continuous_scale="Magma",
                         template=T["plotly"], height=440,
                         labels={"median": "$/SqFt", "material_primary": ""})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Top Architects by Median Sale Price")
            ap = (viz.groupby("architect")["sale_price"]
                  .agg(["median", "count"]).query("count>=5")
                  .sort_values("median", ascending=False).head(12).reset_index())
            fig = px.bar(ap, x="median", y="architect", orientation="h",
                         color="median", color_continuous_scale="Plasma",
                         hover_data={"count": True}, template=T["plotly"], height=440,
                         labels={"median": "Median Sale $", "architect": ""})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Architect Prestige vs Price")
        scatter_d = viz.dropna(subset=["architect_prestige_score", "price_per_sqft"])
        fig = px.scatter(scatter_d, x="architect_prestige_score", y="price_per_sqft",
                         color="construction_era", hover_name="architect",
                         opacity=0.5, trendline="ols", template=T["plotly"],
                         labels={"architect_prestige_score": "Prestige Score",
                                 "price_per_sqft": "$/SqFt"})
        st.plotly_chart(fig, use_container_width=True)

    # ── ERA & CORRELATION ──
    else:
        st.markdown("### Price by Construction Era")
        era_order = ["Pre-1850", "1850–1899", "1900–1919",
                     "1920–1939 (Art Deco)", "1940–1969 (Mid-Century)", "1970+"]
        era_d = viz[viz["construction_era"].isin(era_order)]
        fig = px.box(era_d, x="construction_era", y="price_per_sqft",
                     category_orders={"construction_era": era_order},
                     color="construction_era",
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     template=T["plotly"], height=450,
                     labels={"construction_era": "", "price_per_sqft": "$/SqFt"})
        fig.update_layout(showlegend=False, xaxis_tickangle=15)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Building Age vs Price")
            fig = px.scatter(viz.dropna(subset=["building_age"]),
                             x="building_age", y="sale_price",
                             color="construction_era", opacity=0.4,
                             trendline="ols", template=T["plotly"],
                             labels={"building_age": "Age (years)", "sale_price": "Sale Price ($)"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Altered vs Original Buildings")
            fig = px.violin(viz, x="is_altered", y="price_per_sqft",
                            box=True, color="is_altered", template=T["plotly"],
                            labels={"is_altered": "Altered (1=Yes)", "price_per_sqft": "$/SqFt"})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Correlation Matrix")
        corr_cols = ["sale_price", "price_per_sqft", "building_age", "num_floors",
                     "gross_sqft", "assess_total", "architect_prestige_score",
                     "rare_style_score", "is_altered", "in_historic_district"]
        ac = [c for c in corr_cols if c in viz.columns]
        cm = viz[ac].corr()
        mask = np.triu(np.ones_like(cm, dtype=bool))
        fig_c, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(cm, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close()


# =================================================================
# PAGE 3 — PREDICTION MODELS
# =================================================================
def page3():
    title("Prediction Models",
          "Comparing Baseline (structural-only) vs Heritage-Enhanced (all features)")

    c1, c2, c3 = st.columns(3)
    with c1:
        test_sz = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    with c2:
        rs = st.number_input("Random state", 0, 100, 42)
    with c3:
        do_scale = st.checkbox("Scale features", True)

    X_b = mdf[FB].values
    X_a = mdf[FALL].values
    y = mdf["log_price"].values

    Xb_tr, Xb_te, y_tr, y_te = train_test_split(X_b, y, test_size=test_sz, random_state=rs)
    Xa_tr, Xa_te, _, _ = train_test_split(X_a, y, test_size=test_sz, random_state=rs)

    if do_scale:
        sc1 = StandardScaler(); Xb_tr_s = sc1.fit_transform(Xb_tr); Xb_te_s = sc1.transform(Xb_te)
        sc2 = StandardScaler(); Xa_tr_s = sc2.fit_transform(Xa_tr); Xa_te_s = sc2.transform(Xa_te)
    else:
        Xb_tr_s, Xb_te_s = Xb_tr, Xb_te
        Xa_tr_s, Xa_te_s = Xa_tr, Xa_te

    MODELS = {
        "Linear Regression":  (LinearRegression(), True),
        "Ridge Regression":   (Ridge(alpha=1.0), True),
        "Lasso Regression":   (Lasso(alpha=0.05), True),
        "Decision Tree":      (DecisionTreeRegressor(max_depth=10, random_state=rs), False),
        "Random Forest":      (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1), False),
        "Gradient Boosting":  (GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=rs), False),
    }

    st.markdown("---")
    st.markdown("### Select models to train")
    sel = []
    cols = st.columns(6)
    for (nm, _), col in zip(MODELS.items(), cols):
        if col.checkbox(nm, True, key=f"m_{nm}"):
            sel.append(nm)

    if not sel:
        st.warning("Select at least one model.")
        return

    base_rows, herit_rows = [], []
    trained_h = {}

    with st.spinner("Training models..."):
        for nm in sel:
            proto, needs_scale = MODELS[nm]
            mb = copy.deepcopy(proto)
            mh = copy.deepcopy(proto)
            if needs_scale:
                rb = train_eval(mb, Xb_tr_s, y_tr, Xb_te_s, y_te)
                rh = train_eval(mh, Xa_tr_s, y_tr, Xa_te_s, y_te)
            else:
                rb = train_eval(mb, Xb_tr, y_tr, Xb_te, y_te)
                rh = train_eval(mh, Xa_tr, y_tr, Xa_te, y_te)
            base_rows.append({"Model": nm, **{k: v for k, v in rb.items() if k != "preds" and k != "model"}})
            herit_rows.append({"Model": nm, **{k: v for k, v in rh.items() if k != "preds" and k != "model"}})
            herit_rows[-1]["preds"] = rh["preds"]
            trained_h[nm] = mh

    dfb = pd.DataFrame(base_rows).set_index("Model")
    dfh = pd.DataFrame(herit_rows).set_index("Model")

    st.session_state["trained_h"] = trained_h
    st.session_state["feat_all"] = FALL
    st.session_state["Xa_tr"] = Xa_tr
    st.session_state["Xa_te"] = Xa_te
    st.session_state["y_tr"] = y_tr
    st.session_state["y_te"] = y_te

    # ── Results side-by-side ──
    st.markdown("---")
    st.markdown("### Results: Baseline vs Heritage")
    c1, c2 = st.columns(2)
    fmt_cols = ["R2_train", "R2_test", "MAE", "RMSE"]
    with c1:
        st.markdown("**Baseline (structural only)**")
        st.dataframe(dfb[fmt_cols].round(4), use_container_width=True)
    with c2:
        st.markdown("**Heritage-Enhanced (all features)**")
        st.dataframe(dfh[fmt_cols].round(4), use_container_width=True)

    # ── Uplift chart ──
    st.markdown("---")
    st.markdown("### Heritage Uplift")
    st.caption("How much does adding architectural features improve each model?")
    up = pd.DataFrame({
        "Model": sel,
        "Baseline R²": [dfb.loc[n, "R2_test"] for n in sel],
        "Heritage R²": [dfh.loc[n, "R2_test"] for n in sel],
    })
    up["Uplift"] = up["Heritage R²"] - up["Baseline R²"]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Baseline", x=up["Model"], y=up["Baseline R²"],
                         marker_color="steelblue"))
    fig.add_trace(go.Bar(name="Heritage", x=up["Model"], y=up["Heritage R²"],
                         marker_color=T["accent"]))
    fig.update_layout(barmode="group", template=T["plotly"], height=400,
                      yaxis_title="R² (Test)")
    st.plotly_chart(fig, use_container_width=True)

    best = up.loc[up["Heritage R²"].idxmax()]
    st.success(
        f"Best Heritage Model: **{best['Model']}** — "
        f"R² = {best['Heritage R²']:.4f}  "
        f"(+{best['Uplift']:.4f} over baseline)"
    )

    # ── Actual vs Predicted ──
    st.markdown("---")
    st.markdown("### Actual vs Predicted")
    pick = st.selectbox("Model", sel, key="avp")
    yp = dfh.loc[pick, "preds"]
    fig = px.scatter(x=y_te, y=yp, opacity=0.4, template=T["plotly"],
                     labels={"x": "Actual (log price)", "y": "Predicted (log price)"})
    fig.add_shape(type="line", x0=y_te.min(), x1=y_te.max(),
                  y0=y_te.min(), y1=y_te.max(),
                  line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)


# =================================================================
# PAGE 4 — FEATURE IMPORTANCE
# =================================================================
def page4():
    title("Feature Importance & Explainability",
          "Which features drive the Heritage model — baseline structural or preservation characteristics?")

    if "trained_h" not in st.session_state:
        st.info("Please train models on the **Prediction Models** page first, then return here.")
        return

    trained = st.session_state["trained_h"]
    fnames = st.session_state["feat_all"]
    Xa_te = st.session_state["Xa_te"]
    Xa_tr = st.session_state["Xa_tr"]

    sel = st.selectbox("Model", list(trained.keys()))
    model = trained[sel]

    section = st.radio("View", ["Feature Importance", "SHAP Analysis"], horizontal=True)

    if section == "Feature Importance":
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_)
        else:
            st.warning("This model does not expose importance values.")
            return

        fi = pd.DataFrame({"Feature": fnames, "Importance": imp})
        fi["Type"] = fi["Feature"].apply(lambda f: "Baseline" if f in FB else "Heritage")
        fi = fi.sort_values("Importance", ascending=True)

        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color="Type",
                     color_discrete_map={"Baseline": "steelblue", "Heritage": T["accent"]},
                     template=T["plotly"], height=550,
                     labels={"Importance": "Importance", "Feature": ""})
        st.plotly_chart(fig, use_container_width=True)

        total = fi["Importance"].sum()
        h_share = fi[fi["Type"] == "Heritage"]["Importance"].sum()
        st.info(f"Heritage features account for **{h_share/total*100:.1f}%** of total model importance.")

        if hasattr(model, "coef_"):
            st.markdown("---")
            st.markdown("### Coefficient Direction")
            coef = pd.DataFrame({
                "Feature": fnames, "Coefficient": model.coef_,
                "Type": ["Baseline" if f in FB else "Heritage" for f in fnames],
            }).sort_values("Coefficient", key=abs, ascending=True)
            fig = px.bar(coef, x="Coefficient", y="Feature", orientation="h",
                         color="Type",
                         color_discrete_map={"Baseline": "steelblue", "Heritage": T["accent"]},
                         template=T["plotly"], height=550)
            fig.add_vline(x=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

    else:  # SHAP
        try:
            import shap
            with st.spinner("Computing SHAP values (may take a few seconds)..."):
                if hasattr(model, "feature_importances_"):
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(Xa_te[:250])
                else:
                    explainer = shap.LinearExplainer(model, Xa_tr[:500])
                    sv = explainer.shap_values(Xa_te[:250])

            st.markdown("### SHAP Summary Plot")
            fig_s, ax = plt.subplots(figsize=(11, 7))
            shap.summary_plot(sv, Xa_te[:250], feature_names=fnames, show=False, max_display=15)
            st.pyplot(fig_s)
            plt.close()

            st.markdown("---")
            st.markdown("### Individual Prediction Breakdown")
            idx = st.slider("Sample index", 0, min(249, Xa_te.shape[0] - 1), 0)
            pred_log = model.predict(Xa_te[idx:idx + 1])[0]
            st.metric("Predicted Sale Price", f"${np.expm1(pred_log):,.0f}")

            fig_w, _ = plt.subplots(figsize=(10, 5))
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = ev[0]
            explanation = shap.Explanation(values=sv[idx], base_values=ev,
                                           data=Xa_te[idx], feature_names=fnames)
            shap.waterfall_plot(explanation, show=False, max_display=12)
            st.pyplot(fig_w)
            plt.close()

        except ImportError:
            st.error("SHAP is not installed. Run `pip install shap` and restart the app.")
        except Exception as e:
            st.error(f"SHAP error: {e}")
            st.info("Try selecting a tree-based model for more reliable SHAP analysis.")


# =================================================================
# PAGE 5 — HYPERPARAMETER TUNING
# =================================================================
def page5():
    title("Hyperparameter Tuning",
          "Grid search with optional Weights & Biases experiment tracking")

    X = mdf[FALL].values
    y = mdf["log_price"].values

    c1, c2 = st.columns(2)
    with c1:
        model_name = st.selectbox("Model", [
            "Ridge Regression", "Lasso Regression",
            "Decision Tree", "Random Forest", "Gradient Boosting",
        ])
    with c2:
        test_sz = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="hp_ts")

    st.markdown("---")
    st.markdown("### Hyperparameter Grid")
    grid = []

    if model_name == "Ridge Regression":
        alphas = st.multiselect("Alpha", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                                default=[0.01, 0.1, 1.0, 10.0, 100.0])
        grid = [{"alpha": a} for a in alphas]

    elif model_name == "Lasso Regression":
        alphas = st.multiselect("Alpha", [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                                default=[0.001, 0.01, 0.1, 0.5])
        grid = [{"alpha": a} for a in alphas]

    elif model_name == "Decision Tree":
        depths = st.multiselect("Max Depth", [2, 3, 5, 7, 10, 15, None], default=[3, 5, 10])
        mins = st.multiselect("Min Samples Split", [2, 5, 10, 20], default=[2, 5, 10])
        grid = [{"max_depth": d, "min_samples_split": s} for d in depths for s in mins]

    elif model_name == "Random Forest":
        nes = st.multiselect("N Estimators", [50, 100, 200, 300], default=[50, 100, 200])
        deps = st.multiselect("Max Depth", [5, 10, 15, None], default=[5, 10, 15])
        grid = [{"n_estimators": n, "max_depth": d} for n in nes for d in deps]

    else:  # Gradient Boosting
        lrs = st.multiselect("Learning Rate", [0.01, 0.05, 0.1, 0.2], default=[0.01, 0.1])
        nes = st.multiselect("N Estimators", [50, 100, 200], default=[50, 100])
        deps = st.multiselect("Max Depth", [3, 5, 7], default=[3, 5])
        grid = [{"learning_rate": lr, "n_estimators": n, "max_depth": d}
                for lr in lrs for n in nes for d in deps]

    st.caption(f"Total experiments: **{len(grid)}**")

    # W&B
    st.markdown("---")
    st.markdown("### Weights & Biases (Optional)")
    use_wb = st.checkbox("Log experiments to W&B", False)
    wb_proj = "manhattan-heritage-tuning"
    if use_wb:
        wb_proj = st.text_input("W&B Project Name", "manhattan-heritage-tuning")
        st.caption(
            "**Setup:** `pip install wandb` → `wandb login` (get API key from wandb.ai/settings) "
            "→ tick the checkbox above. Your runs will appear on your W&B dashboard."
        )

    st.markdown("---")
    if st.button("Run Tuning", type="primary"):
        if not grid:
            st.error("Configure at least one hyperparameter combination.")
            return

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_sz, random_state=42)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        needs_scale = model_name in ["Ridge Regression", "Lasso Regression"]

        wb_ok = False
        if use_wb:
            try:
                import wandb
                wb_ok = True
            except ImportError:
                st.warning("wandb not installed — logging locally only.")

        results = []
        prog = st.progress(0)
        for i, params in enumerate(grid):
            prog.progress((i + 1) / len(grid))

            if model_name == "Ridge Regression":       m = Ridge(**params)
            elif model_name == "Lasso Regression":     m = Lasso(**params)
            elif model_name == "Decision Tree":        m = DecisionTreeRegressor(**params, random_state=42)
            elif model_name == "Random Forest":        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            else:                                      m = GradientBoostingRegressor(**params, random_state=42)

            if needs_scale:
                m.fit(Xtr_s, ytr); ptr = m.predict(Xtr_s); pte = m.predict(Xte_s)
            else:
                m.fit(Xtr, ytr); ptr = m.predict(Xtr); pte = m.predict(Xte)

            metrics = {
                "r2_train": r2_score(ytr, ptr),
                "r2_test": r2_score(yte, pte),
                "mae": mean_absolute_error(yte, pte),
                "rmse": np.sqrt(mean_squared_error(yte, pte)),
            }
            results.append({**params, **metrics})

            if wb_ok:
                import wandb
                wandb.init(project=wb_proj, name=f"{model_name}_{i}",
                           config={"model": model_name, "test_size": test_sz, **params},
                           reinit=True)
                wandb.log(metrics)
                wandb.finish()

        prog.empty()
        res_df = pd.DataFrame(results).sort_values("r2_test", ascending=False)

        st.markdown("### Results")
        st.dataframe(
            res_df.style
            .highlight_max(subset=["r2_test"], color="#667eea")
            .highlight_min(subset=["rmse", "mae"], color="#2ecc71"),
            use_container_width=True,
        )

        best = res_df.iloc[0]
        pcols = [c for c in res_df.columns if c not in ["r2_train", "r2_test", "mae", "rmse"]]
        st.success(
            f"Best: R² = {best['r2_test']:.4f}, MAE = {best['mae']:.4f}, "
            f"RMSE = {best['rmse']:.4f} — Params: {dict(best[pcols])}"
        )

        # Charts
        if pcols:
            p0 = pcols[0]
            is_num = pd.api.types.is_numeric_dtype(res_df[p0])
            if is_num:
                fig = px.line(res_df.sort_values(p0), x=p0,
                              y=["r2_train", "r2_test"], markers=True,
                              template=T["plotly"], title=f"R² vs {p0}")
            else:
                fig = px.bar(res_df, x=p0, y=["r2_train", "r2_test"],
                             barmode="group", template=T["plotly"],
                             title=f"R² vs {p0}")
            st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(res_df, x="r2_train", y="r2_test", color="rmse",
                          color_continuous_scale="RdYlGn_r", template=T["plotly"],
                          title="Train vs Test R² (color = RMSE)")
        fig2.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                       line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig2, use_container_width=True)

        if wb_ok and use_wb:
            st.caption(f"View dashboard at wandb.ai → project `{wb_proj}`")


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────
if   "1" in page: page1()
elif "2" in page: page2()
elif "3" in page: page3()
elif "4" in page: page4()
elif "5" in page: page5()
