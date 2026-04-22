"""
Manhattan Heritage Property Valuation
Predicting Manhattan Property Sale Prices with Architectural Heritage Features

Central Question:
Can heritage-related variables improve the prediction of Manhattan property sale prices?
"""

import streamlit as st
import pandas as pd
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

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
    "Porcelain": {
        "header_grad": "linear-gradient(135deg, #1D1D1F 0%, #0071E3 100%)",
        "app_bg": "#FBFBFD",
        "sidebar_bg": "#F5F5F7",
        "text": "#1D1D1F",
        "muted": "#6E6E73",
        "accent": "#0071E3",
        "accent_soft": "#5AC8FA",
        "card_bg": "rgba(255,255,255,0.72)",
        "card_border": "rgba(0,0,0,0.08)",
        "shadow": "0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.06)",
        "plotly": "plotly_white",
        "mpl": "seaborn-v0_8-whitegrid",
        "map_style": "carto-positron",
    },
    "Graphite": {
        "header_grad": "linear-gradient(135deg, #0A84FF 0%, #BF5AF2 100%)",
        "app_bg": "#000000",
        "sidebar_bg": "#1C1C1E",
        "text": "#F5F5F7",
        "muted": "#8E8E93",
        "accent": "#0A84FF",
        "accent_soft": "#BF5AF2",
        "card_bg": "rgba(28,28,30,0.72)",
        "card_border": "rgba(255,255,255,0.10)",
        "shadow": "0 1px 2px rgba(0,0,0,0.4), 0 12px 32px rgba(0,0,0,0.5)",
        "plotly": "plotly_dark",
        "mpl": "dark_background",
        "map_style": "carto-darkmatter",
    },
    "Aurora": {
        "header_grad": "linear-gradient(135deg, #06B6D4 0%, #8B5CF6 100%)",
        "app_bg": "#0B0F1A",
        "sidebar_bg": "#111827",
        "text": "#E5E7EB",
        "muted": "#9CA3AF",
        "accent": "#06B6D4",
        "accent_soft": "#8B5CF6",
        "card_bg": "rgba(17,24,39,0.68)",
        "card_border": "rgba(6,182,212,0.18)",
        "shadow": "0 1px 2px rgba(0,0,0,0.4), 0 12px 40px rgba(6,182,212,0.08)",
        "plotly": "plotly_dark",
        "mpl": "dark_background",
        "map_style": "carto-darkmatter",
    },
}

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("### Manhattan Heritage Valuation")
theme_name = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=0)
T = THEMES[theme_name]

try:
    plt.style.use(T["mpl"])
except Exception:
    plt.style.use("default")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "1. Business Case & Data",
        "2. Visualizations & Maps",
        "3. Prediction Models",
        "4. Feature Importance",
        "5. Hyperparameter Tuning",
        "6. Property Valuator",
    ],
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.01em;
  }}

  [data-testid="stAppViewContainer"] {{background: {T["app_bg"]} !important;}}
  [data-testid="stSidebar"] {{background: {T["sidebar_bg"]} !important;}}
  [data-testid="stSidebar"] * {{color: {T["text"]} !important;}}
  .stMarkdown p, .stMarkdown li, .stMarkdown span {{color: {T["text"]} !important;}}
  h1, h2, h3, h4, h5 {{color: {T["text"]} !important; letter-spacing: -0.02em; font-weight: 700;}}

  .page-title {{
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    background: {T["header_grad"]};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
    letter-spacing: -0.03em;
  }}

  .page-subtitle {{
    font-size: 1.02rem;
    color: {T["muted"]};
    margin-bottom: 1.6rem;
    font-weight: 400;
    max-width: 52rem;
  }}

  .card {{
    background: {T["card_bg"]};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {T["card_border"]};
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    box-shadow: {T["shadow"]};
    transition: transform .18s ease, box-shadow .18s ease;
  }}

  .card:hover {{
    transform: translateY(-2px);
  }}

  .card .val {{
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0;
    color: {T["text"]};
    letter-spacing: -0.02em;
  }}

  .card .lbl {{
    font-size: 0.78rem;
    color: {T["muted"]};
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  .stButton > button {{
    background: {T["accent"]} !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.1rem !important;
    transition: all .15s ease;
  }}

  .stButton > button:hover {{
    filter: brightness(1.08);
    transform: translateY(-1px);
  }}

  [data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
  }}

  div[role="radiogroup"] > label {{
    border-radius: 999px !important;
    padding: 0.3rem 0.9rem !important;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    d = pd.read_csv("Manhattan_Heritage_Analysis.csv", low_memory=False)
    d["sale_date"] = pd.to_datetime(d["sale_date"], errors="coerce")

    zero_cols = [
        "building_name",
        "landmark_orig",
        "landmark_new",
        "style_secondary",
        "material_secondary",
    ]

    for col in zero_cols:
        if col in d.columns:
            d[col] = d[col].astype(str).replace("0", "")

    d["building_name"] = d["building_name"].fillna("").astype(str)
    d["address"] = d["address"].fillna("").astype(str)

    d["display_name"] = d["building_name"].where(
        d["building_name"].str.len() > 0,
        d["address"]
    )

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
    "gross_sqft",
    "land_sqft",
    "num_floors",
    "lot_area",
    "lot_depth",
    "lot_frontage",
    "building_depth",
    "building_frontage",
    "residential_units",
    "commercial_units",
    "total_units",
    "assess_total",
    "assess_land",
    "exempt_total",
    "built_far",
    "resid_far",
    "comm_far",
    "facil_far",
    "zoning_encoded",
    "building_class_code_encoded",
    "sale_month",
]

HERITAGE = [
    "building_age",
    "construction_era_encoded",
    "architect_prestige_score",
    "architect_building_count",
    "rare_style_score",
    "style_frequency",
    "is_landmark",
    "in_historic_district",
    "is_altered",
    "years_since_alteration",
    "material_primary_encoded",
    "style_primary_encoded",
]

@st.cache_data
def prepare_features(dataframe):
    d = dataframe[dataframe["sale_price"] >= 100_000].copy()
    d["log_price"] = np.log1p(d["sale_price"])

    cat_cols = [
        "construction_era",
        "material_primary",
        "style_primary",
        "zoning",
        "building_class_code",
    ]

    for cat in cat_cols:
        if cat in d.columns:
            le = LabelEncoder()
            d[f"{cat}_encoded"] = le.fit_transform(
                d[cat].fillna("Unknown").astype(str)
            )

    for c in ["residential_units", "commercial_units", "total_units"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    all_f = BASELINE + HERITAGE
    avail = [f for f in all_f if f in d.columns]

    d[avail] = d[avail].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=avail, thresh=len(avail) - 6)

    for c in avail:
        d[c] = d[c].fillna(d[c].median())

    return d, [f for f in BASELINE if f in avail], [f for f in HERITAGE if f in avail]


mdf, FB, FH = prepare_features(df)
FALL = FB + FH

# ─────────────────────────────────────────────────────────────
# FRIENDLY LABELS
# ─────────────────────────────────────────────────────────────
FRIENDLY_LABELS = {
    "gross_sqft": "Gross Floor Area",
    "land_sqft": "Lot Size",
    "num_floors": "Number of Floors",
    "lot_area": "Lot Area",
    "lot_depth": "Lot Depth",
    "lot_frontage": "Lot Frontage",
    "building_depth": "Building Depth",
    "building_frontage": "Building Frontage",
    "residential_units": "Residential Units",
    "commercial_units": "Commercial Units",
    "total_units": "Total Units",
    "assess_total": "Total Assessment Value",
    "assess_land": "Land Assessment Value",
    "exempt_total": "Tax Exemption Amount",
    "built_far": "Built FAR",
    "resid_far": "Residential FAR Cap",
    "comm_far": "Commercial FAR Cap",
    "facil_far": "Facility FAR Cap",
    "zoning_encoded": "Zoning District",
    "building_class_code_encoded": "Building Class",
    "sale_month": "Sale Month",
    "sale_price": "Sale Price",
    "price_per_sqft": "Price per Sqft",
    "building_age": "Building Age",
    "construction_era_encoded": "Construction Era",
    "architect_prestige_score": "Architect Prestige Score",
    "architect_building_count": "Architect Portfolio Size",
    "rare_style_score": "Rare Style Score",
    "style_frequency": "Style Frequency",
    "is_landmark": "Individual Landmark",
    "in_historic_district": "In Historic District",
    "is_altered": "Has Been Altered",
    "years_since_alteration": "Years Since Alteration",
    "material_primary_encoded": "Primary Facade Material",
    "style_primary_encoded": "Primary Architectural Style",
}

def flabel(var):
    pretty = FRIENDLY_LABELS.get(var)
    return f"{pretty} ({var})" if pretty else var


MANHATTAN_FACTS = [
    "🏙️ The Empire State Building was built in just 410 days.",
    "🎨 Greenwich Village Historic District was designated in 1969.",
    "🏛️ The Landmarks Preservation Commission was created in 1965.",
    "🧱 Brownstone refers to a specific sandstone historically used in New York townhouses.",
    "💡 The Flatiron Building was completed in 1902.",
    "🌆 Manhattan contains many protected historic districts and landmark buildings.",
    "🏗️ Beaux-Arts and Art Deco styles are deeply associated with Manhattan architecture.",
    "⛪ St. Patrick's Cathedral took more than two decades to complete.",
    "🌃 The Chrysler Building briefly held the title of world’s tallest building.",
]

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
    pred_log = model.predict(Xte)

    actual_dollars = np.expm1(yte)
    pred_dollars = np.expm1(pred_log)

    return {
        "R2_train": r2_score(ytr, model.predict(Xtr)),
        "R2_test": r2_score(yte, pred_log),
        "MAE_dollars": mean_absolute_error(actual_dollars, pred_dollars),
        "RMSE_dollars": np.sqrt(mean_squared_error(actual_dollars, pred_dollars)),
        "preds": pred_log,
        "model": model,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def wiki_lookup(query):
    if not query or not query.strip():
        return None

    try:
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            timeout=4,
        ).json()

        hits = s.get("query", {}).get("search", [])
        if not hits:
            return None

        page_title = hits[0]["title"]

        p = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "pageimages|extracts",
                "exintro": 1,
                "explaintext": 1,
                "piprop": "original",
                "titles": page_title,
                "redirects": 1,
                "format": "json",
            },
            timeout=4,
        ).json()

        page = next(iter(p.get("query", {}).get("pages", {}).values()), {})
        img = page.get("original", {}).get("source")
        extract = (page.get("extract") or "").strip()

        return {
            "title": page.get("title", page_title),
            "url": f"https://en.wikipedia.org/wiki/{page.get('title', page_title).replace(' ', '_')}",
            "extract": extract[:500] + ("..." if len(extract) > 500 else ""),
            "image": img,
        }

    except Exception:
        return None


# =================================================================
# PAGE 1. BUSINESS CASE & DATA
# =================================================================
def page1():
    title(
        "Manhattan Heritage Valuation",
        "Can architectural and preservation features make property price predictions more accurate?"
    )

    st.markdown("---")

    st.markdown("### Research Question")
    st.markdown("""
**Can heritage-related variables improve the prediction of Manhattan property sale prices?**

This project uses a predictive modeling approach. It does not try to prove that historic preservation directly causes higher property values. Instead, it asks whether architectural and preservation-related information can help machine-learning models predict sale prices more accurately.

The project compares two model types:

1. **Baseline Model** — uses standard real-estate variables.
2. **Heritage-Enhanced Model** — uses the same baseline variables plus architectural and preservation variables.

If the heritage-enhanced model performs better, this suggests that heritage-related features contain useful predictive information beyond standard property characteristics.
    """)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Baseline Model")
        st.markdown("""
The baseline model represents a standard real-estate valuation approach. It uses physical, financial, and zoning-related variables, including:

- Gross square footage
- Lot size
- Number of floors
- Unit count
- Assessment value
- Building class
- Zoning district
- FAR variables

These variables describe the property mainly as a real-estate asset.
        """)

    with c2:
        st.markdown("#### Heritage-Enhanced Model")
        st.markdown("""
The heritage-enhanced model keeps all baseline variables and adds architectural and preservation-related features, including:

- Construction era
- Building age
- Architect
- Architectural style
- Facade material
- Landmark status
- Historic district membership
- Alteration history
- Style rarity
- Architect portfolio size

These variables describe the property not only as a physical asset, but also as a historic and architectural object.
        """)

    st.markdown("---")

    st.markdown("### Why This Matters")
    st.markdown("""
Real estate prices are often modeled using measurable structural variables. However, in Manhattan, many properties are valued not only for their size or location, but also for their architectural identity and historic character.

This project tests whether adding heritage-related variables improves prediction accuracy. A better-performing heritage-enhanced model would suggest that architectural and preservation features provide additional information that standard real-estate features alone may miss.
    """)

    st.info("""
**Important limitation:** This project is predictive, not causal.  
A higher-performing heritage-enhanced model does not prove that landmark status or architectural style directly causes higher sale prices. It only shows that heritage-related variables are useful signals for predicting sale prices within the dataset.
    """)

    st.markdown("---")

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
The datasets are merged using **BBL**, which stands for Borough-Block-Lot, New York City's property identifier.
The final dataset focuses on Manhattan properties with matching sales, PLUTO, and heritage-related records.
    """)

    cards([
        ("Final merged records", f"{len(df):,}"),
        ("Engineered features", f"{len(df.columns)}"),
        ("Unique architects", f"{df['architect'].nunique()}"),
        ("With GPS coordinates", f"{df.dropna(subset=['latitude']).shape[0]:,}"),
    ])

    st.markdown("---")

    st.markdown("### Data Preview")
    preview_cols = [
        "BBL",
        "sale_price",
        "gross_sqft",
        "num_floors",
        "building_age",
        "construction_era",
        "style_primary",
        "material_primary",
        "architect",
        "in_historic_district",
        "neighborhood",
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
# PAGE 2. VISUALIZATIONS & MAPS
# =================================================================
def page2():
    title(
        "Visualizations & Maps",
        "Exploring architectural heritage, location, and property value across Manhattan"
    )

    viz = df[
        (df["sale_price"] < df["sale_price"].quantile(0.97))
        & (df["price_per_sqft"] < df["price_per_sqft"].quantile(0.97))
        & df["latitude"].notna()
    ].copy()

    section = st.radio(
        "Section",
        [
            "Interactive Map",
            "Price Analysis",
            "Architectural Patterns",
            "Era & Correlation",
        ],
        horizontal=True,
    )

    if section == "Interactive Map":
        st.markdown("### 3D Property Map")
        st.caption("Each column represents a property. Height and color encode selected variables.")

        c1, c2, c3 = st.columns(3)

        with c1:
            color_var = st.selectbox(
                "Color by",
                [
                    "sale_price",
                    "price_per_sqft",
                    "building_age",
                    "architect_prestige_score",
                    "num_floors",
                    "assess_total",
                ],
                format_func=flabel,
            )

        with c2:
            height_var = st.selectbox(
                "Height by",
                [
                    "sale_price",
                    "gross_sqft",
                    "num_floors",
                    "building_age",
                    "price_per_sqft",
                ],
                format_func=flabel,
            )

        with c3:
            cmap_name = st.selectbox(
                "Color palette",
                ["magma", "inferno", "viridis", "plasma", "turbo", "cividis"],
            )

        map_d = viz.dropna(subset=[color_var, height_var, "latitude", "longitude"]).copy()

        h_vals = pd.to_numeric(map_d[height_var], errors="coerce")
        h_cap = h_vals.quantile(0.97)
        map_d["_h"] = (h_vals.clip(lower=0, upper=h_cap) / h_cap * 800).fillna(0)

        c_vals = pd.to_numeric(map_d[color_var], errors="coerce").fillna(map_d[color_var].median())
        norm = mcolors.Normalize(
            vmin=c_vals.quantile(0.02),
            vmax=c_vals.quantile(0.98),
        )
        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(norm(c_vals.values))

        map_d["_r"] = (rgba[:, 0] * 255).astype(int)
        map_d["_g"] = (rgba[:, 1] * 255).astype(int)
        map_d["_b"] = (rgba[:, 2] * 255).astype(int)

        map_d["price_fmt"] = map_d["sale_price"].apply(lambda v: f"${v:,.0f}")
        map_d["sqft_fmt"] = map_d["gross_sqft"].apply(
            lambda v: f"{v:,.0f} sqft" if pd.notna(v) else "unknown sqft"
        )
        map_d["arch_fmt"] = map_d["architect"].fillna("Unknown architect")
        map_d["style_fmt"] = map_d["style_primary"].fillna("")
        map_d["era_fmt"] = map_d["construction_era"].fillna("")

        pdk_style = "dark" if T["plotly"] == "plotly_dark" else "light"

        layer = pdk.Layer(
            "ColumnLayer",
            data=map_d,
            get_position=["longitude", "latitude"],
            get_elevation="_h",
            elevation_scale=1,
            radius=22,
            get_fill_color=["_r", "_g", "_b", 210],
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )

        view = pdk.ViewState(
            latitude=40.754,
            longitude=-73.987,
            zoom=12.2,
            pitch=50,
            bearing=20,
        )

        tooltip = {
            "html": (
                "<div style='font-family: Inter, sans-serif; padding: 8px 12px; min-width: 180px;'>"
                "<b style='font-size: 14px;'>{display_name}</b><br/>"
                "<span style='color:#9ca3af;'>{price_fmt}</span><br/>"
                "<span>{style_fmt}</span><br/>"
                "<span style='color:#9ca3af;'>by {arch_fmt}</span><br/>"
                "<span style='color:#9ca3af;'>{era_fmt} · {sqft_fmt}</span>"
                "</div>"
            ),
            "style": {
                "backgroundColor": "rgba(17,24,39,0.95)",
                "color": "white",
                "borderRadius": "10px",
                "boxShadow": "0 4px 18px rgba(0,0,0,0.3)",
            },
        }

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            map_provider="carto",
            map_style=pdk_style,
            tooltip=tooltip,
        )

        st.pydeck_chart(deck, use_container_width=True)

        c_min, c_max = c_vals.quantile(0.02), c_vals.quantile(0.98)

        def _fmt(v, varname):
            if "price" in varname or "assess" in varname:
                return f"${v:,.0f}"
            if "age" in varname:
                return f"{v:,.0f} yrs"
            return f"{v:,.0f}"

        stops = [cmap(i / 9) for i in range(10)]
        gradient_css = ", ".join(
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)}) {i*100/9:.0f}%"
            for i, (r, g, b, _) in enumerate(stops)
        )

        pretty_color = FRIENDLY_LABELS.get(color_var, color_var)
        pretty_height = FRIENDLY_LABELS.get(height_var, height_var)

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:14px; padding:10px 14px;
                        background:{T['card_bg']}; border:1px solid {T['card_border']};
                        border-radius:12px; margin-top:8px; font-family:Inter,sans-serif;">
              <div style="font-size:0.78rem; color:{T['muted']}; text-transform:uppercase; letter-spacing:0.05em;">
                Color = {pretty_color}
              </div>
              <div style="font-size:0.82rem; color:{T['text']}; min-width:64px; text-align:right;">
                {_fmt(c_min, color_var)}
              </div>
              <div style="flex:1; height:14px; border-radius:7px;
                          background: linear-gradient(to right, {gradient_css});
                          box-shadow: inset 0 0 0 1px {T['card_border']};"></div>
              <div style="font-size:0.82rem; color:{T['text']}; min-width:64px;">
                {_fmt(c_max, color_var)}
              </div>
              <div style="font-size:0.72rem; color:{T['muted']};">
                ↕ Height = {pretty_height}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Construction Era Map")
        st.caption("This map shows how different construction periods cluster geographically.")

        era_colors = {
            "Pre-1850": "#e63946",
            "1850–1899": "#f77f00",
            "1900–1919": "#fcbf49",
            "1920–1939 (Art Deco)": "#2ec4b6",
            "1940–1969 (Mid-Century)": "#457b9d",
            "1970+": "#a8dadc",
        }

        era_d = viz[viz["construction_era"].isin(era_colors.keys())]

        fig2 = px.scatter_mapbox(
            era_d,
            lat="latitude",
            lon="longitude",
            color="construction_era",
            color_discrete_map=era_colors,
            hover_name="display_name",
            hover_data={
                "sale_price": ":,.0f",
                "construction_era": True,
                "latitude": False,
                "longitude": False,
            },
            zoom=12,
            center={"lat": 40.754, "lon": -73.987},
            height=540,
            template=T["plotly"],
        )

        fig2.update_layout(
            mapbox_style=T["map_style"],
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif section == "Price Analysis":
        st.markdown("### Price Distributions")

        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(
                viz,
                x="sale_price",
                nbins=50,
                title="Sale Price Distribution",
                color_discrete_sequence=[T["accent"]],
                template=T["plotly"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                viz.dropna(subset=["price_per_sqft"]),
                x="price_per_sqft",
                nbins=50,
                title="Price per SqFt Distribution",
                color_discrete_sequence=["#e74c3c"],
                template=T["plotly"],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Neighborhood Price Ranking")

        hood = (
            viz.groupby("neighborhood")["sale_price"]
            .median()
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
        )

        fig = px.bar(
            hood,
            x="sale_price",
            y="neighborhood",
            orientation="h",
            color="sale_price",
            color_continuous_scale="Plasma",
            template=T["plotly"],
            height=550,
            labels={"sale_price": "Median Sale Price ($)", "neighborhood": ""},
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Price by Historic District")

        hd = viz[viz["historic_district"].astype(str).str.len() > 0]

        fig = px.box(
            hd,
            x="historic_district",
            y="price_per_sqft",
            template=T["plotly"],
            height=480,
            labels={"historic_district": "", "price_per_sqft": "$/SqFt"},
        )

        fig.update_xaxes(tickangle=35)
        st.plotly_chart(fig, use_container_width=True)

    elif section == "Architectural Patterns":
        st.markdown("### Architectural Style and Price")

        sp = (
            viz.groupby("style_primary")["price_per_sqft"]
            .agg(["median", "count"])
            .query("count >= 5")
            .sort_values("median", ascending=False)
            .head(20)
            .reset_index()
        )

        fig = px.bar(
            sp,
            x="median",
            y="style_primary",
            orientation="h",
            color="median",
            color_continuous_scale="Viridis",
            hover_data={"count": True},
            template=T["plotly"],
            height=560,
            labels={"median": "Median $/SqFt", "style_primary": ""},
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Facade Material")

            mp = (
                viz.groupby("material_primary")["price_per_sqft"]
                .agg(["median", "count"])
                .query("count >= 5")
                .sort_values("median", ascending=False)
                .head(12)
                .reset_index()
            )

            fig = px.bar(
                mp,
                x="median",
                y="material_primary",
                orientation="h",
                color="median",
                color_continuous_scale="Magma",
                template=T["plotly"],
                height=440,
                labels={"median": "$/SqFt", "material_primary": ""},
            )

            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Top Architects by Median Sale Price")

            ap = (
                viz.groupby("architect")["sale_price"]
                .agg(["median", "count"])
                .query("count >= 5")
                .sort_values("median", ascending=False)
                .head(12)
                .reset_index()
            )

            fig = px.bar(
                ap,
                x="median",
                y="architect",
                orientation="h",
                color="median",
                color_continuous_scale="Plasma",
                hover_data={"count": True},
                template=T["plotly"],
                height=440,
                labels={"median": "Median Sale $", "architect": ""},
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Architect Signal vs Price")

        scatter_d = viz.dropna(subset=["architect_prestige_score", "price_per_sqft"])

        fig = px.scatter(
            scatter_d,
            x="architect_prestige_score",
            y="price_per_sqft",
            color="construction_era",
            hover_name="architect",
            opacity=0.5,
            trendline="ols",
            template=T["plotly"],
            labels={
                "architect_prestige_score": "Architect Portfolio Score",
                "price_per_sqft": "$/SqFt",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Architect Value Leaderboard")
        st.caption("This table summarizes price patterns by architect. It is descriptive, not causal.")

        min_n = st.slider("Minimum buildings per architect", 3, 25, 5, key="arch_minN")

        market = viz[
            (viz["sale_price"] >= 100_000)
            & (viz["price_per_sqft"] >= 50)
        ].copy()

        city_med_psf = market["price_per_sqft"].median()

        g = (
            market.dropna(subset=["architect"])
            .groupby("architect")
            .agg(
                buildings=("BBL", "count"),
                median_price=("sale_price", "median"),
                median_psf=("price_per_sqft", "median"),
                landmark_share=("is_landmark", "mean"),
                prestige=("architect_prestige_score", "mean"),
                hist_dist_share=("in_historic_district", "mean"),
            )
            .reset_index()
        )

        g = g[g["buildings"] >= min_n].copy()
        g["price_position_pct"] = (g["median_psf"] / city_med_psf - 1) * 100
        g = g.sort_values("price_position_pct", ascending=False)
        g.insert(0, "Rank", range(1, len(g) + 1))

        leaderboard = g.rename(columns={
            "architect": "Architect",
            "buildings": "# Buildings",
            "median_price": "Median Sale Price",
            "median_psf": "Median $/SqFt",
            "price_position_pct": "Price Position %",
            "landmark_share": "% Landmarked",
            "hist_dist_share": "% in Hist. Dist.",
            "prestige": "Architect Signal",
        })

        st.dataframe(
            leaderboard.style.format({
                "Median Sale Price": "${:,.0f}",
                "Median $/SqFt": "${:,.0f}",
                "Price Position %": "{:+.1f}%",
                "% Landmarked": "{:.0%}",
                "% in Hist. Dist.": "{:.0%}",
                "Architect Signal": "{:.2f}",
            }).background_gradient(subset=["Price Position %"], cmap="RdYlGn"),
            use_container_width=True,
            height=460,
        )

    else:
        st.markdown("### Price by Construction Era")

        era_order = [
            "Pre-1850",
            "1850–1899",
            "1900–1919",
            "1920–1939 (Art Deco)",
            "1940–1969 (Mid-Century)",
            "1970+",
        ]

        era_d = viz[viz["construction_era"].isin(era_order)]

        fig = px.box(
            era_d,
            x="construction_era",
            y="price_per_sqft",
            category_orders={"construction_era": era_order},
            color="construction_era",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template=T["plotly"],
            height=450,
            labels={"construction_era": "", "price_per_sqft": "$/SqFt"},
        )

        fig.update_layout(showlegend=False, xaxis_tickangle=15)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Building Age vs Sale Price")

            fig = px.scatter(
                viz.dropna(subset=["building_age"]),
                x="building_age",
                y="sale_price",
                color="construction_era",
                opacity=0.4,
                trendline="ols",
                template=T["plotly"],
                labels={
                    "building_age": "Age",
                    "sale_price": "Sale Price ($)",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Altered vs Original Buildings")

            fig = px.violin(
                viz,
                x="is_altered",
                y="price_per_sqft",
                box=True,
                color="is_altered",
                template=T["plotly"],
                labels={
                    "is_altered": "Altered (1 = Yes)",
                    "price_per_sqft": "$/SqFt",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Correlation Matrix")

        corr_cols = [
            "sale_price",
            "price_per_sqft",
            "building_age",
            "num_floors",
            "gross_sqft",
            "assess_total",
            "architect_prestige_score",
            "rare_style_score",
            "is_altered",
            "in_historic_district",
        ]

        ac = [c for c in corr_cols if c in viz.columns]
        cm = viz[ac].corr()
        mask = np.triu(np.ones_like(cm, dtype=bool))

        fig_c, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(
            cm,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            vmin=-1,
            vmax=1,
        )

        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close()


# =================================================================
# PAGE 3. PREDICTION MODELS
# =================================================================
def page3():
    title(
        "Prediction Models",
        "Comparing a structural baseline model with a heritage-enhanced model"
    )

    st.markdown("""
This page directly tests the central research question.  
The baseline model uses only standard real-estate variables.  
The heritage-enhanced model adds architectural and preservation variables.

If the heritage-enhanced model has a higher test R² or lower dollar error, then heritage-related variables improve prediction.
    """)

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
    idx_all = np.arange(len(mdf))

    Xb_tr, Xb_te, y_tr, y_te, _, te_idx = train_test_split(
        X_b,
        y,
        idx_all,
        test_size=test_sz,
        random_state=rs,
    )

    Xa_tr, Xa_te, _, _ = train_test_split(
        X_a,
        y,
        test_size=test_sz,
        random_state=rs,
    )

    if do_scale:
        sc1 = StandardScaler()
        Xb_tr_s = sc1.fit_transform(Xb_tr)
        Xb_te_s = sc1.transform(Xb_te)

        sc2 = StandardScaler()
        Xa_tr_s = sc2.fit_transform(Xa_tr)
        Xa_te_s = sc2.transform(Xa_te)
    else:
        Xb_tr_s, Xb_te_s = Xb_tr, Xb_te
        Xa_tr_s, Xa_te_s = Xa_tr, Xa_te

    MODELS = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=1.0), True),
        "Lasso Regression": (Lasso(alpha=0.05), True),
        "Elastic Net": (ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=rs), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=10, random_state=rs), False),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=rs,
                n_jobs=-1,
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=rs,
            ),
            False,
        ),
    }

    if HAS_LGBM:
        MODELS["LightGBM"] = (
            lgb.LGBMRegressor(
                n_estimators=150,
                num_leaves=31,
                learning_rate=0.08,
                random_state=rs,
                n_jobs=-1,
                verbose=-1,
            ),
            False,
        )

    if HAS_CATBOOST:
        MODELS["CatBoost"] = (
            CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.08,
                random_seed=rs,
                verbose=0,
                allow_writing_files=False,
                thread_count=-1,
            ),
            False,
        )

    st.markdown("---")
    st.markdown("### Select models to train")

    sel = []
    names = list(MODELS.keys())
    per_row = 4

    for start in range(0, len(names), per_row):
        row = names[start:start + per_row]
        cols = st.columns(per_row)

        for nm, col in zip(row, cols):
            if col.checkbox(nm, True, key=f"m_{nm}"):
                sel.append(nm)

    if not sel:
        st.warning("Select at least one model.")
        return

    base_rows = []
    herit_rows = []
    trained_h = {}
    timings = {}

    import random
    facts_pool = random.sample(MANHATTAN_FACTS, k=min(len(sel), len(MANHATTAN_FACTS)))

    progress = st.progress(0, text="Starting...")
    status = st.empty()
    fact_box = st.empty()
    leaderboard_box = st.empty()
    total_t0 = time.perf_counter()

    for i, nm in enumerate(sel, start=1):
        status.markdown(
            f"&nbsp;&nbsp;**Training {i}/{len(sel)}: {nm}**"
            f" &middot; <span style='color:#6E6E73'>baseline + heritage-enhanced</span>",
            unsafe_allow_html=True,
        )

        fact_box.info(facts_pool[(i - 1) % len(facts_pool)])
        t0 = time.perf_counter()

        proto, needs_scale = MODELS[nm]
        mb = copy.deepcopy(proto)
        mh = copy.deepcopy(proto)

        if needs_scale:
            rb = train_eval(mb, Xb_tr_s, y_tr, Xb_te_s, y_te)
            rh = train_eval(mh, Xa_tr_s, y_tr, Xa_te_s, y_te)
        else:
            rb = train_eval(mb, Xb_tr, y_tr, Xb_te, y_te)
            rh = train_eval(mh, Xa_tr, y_tr, Xa_te, y_te)

        elapsed = time.perf_counter() - t0
        timings[nm] = elapsed

        base_rows.append({
            "Model": nm,
            **{k: v for k, v in rb.items() if k not in ("preds", "model")},
            "Time(s)": round(elapsed, 2),
        })

        herit_rows.append({
            "Model": nm,
            **{k: v for k, v in rh.items() if k not in ("preds", "model")},
            "Time(s)": round(elapsed, 2),
        })

        herit_rows[-1]["preds"] = rh["preds"]
        trained_h[nm] = mh

        progress.progress(
            i / len(sel),
            text=f"Done {i}/{len(sel)} · {nm} took {elapsed:.1f}s"
        )

        live_lb = (
            pd.DataFrame(herit_rows)[["Model", "R2_test"]]
            .sort_values("R2_test", ascending=True)
        )

        is_winner = live_lb["R2_test"] == live_lb["R2_test"].max()
        bar_colors = [T["accent"] if w else "#94a3b8" for w in is_winner]

        live_fig = go.Figure(go.Bar(
            x=live_lb["R2_test"],
            y=live_lb["Model"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in live_lb["R2_test"]],
            textposition="outside",
        ))

        live_fig.update_layout(
            template=T["plotly"],
            height=max(220, 38 * len(live_lb) + 80),
            margin=dict(l=10, r=30, t=30, b=10),
            xaxis_title="R² Test Score",
            title=dict(
                text=f"🏆 Live Leaderboard — leader: {live_lb.iloc[-1]['Model']}",
                x=0.02,
                font=dict(size=14),
            ),
            xaxis=dict(range=[0, max(0.7, live_lb["R2_test"].max() * 1.1)]),
        )

        leaderboard_box.plotly_chart(
            live_fig,
            use_container_width=True,
            key=f"live_lb_{i}",
        )

    total_elapsed = time.perf_counter() - total_t0

    progress.empty()
    fact_box.empty()

    status.success(
        f"✅ Trained {len(sel)} model{'s' if len(sel) > 1 else ''} in {total_elapsed:.1f}s "
        f"(slowest: {max(timings, key=timings.get)} at {max(timings.values()):.1f}s)"
    )

    st.balloons()

    dfb = pd.DataFrame(base_rows).set_index("Model")
    dfh = pd.DataFrame(herit_rows).set_index("Model")

    st.session_state["trained_h"] = trained_h
    st.session_state["feat_all"] = FALL
    st.session_state["Xa_tr"] = Xa_tr
    st.session_state["Xa_te"] = Xa_te
    st.session_state["y_tr"] = y_tr
    st.session_state["y_te"] = y_te
    st.session_state["te_idx"] = te_idx

    st.markdown("---")
    st.markdown("### Results: Baseline vs Heritage-Enhanced")

    c1, c2 = st.columns(2)

    fmt_cols = [
        "R2_train",
        "R2_test",
        "MAE_dollars",
        "RMSE_dollars",
        "Time(s)",
    ]

    with c1:
        st.markdown("**Baseline Model — structural variables only**")
        st.dataframe(
            dfb[fmt_cols].style.format({
                "R2_train": "{:.4f}",
                "R2_test": "{:.4f}",
                "MAE_dollars": "${:,.0f}",
                "RMSE_dollars": "${:,.0f}",
                "Time(s)": "{:.2f}",
            }),
            use_container_width=True,
        )

    with c2:
        st.markdown("**Heritage-Enhanced Model — baseline + heritage variables**")
        st.dataframe(
            dfh[fmt_cols].style.format({
                "R2_train": "{:.4f}",
                "R2_test": "{:.4f}",
                "MAE_dollars": "${:,.0f}",
                "RMSE_dollars": "${:,.0f}",
                "Time(s)": "{:.2f}",
            }),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Heritage Feature Uplift")
    st.caption("This chart shows whether adding architectural and preservation variables improves test performance.")

    up = pd.DataFrame({
        "Model": sel,
        "Baseline R²": [dfb.loc[n, "R2_test"] for n in sel],
        "Heritage-Enhanced R²": [dfh.loc[n, "R2_test"] for n in sel],
    })

    up["Uplift"] = up["Heritage-Enhanced R²"] - up["Baseline R²"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Baseline",
        x=up["Model"],
        y=up["Baseline R²"],
        marker_color="steelblue",
    ))

    fig.add_trace(go.Bar(
        name="Heritage-Enhanced",
        x=up["Model"],
        y=up["Heritage-Enhanced R²"],
        marker_color=T["accent"],
    ))

    fig.update_layout(
        barmode="group",
        template=T["plotly"],
        height=400,
        yaxis_title="R² Test Score",
    )

    st.plotly_chart(fig, use_container_width=True)

    best = up.loc[up["Heritage-Enhanced R²"].idxmax()]

    st.success(
        f"Best heritage-enhanced model: **{best['Model']}**. "
        f"R² = {best['Heritage-Enhanced R²']:.4f}. "
        f"Uplift over baseline = {best['Uplift']:+.4f}."
    )

    st.markdown("---")
    st.markdown("### Actual vs Predicted")

    pick = st.selectbox("Model", sel, key="avp")
    yp = dfh.loc[pick, "preds"]

    fig = px.scatter(
        x=y_te,
        y=yp,
        opacity=0.4,
        template=T["plotly"],
        labels={
            "x": "Actual log price",
            "y": "Predicted log price",
        },
    )

    fig.add_shape(
        type="line",
        x0=y_te.min(),
        x1=y_te.max(),
        y0=y_te.min(),
        y1=y_te.max(),
        line=dict(color="red", dash="dash"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Where the Model Disagrees with the Market")
    st.caption("""
Residuals compare actual price with predicted price. Positive residuals mean the market paid more than the model expected. Negative residuals mean the market paid less than the model expected.
    """)

    test_rows = mdf.iloc[te_idx].copy().reset_index(drop=True)
    test_rows["actual_price"] = np.expm1(y_te)
    test_rows["pred_price"] = np.expm1(yp)
    test_rows["residual_log"] = y_te - yp
    test_rows["residual_pct"] = (
        test_rows["actual_price"] / test_rows["pred_price"] - 1
    ) * 100
    test_rows["abs_dollar_diff"] = (
        test_rows["actual_price"] - test_rows["pred_price"]
    )

    map_d = test_rows.dropna(subset=["latitude", "longitude"]).copy()

    diverging = [
        [0.0, "#dc2626"],
        [0.25, "#f87171"],
        [0.5, "#fef3c7"],
        [0.75, "#4ade80"],
        [1.0, "#15803d"],
    ]

    fig_map = px.scatter_mapbox(
        map_d,
        lat="latitude",
        lon="longitude",
        color="residual_pct",
        color_continuous_scale=diverging,
        color_continuous_midpoint=0,
        range_color=[-80, 80],
        size=map_d["residual_pct"].abs().clip(5, 150),
        size_max=24,
        hover_name="display_name",
        hover_data={
            "actual_price": ":$,.0f",
            "pred_price": ":$,.0f",
            "residual_pct": ":+.1f",
            "architect": True,
            "construction_era": True,
            "latitude": False,
            "longitude": False,
        },
        zoom=11.5,
        height=520,
        template=T["plotly"],
        mapbox_style=T["map_style"],
        labels={"residual_pct": "Residual %"},
    )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Residual %",
            ticksuffix="%",
            tickvals=[-80, -40, 0, 40, 80],
            ticktext=[
                "−80%<br>Below model",
                "−40%",
                "0%<br>On model",
                "+40%",
                "+80%<br>Above model",
            ],
        ),
    )

    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### Biggest Model Misses")

    show_cols = [
        "display_name",
        "architect",
        "construction_era",
        "actual_price",
        "pred_price",
        "residual_pct",
        "neighborhood",
    ]

    rename = {
        "display_name": "Property",
        "architect": "Architect",
        "construction_era": "Era",
        "actual_price": "Actual",
        "pred_price": "Predicted",
        "residual_pct": "Diff %",
        "neighborhood": "Neighborhood",
    }

    fmt = {
        "Actual": "${:,.0f}",
        "Predicted": "${:,.0f}",
        "Diff %": "{:+.1f}%",
    }

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Top 10 Above Model Prediction**")
        top_above = (
            test_rows.sort_values("residual_pct", ascending=False)
            .head(10)[show_cols]
            .rename(columns=rename)
        )

        st.dataframe(
            top_above.style.format(fmt),
            use_container_width=True,
            height=380,
        )

    with c2:
        st.markdown("**Top 10 Below Model Prediction**")
        top_below = (
            test_rows.sort_values("residual_pct", ascending=True)
            .head(10)[show_cols]
            .rename(columns=rename)
        )

        st.dataframe(
            top_below.style.format(fmt),
            use_container_width=True,
            height=380,
        )


# =================================================================
# PAGE 4. FEATURE IMPORTANCE
# =================================================================
def page4():
    title(
        "Feature Importance & Explainability",
        "Identifying which variables help the heritage-enhanced model predict sale price"
    )

    if "trained_h" not in st.session_state:
        st.info("Please train models on the Prediction Models page first, then return here.")
        return

    trained = st.session_state["trained_h"]
    fnames = st.session_state["feat_all"]
    Xa_te = st.session_state["Xa_te"]
    Xa_tr = st.session_state["Xa_tr"]

    sel = st.selectbox("Model", list(trained.keys()))
    model = trained[sel]

    section = st.radio(
        "View",
        ["Feature Importance", "SHAP Analysis"],
        horizontal=True,
    )

    if section == "Feature Importance":
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_)
        else:
            st.warning("This model does not expose importance values.")
            return

        fi = pd.DataFrame({
            "Feature": [flabel(f) for f in fnames],
            "Importance": imp,
            "Type": ["Baseline" if f in FB else "Heritage" for f in fnames],
        }).sort_values("Importance", ascending=True)

        fig = px.bar(
            fi,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Type",
            color_discrete_map={
                "Baseline": "steelblue",
                "Heritage": T["accent"],
            },
            template=T["plotly"],
            height=550,
            labels={"Importance": "Importance", "Feature": ""},
        )

        st.plotly_chart(fig, use_container_width=True)

        total = fi["Importance"].sum()
        h_share = fi[fi["Type"] == "Heritage"]["Importance"].sum()

        if total > 0:
            st.info(
                f"Heritage-related features account for **{h_share / total * 100:.1f}%** "
                f"of this model's total importance."
            )

        if hasattr(model, "coef_"):
            st.markdown("---")
            st.markdown("### Coefficient Direction")

            coef = pd.DataFrame({
                "Feature": [flabel(f) for f in fnames],
                "Coefficient": model.coef_,
                "Type": ["Baseline" if f in FB else "Heritage" for f in fnames],
            }).sort_values("Coefficient", key=abs, ascending=True)

            fig = px.bar(
                coef,
                x="Coefficient",
                y="Feature",
                orientation="h",
                color="Type",
                color_discrete_map={
                    "Baseline": "steelblue",
                    "Heritage": T["accent"],
                },
                template=T["plotly"],
                height=550,
            )

            fig.add_vline(x=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

    else:
        try:
            import shap

            with st.spinner("Computing SHAP values..."):
                if hasattr(model, "feature_importances_"):
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(Xa_te[:250])
                else:
                    explainer = shap.LinearExplainer(model, Xa_tr[:500])
                    sv = explainer.shap_values(Xa_te[:250])

            st.markdown("### SHAP Summary Plot")

            fig_s, ax = plt.subplots(figsize=(11, 7))
            shap.summary_plot(
                sv,
                Xa_te[:250],
                feature_names=[flabel(f) for f in fnames],
                show=False,
                max_display=15,
            )

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

            explanation = shap.Explanation(
                values=sv[idx],
                base_values=ev,
                data=Xa_te[idx],
                feature_names=[flabel(f) for f in fnames],
            )

            shap.waterfall_plot(explanation, show=False, max_display=12)
            st.pyplot(fig_w)
            plt.close()

        except ImportError:
            st.error("SHAP is not installed. Run `pip install shap` and restart the app.")
        except Exception as e:
            st.error(f"SHAP error: {e}")
            st.info("Try selecting a tree-based model for more reliable SHAP analysis.")


# =================================================================
# PAGE 5. HYPERPARAMETER TUNING
# =================================================================
def page5():
    title(
        "Hyperparameter Tuning",
        "Testing different model settings to improve prediction performance"
    )

    with st.expander("📚 What is hyperparameter tuning?", expanded=False):
        st.markdown("""
Hyperparameters are the model settings chosen before training.  
For example, a random forest needs to know how many trees to use, and a decision tree needs to know how deep it can grow.

This page tries different combinations of hyperparameters and compares their performance.
The goal is to find a stronger version of the heritage-enhanced prediction model.
        """)

    X = mdf[FALL].values
    y = mdf["log_price"].values

    tune_choices = [
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
    ]

    if HAS_LGBM:
        tune_choices.append("LightGBM")

    c1, c2 = st.columns(2)

    with c1:
        model_name = st.selectbox("Model", tune_choices)

    with c2:
        test_sz = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="hp_ts")

    st.markdown("---")
    st.markdown("### Hyperparameter Grid")

    grid = []

    if model_name == "Ridge Regression":
        alphas = st.multiselect(
            "Alpha",
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            default=[0.01, 0.1, 1.0, 10.0, 100.0],
        )
        grid = [{"alpha": a} for a in alphas]

    elif model_name == "Lasso Regression":
        alphas = st.multiselect(
            "Alpha",
            [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            default=[0.001, 0.01, 0.1, 0.5],
        )
        grid = [{"alpha": a} for a in alphas]

    elif model_name == "Elastic Net":
        alphas = st.multiselect(
            "Alpha",
            [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            default=[0.01, 0.05, 0.1],
        )

        l1s = st.multiselect(
            "L1 ratio",
            [0.1, 0.3, 0.5, 0.7, 0.9],
            default=[0.3, 0.5, 0.7],
        )

        grid = [{"alpha": a, "l1_ratio": l} for a in alphas for l in l1s]

    elif model_name == "LightGBM":
        nes = st.multiselect(
            "N Estimators",
            [100, 200, 400, 800],
            default=[200, 400],
        )

        leaves = st.multiselect(
            "Num Leaves",
            [15, 31, 63, 127],
            default=[31, 63],
        )

        lrs = st.multiselect(
            "Learning Rate",
            [0.01, 0.05, 0.1],
            default=[0.05, 0.1],
        )

        grid = [
            {
                "n_estimators": n,
                "num_leaves": lv,
                "learning_rate": lr,
            }
            for n in nes
            for lv in leaves
            for lr in lrs
        ]

    elif model_name == "Decision Tree":
        depths = st.multiselect(
            "Max Depth",
            [2, 3, 5, 7, 10, 15, None],
            default=[3, 5, 10],
        )

        mins = st.multiselect(
            "Min Samples Split",
            [2, 5, 10, 20],
            default=[2, 5, 10],
        )

        grid = [{"max_depth": d, "min_samples_split": s} for d in depths for s in mins]

    elif model_name == "Random Forest":
        nes = st.multiselect(
            "N Estimators",
            [50, 100, 200, 300],
            default=[50, 100, 200],
        )

        deps = st.multiselect(
            "Max Depth",
            [5, 10, 15, None],
            default=[5, 10, 15],
        )

        grid = [{"n_estimators": n, "max_depth": d} for n in nes for d in deps]

    else:
        lrs = st.multiselect(
            "Learning Rate",
            [0.01, 0.05, 0.1, 0.2],
            default=[0.01, 0.1],
        )

        nes = st.multiselect(
            "N Estimators",
            [50, 100, 200],
            default=[50, 100],
        )

        deps = st.multiselect(
            "Max Depth",
            [3, 5, 7],
            default=[3, 5],
        )

        grid = [
            {
                "learning_rate": lr,
                "n_estimators": n,
                "max_depth": d,
            }
            for lr in lrs
            for n in nes
            for d in deps
        ]

    st.caption(f"Total experiments: **{len(grid)}**")

    st.markdown("---")
    st.markdown("### Weights & Biases Logging")

    use_wb = st.checkbox("Log experiments to W&B", False)
    wb_proj = "manhattan-heritage-pricing"
    wb_entity = ""
    wb_key = ""

    if use_wb:
        wc1, wc2 = st.columns(2)

        with wc1:
            wb_proj = st.text_input("Project name", "manhattan-heritage-pricing")

        with wc2:
            wb_entity = st.text_input("Entity / team optional", "")

        wb_key = st.text_input(
            "API key optional",
            "",
            type="password",
            help="Leave empty if you already ran wandb login.",
        )

    st.markdown("---")

    if st.button("Run Tuning", type="primary"):
        if not grid:
            st.error("Configure at least one hyperparameter combination.")
            return

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=test_sz,
            random_state=42,
        )

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        needs_scale = model_name in [
            "Ridge Regression",
            "Lasso Regression",
            "Elastic Net",
        ]

        wb_ok = False
        wandb_mod = None

        if use_wb:
            try:
                import wandb as wandb_mod

                if wb_key.strip():
                    wandb_mod.login(key=wb_key.strip(), relogin=True)

                api_key_present = bool(wandb_mod.api.api_key)

                if not api_key_present:
                    st.error(
                        "W&B has no API key in this session. Run wandb login or paste your key."
                    )
                else:
                    wb_ok = True

            except ImportError:
                st.warning("wandb not installed. Logging locally only.")
            except Exception as e:
                st.error(f"W&B login failed: {e}")

        def short_tag(p):
            bits = []

            for k, v in p.items():
                if v is None:
                    v = "auto"

                if isinstance(v, float):
                    bits.append(f"{k[:3]}{v:g}")
                else:
                    bits.append(f"{k[:3]}{v}")

            return "_".join(bits)

        ts = pd.Timestamp.utcnow().strftime("%m%d-%H%M")
        model_slug = model_name.lower().replace(" ", "")

        results = []
        prog = st.progress(0)
        wb_errors = 0

        for i, params in enumerate(grid):
            prog.progress((i + 1) / len(grid))

            if model_name == "Ridge Regression":
                m = Ridge(**params)
            elif model_name == "Lasso Regression":
                m = Lasso(**params)
            elif model_name == "Elastic Net":
                m = ElasticNet(**params, random_state=42)
            elif model_name == "Decision Tree":
                m = DecisionTreeRegressor(**params, random_state=42)
            elif model_name == "Random Forest":
                m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            elif model_name == "Gradient Boosting":
                m = GradientBoostingRegressor(**params, random_state=42)
            else:
                m = lgb.LGBMRegressor(
                    **params,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )

            if needs_scale:
                m.fit(Xtr_s, ytr)
                ptr = m.predict(Xtr_s)
                pte = m.predict(Xte_s)
            else:
                m.fit(Xtr, ytr)
                ptr = m.predict(Xtr)
                pte = m.predict(Xte)

            actual_dollars = np.expm1(yte)
            pred_dollars = np.expm1(pte)

            metrics = {
                "r2_train": r2_score(ytr, ptr),
                "r2_test": r2_score(yte, pte),
                "mae_dollars": mean_absolute_error(actual_dollars, pred_dollars),
                "rmse_dollars": np.sqrt(mean_squared_error(actual_dollars, pred_dollars)),
            }

            results.append({**params, **metrics})

            if wb_ok:
                try:
                    init_kwargs = dict(
                        project=wb_proj,
                        name=f"{model_slug}_{short_tag(params)}_{ts}",
                        group=model_name,
                        tags=[model_slug, "heritage-features", "log-target"],
                        config={
                            "model": model_name,
                            "test_size": test_sz,
                            **params,
                        },
                        reinit=True,
                    )

                    if wb_entity.strip():
                        init_kwargs["entity"] = wb_entity.strip()

                    run = wandb_mod.init(**init_kwargs)
                    run.log(metrics)
                    run.finish()

                except Exception as e:
                    wb_errors += 1

                    if wb_errors <= 2:
                        st.warning(f"W&B logging failed for run {i}: {e}")

        prog.empty()

        res_df = pd.DataFrame(results).sort_values("r2_test", ascending=False)

        st.markdown("### Results")

        st.dataframe(
            res_df.style
            .highlight_max(subset=["r2_test"], color="#667eea")
            .highlight_min(subset=["rmse_dollars", "mae_dollars"], color="#2ecc71")
            .format({
                "r2_train": "{:.4f}",
                "r2_test": "{:.4f}",
                "mae_dollars": "${:,.0f}",
                "rmse_dollars": "${:,.0f}",
            }),
            use_container_width=True,
        )

        best = res_df.iloc[0]

        pcols = [
            c for c in res_df.columns
            if c not in ["r2_train", "r2_test", "mae_dollars", "rmse_dollars"]
        ]

        st.success(
            f"Best: R² = {best['r2_test']:.4f}, "
            f"MAE = ${best['mae_dollars']:,.0f}, "
            f"RMSE = ${best['rmse_dollars']:,.0f}. "
            f"Params: {dict(best[pcols])}"
        )

        if pcols:
            p0 = pcols[0]
            is_num = pd.api.types.is_numeric_dtype(res_df[p0])

            if is_num:
                fig = px.line(
                    res_df.sort_values(p0),
                    x=p0,
                    y=["r2_train", "r2_test"],
                    markers=True,
                    template=T["plotly"],
                    title=f"R² vs {p0}",
                )
            else:
                fig = px.bar(
                    res_df,
                    x=p0,
                    y=["r2_train", "r2_test"],
                    barmode="group",
                    template=T["plotly"],
                    title=f"R² vs {p0}",
                )

            st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            res_df,
            x="r2_train",
            y="r2_test",
            color="rmse_dollars",
            color_continuous_scale="RdYlGn_r",
            template=T["plotly"],
            title="Train vs Test R²",
        )

        fig2.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            line=dict(dash="dash", color="gray"),
        )

        st.plotly_chart(fig2, use_container_width=True)

        if wb_ok and use_wb:
            ent_part = f"{wb_entity.strip()}/" if wb_entity.strip() else ""
            url = f"https://wandb.ai/{ent_part}{wb_proj}"
            st.success(f"Logged {len(grid) - wb_errors}/{len(grid)} runs. [Open dashboard ↗]({url})")


# =================================================================
# PAGE 6. PROPERTY VALUATOR
# =================================================================
@st.cache_resource
def get_valuation_models():
    X_b = mdf[FB].values
    X_a = mdf[FALL].values
    y = mdf["log_price"].values

    Xb_tr, _, y_tr, _ = train_test_split(
        X_b,
        y,
        test_size=0.2,
        random_state=42,
    )

    Xa_tr, _, _, _ = train_test_split(
        X_a,
        y,
        test_size=0.2,
        random_state=42,
    )

    mb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42,
    ).fit(Xb_tr, y_tr)

    mh = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42,
    ).fit(Xa_tr, y_tr)

    return mb, mh, Xa_tr


@st.cache_resource
def get_quantile_models():
    if not HAS_LGBM:
        return None

    X_a = mdf[FALL].values
    y = mdf["log_price"].values

    Xa_tr, _, y_tr, _ = train_test_split(
        X_a,
        y,
        test_size=0.2,
        random_state=42,
    )

    out = {}

    for q in (0.1, 0.5, 0.9):
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=400,
            num_leaves=31,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ).fit(Xa_tr, y_tr)

        out[q] = m

    return out


def page6():
    title(
        "Property Valuator",
        "Compare a structural prediction with a heritage-enhanced prediction for one property"
    )

    with st.spinner("Warming up valuation models..."):
        mb, mh, bg_ref = get_valuation_models()

    pool = mdf.dropna(subset=FALL).copy()

    st.markdown("### Find a property")

    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        query = st.text_input(
            "Search address, building name, or architect",
            "",
        ).strip().lower()

    with c2:
        only_landmarks = st.checkbox("Landmarks only", False)

    with c3:
        in_hd = st.checkbox("Historic district only", False)

    subset = pool.copy()

    if only_landmarks:
        subset = subset[subset["is_landmark"] == 1]

    if in_hd:
        subset = subset[subset["in_historic_district"] == 1]

    if query:
        m = (
            subset["address"].astype(str).str.lower().str.contains(query, na=False)
            | subset["building_name"].astype(str).str.lower().str.contains(query, na=False)
            | subset["architect"].astype(str).str.lower().str.contains(query, na=False)
        )

        subset = subset[m]

    subset = subset.sort_values("sale_price", ascending=False)

    if len(subset) == 0:
        st.info("No matches. Try a different search.")
        return

    st.caption(f"{len(subset):,} matching properties. Showing top 200 by price.")

    def label(i):
        r = subset.loc[i]
        nm = r.get("display_name") or r.get("address", "Unknown")
        arch = r["architect"] if pd.notna(r.get("architect")) else "Unknown architect"
        return f"{nm} · {arch} · ${r['sale_price']:,.0f}"

    pick_idx = st.selectbox(
        "Pick a property",
        subset.index[:200].tolist(),
        format_func=label,
    )

    row = subset.loc[pick_idx]

    x_base = row[FB].values.reshape(1, -1).astype(float)
    x_all = row[FALL].values.reshape(1, -1).astype(float)

    pred_base = float(np.expm1(mb.predict(x_base)[0]))
    pred_her = float(np.expm1(mh.predict(x_all)[0]))
    actual = float(row["sale_price"])

    heritage_adjustment = pred_her - pred_base
    heritage_adjustment_pct = (
        heritage_adjustment / pred_base * 100 if pred_base else 0
    )

    gap = actual - pred_her
    gap_pct = gap / pred_her * 100 if pred_her else 0

    st.markdown("---")

    adjustment_color = T["accent"] if heritage_adjustment >= 0 else "#ef4444"

    cols = st.columns(4)

    cols[0].markdown(
        f'<div class="card"><p class="val">${actual:,.0f}</p>'
        f'<p class="lbl">Actual sale price</p></div>',
        unsafe_allow_html=True,
    )

    cols[1].markdown(
        f'<div class="card"><p class="val">${pred_base:,.0f}</p>'
        f'<p class="lbl">Baseline prediction</p></div>',
        unsafe_allow_html=True,
    )

    cols[2].markdown(
        f'<div class="card"><p class="val">${pred_her:,.0f}</p>'
        f'<p class="lbl">Heritage-enhanced prediction</p></div>',
        unsafe_allow_html=True,
    )

    cols[3].markdown(
        f'<div class="card"><p class="val" style="color:{adjustment_color}">'
        f'{heritage_adjustment:+,.0f}</p>'
        f'<p class="lbl">Heritage feature adjustment ({heritage_adjustment_pct:+.1f}%)</p></div>',
        unsafe_allow_html=True,
    )

    if abs(gap_pct) < 10:
        st.success(
            f"Actual price is close to the heritage-enhanced prediction "
            f"({gap_pct:+.1f}%)."
        )
    elif gap_pct > 0:
        st.info(
            f"Actual sale price is **above** the heritage-enhanced prediction by "
            f"{gap_pct:+.1f}% (${abs(gap):,.0f})."
        )
    else:
        st.warning(
            f"Actual sale price is **below** the heritage-enhanced prediction by "
            f"{gap_pct:+.1f}% (${abs(gap):,.0f})."
        )

    qm = get_quantile_models()

    if qm is not None:
        q_low = float(np.expm1(qm[0.1].predict(x_all)[0]))
        q_med = float(np.expm1(qm[0.5].predict(x_all)[0]))
        q_high = float(np.expm1(qm[0.9].predict(x_all)[0]))

        inside = q_low <= actual <= q_high

        st.markdown("---")
        st.markdown("### Prediction Interval")
        st.caption("This band shows the 10th, 50th, and 90th percentile predictions.")

        actual_color = "#10b981" if inside else "#ef4444"

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[q_low, q_high],
            y=[0, 0],
            mode="lines",
            line=dict(color=T["accent"], width=18),
            opacity=0.35,
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=[q_low, q_high],
            y=[0, 0],
            mode="markers+text",
            marker=dict(size=10, color=T["accent"]),
            text=[
                f"P10<br>${q_low:,.0f}",
                f"P90<br>${q_high:,.0f}",
            ],
            textposition=["top left", "top right"],
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=[q_med],
            y=[0],
            mode="markers+text",
            marker=dict(size=20, color=T["accent"], line=dict(color="white", width=2)),
            text=[f"Model median<br>${q_med:,.0f}"],
            textposition="bottom center",
            name="Median",
        ))

        fig.add_trace(go.Scatter(
            x=[actual],
            y=[0],
            mode="markers+text",
            marker=dict(
                size=22,
                color=actual_color,
                symbol="diamond",
                line=dict(color="white", width=2),
            ),
            text=[f"Actual<br>${actual:,.0f}"],
            textposition="bottom center",
            name="Actual",
        ))

        pad = max(q_high - q_low, 1) * 0.15

        fig.update_layout(
            template=T["plotly"],
            height=230,
            xaxis=dict(
                tickformat="$,.0f",
                range=[
                    min(q_low, actual) - pad,
                    max(q_high, actual) + pad,
                ],
            ),
            yaxis=dict(visible=False, range=[-1, 1]),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        if inside:
            st.caption("Actual price sits inside the 80% prediction band.")
        elif actual > q_high:
            st.caption(f"Actual price is above the P90 by ${actual - q_high:,.0f}.")
        else:
            st.caption(f"Actual price is below the P10 by ${q_low - actual:,.0f}.")

    bname = str(row.get("building_name") or "").strip()
    arch = str(row.get("architect") or "").strip()

    wiki = None

    if bname and bname.lower() not in ("0", "none", "nan"):
        wiki = wiki_lookup(bname)

    if (not wiki or not wiki.get("image")) and arch and arch.lower() not in ("unknown", "0", "nan"):
        arch_wiki = wiki_lookup(arch)

        if arch_wiki and arch_wiki.get("image"):
            wiki = arch_wiki

    if wiki and (wiki.get("image") or wiki.get("extract")):
        st.markdown("---")
        st.markdown("### Heritage Spotlight")

        wc1, wc2 = st.columns([1, 2])

        with wc1:
            if wiki.get("image"):
                st.image(wiki["image"], use_container_width=True)
            else:
                st.caption("No photo found on Wikipedia.")

        with wc2:
            st.markdown(f"**{wiki['title']}** · [Open on Wikipedia ↗]({wiki['url']})")

            if wiki.get("extract"):
                st.markdown(
                    f"<p style='color:{T['muted']};line-height:1.55;'>{wiki['extract']}</p>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("### Building Profile")

    pc1, pc2 = st.columns([1, 1])

    with pc1:
        def fmt(v, kind="str"):
            if pd.isna(v) or v == "" or v == 0:
                return "Unknown"
            if kind == "int":
                return f"{int(v):,}"
            if kind == "year":
                return f"{int(v)}"
            if kind == "sqft":
                return f"{v:,.0f}"
            return str(v)

        info = [
            ("Address", fmt(row.get("address"))),
            ("Building name", fmt(row.get("building_name"))),
            ("Architect", fmt(row.get("architect"))),
            ("Style", fmt(row.get("style_primary"))),
            ("Facade material", fmt(row.get("material_primary"))),
            ("Era", fmt(row.get("construction_era"))),
            ("Year built", fmt(row.get("construction_year"), "year")),
            ("Floors", fmt(row.get("num_floors"), "int")),
            ("Gross sqft", fmt(row.get("gross_sqft"), "sqft")),
            ("Neighborhood", fmt(row.get("neighborhood"))),
            ("Landmark", "Yes" if row.get("is_landmark") == 1 else "No"),
            ("Historic district", "Yes" if row.get("in_historic_district") == 1 else "No"),
            (
                "Altered since",
                fmt(row.get("alteration_year"), "year")
                if row.get("is_altered") == 1
                else "Original",
            ),
        ]

        info_df = pd.DataFrame(info, columns=["Field", "Value"])

        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True,
            height=480,
        )

    with pc2:
        near = pool[
            (pool["latitude"].between(row["latitude"] - 0.01, row["latitude"] + 0.01))
            & (pool["longitude"].between(row["longitude"] - 0.012, row["longitude"] + 0.012))
        ].copy()

        fig = px.scatter_mapbox(
            near,
            lat="latitude",
            lon="longitude",
            opacity=0.35,
            color_discrete_sequence=[T["muted"]],
            hover_name="display_name",
            zoom=14.5,
            center={
                "lat": row["latitude"],
                "lon": row["longitude"],
            },
            height=480,
            template=T["plotly"],
        )

        fig.add_trace(go.Scattermapbox(
            lat=[row["latitude"]],
            lon=[row["longitude"]],
            mode="markers",
            marker=dict(size=24, color=T["accent"]),
            hovertext=[row.get("display_name", "This property")],
            name="Selected",
            showlegend=False,
        ))

        fig.update_layout(
            mapbox_style=T["map_style"],
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Why This Price? Feature Contributions")
    st.caption("This explains how different features push the heritage-enhanced prediction up or down.")

    try:
        import shap

        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(mh)
            sv = explainer.shap_values(x_all)
            ev = explainer.expected_value

            if hasattr(ev, "__len__"):
                ev = ev[0]

        explanation = shap.Explanation(
            values=np.asarray(sv[0]),
            base_values=float(ev),
            data=x_all[0],
            feature_names=[flabel(f) for f in FALL],
        )

        fig_w, _ = plt.subplots(figsize=(11, 6))
        shap.waterfall_plot(explanation, show=False, max_display=12)
        plt.tight_layout()

        st.pyplot(fig_w)
        plt.close()

        base_dollars = float(np.expm1(ev))

        st.caption(
            f"Model baseline for a typical property: approximately ${base_dollars:,.0f}. "
            f"Feature contributions move the prediction to ${pred_her:,.0f}."
        )

    except ImportError:
        st.error("SHAP is not installed. Run `pip install shap`.")
    except Exception as e:
        st.warning(f"SHAP unavailable for this sample: {e}")


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────
if "1" in page:
    page1()
elif "2" in page:
    page2()
elif "3" in page:
    page3()
elif "4" in page:
    page4()
elif "5" in page:
    page5()
elif "6" in page:
    page6()
