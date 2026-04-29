"""
Manhattan Heritage Property Valuation
Predicting Manhattan Property Sale Prices with Architectural Heritage Features

Authors:
- Tianlai Zhang
- William Zheng
- Haochen Zhang

Central Question:
Can heritage-related variables improve the prediction of Manhattan property sale prices?
"""

import copy
import random
import time
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

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
st.markdown(
    f"""
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
    max-width: 54rem;
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
""",
    unsafe_allow_html=True,
)

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

    for col in [
        "building_name",
        "address",
        "architect",
        "style_primary",
        "material_primary",
        "historic_district",
        "construction_era",
        "zoning",
        "building_class_code",
    ]:
        if col in d.columns:
            d[col] = d[col].fillna("").astype(str)

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
# FEATURE SETS USED IN THE PREDICTION MODEL
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
    "architect_encoded",
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
        "architect",
        "material_primary",
        "style_primary",
        "zoning",
        "building_class_code",
    ]

    encoders = {}
    for cat in cat_cols:
        if cat in d.columns:
            le = LabelEncoder()
            d[f"{cat}_encoded"] = le.fit_transform(d[cat].fillna("Unknown").astype(str))
            encoders[cat] = le

    for c in ["residential_units", "commercial_units", "total_units"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    all_features = BASELINE + HERITAGE
    available = [f for f in all_features if f in d.columns]

    d[available] = d[available].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=available, thresh=len(available) - 6)

    for c in available:
        d[c] = d[c].fillna(d[c].median())

    return d, [f for f in BASELINE if f in available], [f for f in HERITAGE if f in available], encoders


mdf, FB, FH, LABEL_ENCODERS = prepare_features(df)
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
    "zoning_encoded": "Zoning District (Encoded)",
    "building_class_code_encoded": "Building Class (Encoded)",
    "sale_month": "Sale Month",
    "sale_price": "Sale Price",
    "price_per_sqft": "Price per SqFt",
    "building_age": "Building Age",
    "construction_era_encoded": "Construction Era (Encoded)",
    "architect_encoded": "Architect (Encoded)",
    "is_landmark": "Individual Landmark",
    "in_historic_district": "In Historic District",
    "is_altered": "Has Been Altered",
    "years_since_alteration": "Years Since Alteration",
    "material_primary_encoded": "Primary Facade Material (Encoded)",
    "style_primary_encoded": "Primary Architectural Style (Encoded)",
}

VARIABLE_DOCS = [
    {
        "Model": "Baseline",
        "Variable": "gross_sqft",
        "Meaning": "Gross building floor area used by the model.",
        "Source": "Sales, fallback to PLUTO",
        "Type": "Semi-engineered",
        "Engineering": "Starts from Sales 'GROSS SQUARE FEET'; when missing, falls back to PLUTO 'bldgarea'; zeros treated as missing.",
    },
    {
        "Model": "Baseline",
        "Variable": "land_sqft",
        "Meaning": "Land area for the property lot.",
        "Source": "Sales, fallback to PLUTO",
        "Type": "Semi-engineered",
        "Engineering": "Starts from Sales 'LAND SQUARE FEET'; when missing, falls back to PLUTO 'lotarea'; zeros treated as missing.",
    },
    {
        "Model": "Baseline",
        "Variable": "num_floors",
        "Meaning": "Number of floors in the building.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "lot_area",
        "Meaning": "Lot area from PLUTO.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "lot_depth",
        "Meaning": "Depth of the lot.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "lot_frontage",
        "Meaning": "Street frontage of the lot.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "building_depth",
        "Meaning": "Depth of the building footprint.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "building_frontage",
        "Meaning": "Street frontage of the building.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "residential_units",
        "Meaning": "Number of residential units.",
        "Source": "Sales",
        "Type": "Cleaned for modeling",
        "Engineering": "Converted to numeric; missing values filled with 0 in app.py for modeling.",
    },
    {
        "Model": "Baseline",
        "Variable": "commercial_units",
        "Meaning": "Number of commercial units.",
        "Source": "Sales",
        "Type": "Cleaned for modeling",
        "Engineering": "Converted to numeric; missing values filled with 0 in app.py for modeling.",
    },
    {
        "Model": "Baseline",
        "Variable": "total_units",
        "Meaning": "Total recorded unit count.",
        "Source": "Sales",
        "Type": "Cleaned for modeling",
        "Engineering": "Converted to numeric; missing values filled with 0 in app.py for modeling.",
    },
    {
        "Model": "Baseline",
        "Variable": "assess_total",
        "Meaning": "Total assessed value.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "assess_land",
        "Meaning": "Land-only assessed value.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "exempt_total",
        "Meaning": "Total tax-exempt assessed value.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "built_far",
        "Meaning": "Built floor-area ratio.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "resid_far",
        "Meaning": "Residential FAR cap.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "comm_far",
        "Meaning": "Commercial FAR cap.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "facil_far",
        "Meaning": "Facility FAR cap.",
        "Source": "PLUTO",
        "Type": "Raw cleaned",
        "Engineering": "Converted to numeric.",
    },
    {
        "Model": "Baseline",
        "Variable": "zoning_encoded",
        "Meaning": "Zoning district transformed into numeric labels for modeling.",
        "Source": "PLUTO zonedist1",
        "Type": "Engineered in app.py",
        "Engineering": "Original text zoning district is label-encoded in app.py so the model can use it.",
    },
    {
        "Model": "Baseline",
        "Variable": "building_class_code_encoded",
        "Meaning": "Building class code transformed into numeric labels for modeling.",
        "Source": "Sales building class at present",
        "Type": "Engineered in app.py",
        "Engineering": "Original text class code is label-encoded in app.py so the model can use it.",
    },
    {
        "Model": "Baseline",
        "Variable": "sale_month",
        "Meaning": "Calendar month of sale, used as a timing control.",
        "Source": "Derived from sale_date",
        "Type": "Engineered in prepare_data.py",
        "Engineering": "Created from the sale date with sale_date.dt.month.",
    },
    {
        "Model": "Heritage",
        "Variable": "building_age",
        "Meaning": "Approximate age of the building at the time of analysis.",
        "Source": "Derived from construction_year",
        "Type": "Engineered in prepare_data.py",
        "Engineering": "Computed as 2026 - construction_year after construction_year was assembled from Landmark Date_Low with fallback to PLUTO yearbuilt.",
    },
    {
        "Model": "Heritage",
        "Variable": "construction_era_encoded",
        "Meaning": "Construction period bucket transformed into numeric labels for modeling.",
        "Source": "Derived from construction_year",
        "Type": "Engineered in prepare_data.py and app.py",
        "Engineering": "Construction year is bucketed into eras like Pre-1850, 1850–1899, 1900–1919, 1920–1939, 1940–1969, and 1970+; that era label is then encoded in app.py.",
    },
    {
        "Model": "Heritage",
        "Variable": "architect_encoded",
        "Meaning": "Architect identity transformed into numeric labels for modeling.",
        "Source": "Landmark database Arch_Build",
        "Type": "Engineered in app.py",
        "Engineering": "The original architect name/category is label-encoded in app.py so architect can enter the model as a categorical heritage feature.",
    },
    {
        "Model": "Heritage",
        "Variable": "is_landmark",
        "Meaning": "Whether the property is coded as an individual landmark.",
        "Source": "Landmark database",
        "Type": "Engineered in prepare_data.py",
        "Engineering": "Built as a binary flag from LM_Orig and LM_New.",
    },
    {
        "Model": "Heritage",
        "Variable": "in_historic_district",
        "Meaning": "Whether the property is in a historic district.",
        "Source": "Landmark database",
        "Type": "Engineered in prepare_data.py",
        "Engineering": "Built as a binary flag: 1 when Hist_Dist is non-empty, otherwise 0.",
    },
    {
        "Model": "Heritage",
        "Variable": "is_altered",
        "Meaning": "Whether the building shows recorded alteration history.",
        "Source": "Landmark + PLUTO",
        "Type": "Engineered in prepare_data.py",
        "Engineering": "Built as a binary flag from Altered and alteration_year information.",
    },
    {
        "Model": "Heritage",
        "Variable": "years_since_alteration",
        "Meaning": "How long ago the most recent alteration occurred.",
        "Source": "Derived from alteration_year",
        "Type": "Engineered in prepare_data.py",
        "Engineering": "Computed as 2026 - alteration_year, with impossible negatives set to missing.",
    },
    {
        "Model": "Heritage",
        "Variable": "material_primary_encoded",
        "Meaning": "Primary facade material transformed into numeric labels for modeling.",
        "Source": "Landmark database Mat_Prim",
        "Type": "Engineered in app.py",
        "Engineering": "The original text material category is label-encoded in app.py.",
    },
    {
        "Model": "Heritage",
        "Variable": "style_primary_encoded",
        "Meaning": "Primary architectural style transformed into numeric labels for modeling.",
        "Source": "Landmark database Style_Prim",
        "Type": "Engineered in app.py",
        "Engineering": "The original text style category is label-encoded in app.py.",
    },
]

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
def flabel(var: str) -> str:
    pretty = FRIENDLY_LABELS.get(var)
    return f"{pretty} ({var})" if pretty else var


def title(text: str, sub: str = ""):
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
def wiki_lookup(query: str):
    if not query or not query.strip():
        return None
    try:
        search_resp = requests.get(
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

        hits = search_resp.get("query", {}).get("search", [])
        if not hits:
            return None

        page_title = hits[0]["title"]

        page_resp = requests.get(
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

        page = next(iter(page_resp.get("query", {}).get("pages", {}).values()), {})
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


def variable_dictionary_df(model_type: str):
    rows = [r for r in VARIABLE_DOCS if r["Model"] == model_type]
    return pd.DataFrame(rows)[["Variable", "Meaning", "Source", "Type", "Engineering"]]


def dataset_missing_info(dframe: pd.DataFrame):
    return pd.DataFrame(
        {
            "Column": dframe.columns,
            "Type": dframe.dtypes.astype(str).values,
            "Non-Null": dframe.notna().sum().values,
            "Missing %": (dframe.isna().mean() * 100).round(1).astype(str) + "%",
        }
    )


# =================================================================
# PAGE 1. BUSINESS CASE & DATA
# =================================================================
def page1():
    title(
        "Manhattan Heritage Valuation",
        "Can architectural and preservation features make property price predictions more accurate?"
    )

    st.markdown("<small>**Authors**: Tianlai Zhang, William Zheng, Haochen Zhang</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Research Question")
    st.markdown(
        """
**Can heritage-related variables improve the prediction of Manhattan property sale prices?**

This project is predictive rather than causal.  
It does **not** try to prove that heritage status directly causes higher prices.  
Instead, it compares two models:

- a **Baseline Model** built from standard real-estate variables
- a **Heritage-Enhanced Model** that adds architectural and preservation variables

If the heritage-enhanced model performs better, then heritage-related information contributes useful predictive signal.
"""
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Baseline Model")
        st.markdown(
            """
The baseline model treats each property mainly as a real-estate asset.  
It uses size, lot dimensions, unit counts, assessments, FAR, zoning, building class, and sale timing.
"""
        )
    with c2:
        st.markdown("#### Heritage-Enhanced Model")
        st.markdown(
            """
The heritage-enhanced model keeps every baseline variable and adds these heritage features:

- building age
- construction era
- architect
- landmark status
- historic district status
- alteration status
- years since alteration
- facade material
- architectural style
"""
        )

    st.markdown("---")
    st.markdown("### Data Sources")
    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("**NYC Rolling Sales**")
        st.caption("Sale price, date, square footage, unit counts, building class, neighborhood.")
        st.metric("Records", "18,817")

    with d2:
        st.markdown("**MapPLUTO**")
        st.caption("Lot dimensions, building dimensions, FAR, zoning, assessments, coordinates.")
        st.metric("Records", "42,600")

    with d3:
        st.markdown("**Landmark Database**")
        st.caption("Architectural style, material, architect, landmark flags, historic district information.")
        st.metric("Records", "14,610")

    st.markdown(
        """
The merged CSV was built by joining the three Manhattan-only datasets on **BBL**  
(Borough-Block-Lot, New York City's parcel identifier).

The app does **not** invent outside data.  
However, several columns were **engineered** from raw fields, such as binary landmark indicators, construction era bins, sale month, building age, years since alteration, and encoded categorical variables.
"""
    )

    cards(
        [
            ("Final merged records", f"{len(df):,}"),
            ("Columns in merged CSV", f"{len(df.columns)}"),
            ("Rows used for modeling", f"{len(mdf):,}"),
            ("Heritage model variables", f"{len(FH)}"),
        ]
    )

    st.markdown("---")
    st.markdown("### What happens in the data preparation pipeline?")
    st.markdown(
        """
1. Read three raw datasets: Landmark, Sales, and PLUTO.  
2. Filter each dataset to Manhattan only.  
3. Construct a clean BBL for Sales.  
4. Merge datasets on BBL.  
5. Clean numeric fields such as prices, square footage, assessments, and FAR.  
6. Create derived variables such as `is_landmark`, `in_historic_district`, `construction_year`, `building_age`, `construction_era`, `is_altered`, `years_since_alteration`, `sale_year`, and `sale_month`.  
7. In `app.py`, encode the categorical fields needed by the model, such as zoning, building class, construction era, architect, material, and style.  
8. Restrict the modeling sample to transactions with sale price at least $100,000 and then impute the remaining missing values with medians where needed.
"""
    )

    st.markdown("---")
    st.markdown("### Prediction Variables Used in the Model")
    st.caption("These are the only variables currently used in the prediction model. Other columns in the CSV are kept for exploration and visualization.")

    tab1, tab2 = st.tabs(["Baseline variables", "Heritage variables"])
    with tab1:
        st.dataframe(variable_dictionary_df("Baseline"), use_container_width=True, height=560)
    with tab2:
        st.dataframe(variable_dictionary_df("Heritage"), use_container_width=True, height=460)

    st.markdown("---")
    st.markdown("### Data Preview")
    st.caption("This preview shows the target and the human-readable versions of the model variables. Encoded columns are used in modeling, but not displayed here.")

    preview_cols = [
        "sale_price",
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
        "zoning",
        "building_class_code",
        "sale_month",
        "building_age",
        "construction_era",
        "architect",
        "is_landmark",
        "in_historic_district",
        "is_altered",
        "years_since_alteration",
        "material_primary",
        "style_primary",
    ]

    preview_cols = [c for c in preview_cols if c in mdf.columns]
    preview_df = mdf[preview_cols].head(20).copy()

    preview_rename = {
        "sale_price": "Sale Price",
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
        "zoning": "Zoning District",
        "building_class_code": "Building Class",
        "sale_month": "Sale Month",
        "building_age": "Building Age",
        "construction_era": "Construction Era",
        "architect": "Architect",
        "is_landmark": "Individual Landmark",
        "in_historic_district": "In Historic District",
        "is_altered": "Has Been Altered",
        "years_since_alteration": "Years Since Alteration",
        "material_primary": "Primary Facade Material",
        "style_primary": "Primary Architectural Style",
    }

    preview_df = preview_df.rename(columns=preview_rename)
    st.dataframe(preview_df, use_container_width=True, height=380)

    st.markdown("---")
    st.markdown("### Missing Values")
    st.caption("The first table shows the loaded analysis dataset. The second shows the model-ready dataset after app-level filtering, encoding, and imputation.")

    mv1, mv2 = st.tabs(["Full dataset", "Model-ready dataset"])

    with mv1:
        st.markdown(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns)}")
        st.dataframe(dataset_missing_info(df), use_container_width=True, height=420)

    with mv2:
        st.markdown(f"**Rows:** {len(mdf):,}  |  **Columns:** {len(mdf.columns)}")
        st.dataframe(dataset_missing_info(mdf), use_container_width=True, height=420)
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
        ["Interactive Map", "Price Analysis", "Architectural Patterns", "Era & Correlation"],
        horizontal=True,
    )

    if section == "Interactive Map":
        st.markdown("### 3D Property Map")
        st.caption("Each column represents a property. Height and color encode selected variables.")

        c1, c2, c3 = st.columns(3)
        with c1:
            color_var = st.selectbox(
                "Color by",
                ["sale_price", "price_per_sqft", "building_age", "num_floors", "assess_total"],
                format_func=flabel,
            )
        with c2:
            height_var = st.selectbox(
                "Height by",
                ["sale_price", "gross_sqft", "num_floors", "building_age", "price_per_sqft"],
                format_func=flabel,
            )
        with c3:
            cmap_name = st.selectbox("Color palette", ["magma", "inferno", "viridis", "plasma", "turbo", "cividis"])

        map_d = viz.dropna(subset=[color_var, height_var, "latitude", "longitude"]).copy()

        h_vals = pd.to_numeric(map_d[height_var], errors="coerce")
        h_cap = h_vals.quantile(0.97)
        if pd.isna(h_cap) or h_cap == 0:
            h_cap = 1
        map_d["_h"] = (h_vals.clip(lower=0, upper=h_cap) / h_cap * 800).fillna(0)

        c_vals = pd.to_numeric(map_d[color_var], errors="coerce").fillna(map_d[color_var].median())
        norm = mcolors.Normalize(vmin=c_vals.quantile(0.02), vmax=c_vals.quantile(0.98))
        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(norm(c_vals.values))
        map_d["_r"] = (rgba[:, 0] * 255).astype(int)
        map_d["_g"] = (rgba[:, 1] * 255).astype(int)
        map_d["_b"] = (rgba[:, 2] * 255).astype(int)

        map_d["price_fmt"] = map_d["sale_price"].apply(lambda v: f"${v:,.0f}")
        map_d["sqft_fmt"] = map_d["gross_sqft"].apply(lambda v: f"{v:,.0f} sqft" if pd.notna(v) else "unknown sqft")
        map_d["style_fmt"] = map_d["style_primary"].fillna("")
        map_d["era_fmt"] = map_d["construction_era"].fillna("")
        map_d["architect_fmt"] = map_d["architect"].fillna("Unknown architect")

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
                "<span style='color:#9ca3af;'>by {architect_fmt}</span><br/>"
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
            hover_data={"sale_price": ":,.0f", "construction_era": True, "latitude": False, "longitude": False},
            zoom=12,
            center={"lat": 40.754, "lon": -73.987},
            height=540,
            template=T["plotly"],
        )
        fig2.update_layout(mapbox_style=T["map_style"], margin={"r": 0, "t": 0, "l": 0, "b": 0})
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

        hood = viz.groupby("neighborhood")["sale_price"].median().sort_values(ascending=False).head(20).reset_index()
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
        st.markdown("### Landmark vs Non-Landmark")
        temp = viz.copy()
        temp["landmark_label"] = np.where(temp["is_landmark"] == 1, "Landmark", "Not Landmark")
        fig = px.box(
            temp,
            x="landmark_label",
            y="price_per_sqft",
            color="landmark_label",
            template=T["plotly"],
            labels={"landmark_label": "", "price_per_sqft": "$/SqFt"},
        )
        st.plotly_chart(fig, use_container_width=True)

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
                labels={"building_age": "Age", "sale_price": "Sale Price ($)"},
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
                labels={"is_altered": "Altered (1 = Yes)", "price_per_sqft": "$/SqFt"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Correlation Matrix")

        left_col, right_col = st.columns([3, 1.35])

        with right_col:
            st.markdown("#### How to read this")
            st.markdown(
                """
This matrix is meant for **continuous numeric variables first**.

So by default, it focuses on variables such as:

- sale price
- square footage
- lot dimensions
- assessments
- FAR
- building age
- years since alteration

**Binary variables** like landmark status and historic district status, and **encoded categorical variables** like architect, style, material, zoning, and building class are **not shown by default**.

Why:
- binary variables may have very little variation in the current filtered sample
- encoded variables are labels, not true continuous numbers
- Pearson correlation is more interpretable for continuous variables

You can still include them by turning on the option below, but interpret those results more cautiously.
"""
            )

        with left_col:
            st.caption(
                "Choose which variables to include. Default selections prioritize continuous numeric variables. "
                "Variables with no variation in the current sample are removed automatically."
            )

            continuous_corr_candidates = [
                "sale_price",
                "price_per_sqft",
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
                "building_age",
                "years_since_alteration",
                "sale_month",
            ]

            encoded_binary_candidates = [
                "zoning_encoded",
                "building_class_code_encoded",
                "construction_era_encoded",
                "architect_encoded",
                "material_primary_encoded",
                "style_primary_encoded",
                "is_landmark",
                "in_historic_district",
                "is_altered",
            ]

            include_encoded_binary = st.checkbox(
                "Include encoded / binary variables",
                value=False,
                key="include_encoded_binary_corr",
            )

            corr_candidates = continuous_corr_candidates.copy()
            if include_encoded_binary:
                corr_candidates += encoded_binary_candidates

            available_corr_cols = [
                c for c in corr_candidates
                if c in viz.columns and pd.api.types.is_numeric_dtype(viz[c])
            ]

            default_corr_cols = [
                c for c in [
                    "sale_price",
                    "price_per_sqft",
                    "gross_sqft",
                    "land_sqft",
                    "num_floors",
                    "lot_area",
                    "assess_total",
                    "assess_land",
                    "built_far",
                    "building_age",
                    "years_since_alteration",
                    "sale_month",
                ]
                if c in available_corr_cols
            ]

            selected_corr_cols = st.multiselect(
                "Variables to include",
                options=available_corr_cols,
                default=default_corr_cols,
                format_func=flabel,
                key="corr_var_selector",
            )

            if len(selected_corr_cols) < 2:
                st.warning("Please select at least 2 variables.")
            else:
                valid_corr_cols = [
                    c for c in selected_corr_cols
                    if viz[c].nunique(dropna=True) > 1
                ]

                removed_cols = [c for c in selected_corr_cols if c not in valid_corr_cols]

                if len(valid_corr_cols) < 2:
                    st.warning("Not enough variables with variation in the current sample to compute correlations.")
                else:
                    if removed_cols:
                        st.info(
                            "Removed from the matrix because they have no variation in the current filtered sample: "
                            + ", ".join([flabel(c) for c in removed_cols])
                        )

                    cm = viz[valid_corr_cols].corr()
                    mask = np.triu(np.ones_like(cm, dtype=bool))

                    fig_height = max(8, min(18, len(valid_corr_cols) * 0.48))
                    fig_width = max(10, min(20, len(valid_corr_cols) * 0.58))

                    fig_c, ax = plt.subplots(figsize=(fig_width, fig_height))
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
                        cbar_kws={"shrink": 0.8},
                    )
                    ax.set_title("Selected Variable Correlation Matrix")
                    plt.xticks(rotation=45, ha="right")
                    plt.yticks(rotation=0)
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

    st.markdown(
        """
This page directly tests the research question.

- **Baseline model** = standard real-estate variables only
- **Heritage-enhanced model** = baseline variables + selected heritage variables, including architect

The target is **log-transformed sale price**.
Performance is shown with **R²** and dollar-scale error metrics.
"""
    )

    # ─────────────────────────────────────────────────────────────
    # DESIGN EVOLUTION — how our feature set evolved
    # ─────────────────────────────────────────────────────────────
    with st.expander("📜 Our Design Journey — How the feature set evolved", expanded=False):
        st.markdown(
            """
We did **not** arrive at the current feature set in one shot.
The story below is the most important methodological finding of the project.
"""
        )

        ev_c1, ev_c2 = st.columns(2)

        with ev_c1:
            st.markdown("##### Phase 1 — Naive Baseline (11 features)")
            st.markdown(
                """
`gross_sqft`, `lot_size`, `num_floors`, `year_built`,
`residential_units`, `commercial_units`, `total_units`,
`bldg_class`, `borough`, `zoning_code`, `council_district`
"""
            )
            st.markdown(
                """
**Results**
- Linear R² ≈ **0.12**
- Random Forest R² ≈ **0.50**
- CatBoost R² ≈ **0.53**

**Heritage uplift = +5% to +8%** → looked like a huge win.
"""
            )

        with ev_c2:
            st.markdown("##### Phase 2 — Expanded Baseline (21 features)")
            st.markdown(
                """
Added the variables every professional valuation uses:

`assess_total` ← **the game changer**, `assess_land`,
`exempt_total`, `land_area`, `bldg_area`,
`comm_far`, `resid_far`, `built_far`,
`sale_month`, `year_alter1`
"""
            )
            st.markdown(
                """
**Results**
- Linear R²: 0.12 → **0.19** (+58% relative)
- Random Forest R²: 0.50 → **0.59** (+18%)
- CatBoost R²: 0.53 → **0.61** (+15%)

**Heritage uplift collapsed to +0.65%.**
"""
            )

        # Chart: baseline R² before vs after expansion
        evo_df = pd.DataFrame({
            "Model": ["Linear", "Random Forest", "CatBoost"],
            "Phase 1 baseline (11 feats)": [0.12, 0.50, 0.53],
            "Phase 2 baseline (21 feats)": [0.19, 0.59, 0.61],
        })

        fig_evo = go.Figure()
        fig_evo.add_trace(go.Bar(
            name="Phase 1 — 11 features",
            x=evo_df["Model"],
            y=evo_df["Phase 1 baseline (11 feats)"],
            marker_color="#94a3b8",
            text=[f"{v:.2f}" for v in evo_df["Phase 1 baseline (11 feats)"]],
            textposition="outside",
        ))
        fig_evo.add_trace(go.Bar(
            name="Phase 2 — 21 features (+ assess_total)",
            x=evo_df["Model"],
            y=evo_df["Phase 2 baseline (21 feats)"],
            marker_color=T["accent"],
            text=[f"{v:.2f}" for v in evo_df["Phase 2 baseline (21 feats)"]],
            textposition="outside",
        ))
        fig_evo.update_layout(
            template=T["plotly"],
            barmode="group",
            height=340,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis_title="R² (test)",
            yaxis=dict(range=[0, 0.75]),
            title=dict(text="Baseline R² jumped after adding assess_total + FAR + sale_month",
                       x=0.02, font=dict(size=14)),
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_evo, use_container_width=True)

        # Chart: heritage uplift collapse
        uplift_df = pd.DataFrame({
            "Stage": ["Phase 1 baseline\n(weak — 11 feats)",
                      "Phase 2 baseline\n(strong — 21 feats)"],
            "Heritage uplift (R²)": [0.065, 0.0065],
        })
        fig_up = go.Figure(go.Bar(
            x=uplift_df["Stage"],
            y=uplift_df["Heritage uplift (R²)"],
            marker_color=["#94a3b8", T["accent"]],
            text=[f"+{v*100:.2f}%" for v in uplift_df["Heritage uplift (R²)"]],
            textposition="outside",
        ))
        fig_up.update_layout(
            template=T["plotly"],
            height=320,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis_title="Heritage uplift (Δ R²)",
            yaxis=dict(range=[0, 0.085], tickformat=".1%"),
            title=dict(text="Heritage uplift collapsed once assess_total entered the baseline",
                       x=0.02, font=dict(size=14)),
        )
        st.plotly_chart(fig_up, use_container_width=True)

        st.info(
            "**Why the collapse is the key insight.** "
            "City assessors already consider landmark status, era, style, and historic district when "
            "they compute `assess_total` — and `assess_total` correlates with sale price at **r ≈ 0.70**. "
            "Once that single variable is in the model, explicit heritage features add little *new* signal. "
            "Heritage value didn't vanish; the **market had already priced it in**. "
            "The +0.65% uplift is evidence of **market efficiency**, not of feature failure."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        test_sz = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    with c2:
        rs = st.number_input("Random state", 0, 100, 42)
    with c3:
        do_scale = st.checkbox("Scale linear models", True)

    idx_all = np.arange(len(mdf))
    train_idx, test_idx = train_test_split(idx_all, test_size=test_sz, random_state=rs)

    X_b = mdf[FB].values
    X_a = mdf[FALL].values
    y = mdf["log_price"].values

    Xb_tr, Xb_te = X_b[train_idx], X_b[test_idx]
    Xa_tr, Xa_te = X_a[train_idx], X_a[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    if do_scale:
        scb = StandardScaler()
        Xb_tr_s = scb.fit_transform(Xb_tr)
        Xb_te_s = scb.transform(Xb_te)

        sca = StandardScaler()
        Xa_tr_s = sca.fit_transform(Xa_tr)
        Xa_te_s = sca.transform(Xa_te)
    else:
        Xb_tr_s, Xb_te_s = Xb_tr, Xb_te
        Xa_tr_s, Xa_te_s = Xa_tr, Xa_te
        sca = None

    MODELS = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=1.0), True),
        "Lasso Regression": (Lasso(alpha=0.05), True),
        "Elastic Net": (ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=rs), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=10, random_state=rs), False),
        "Random Forest": (
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=rs),
            False,
        ),
    }

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

    base_rows, herit_rows = [], []
    trained_h = {}
    timings = {}

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

        base_rows.append(
            {"Model": nm, **{k: v for k, v in rb.items() if k not in ("preds", "model")}, "Time(s)": round(elapsed, 2)}
        )
        herit_rows.append(
            {"Model": nm, **{k: v for k, v in rh.items() if k not in ("preds", "model")}, "Time(s)": round(elapsed, 2)}
        )
        herit_rows[-1]["preds"] = rh["preds"]

        trained_h[nm] = {
            "model": mh,
            "needs_scale": needs_scale,
        }

        progress.progress(i / len(sel), text=f"Done {i}/{len(sel)} · {nm} took {elapsed:.1f}s")

        live_lb = pd.DataFrame(herit_rows)[["Model", "R2_test"]].sort_values("R2_test", ascending=True)
        is_winner = live_lb["R2_test"] == live_lb["R2_test"].max()
        bar_colors = [T["accent"] if w else "#94a3b8" for w in is_winner]

        live_fig = go.Figure(
            go.Bar(
                x=live_lb["R2_test"],
                y=live_lb["Model"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.3f}" for v in live_lb["R2_test"]],
                textposition="outside",
            )
        )
        live_fig.update_layout(
            template=T["plotly"],
            height=max(220, 38 * len(live_lb) + 80),
            margin=dict(l=10, r=30, t=30, b=10),
            xaxis_title="R² Test Score",
            title=dict(text=f"🏆 Live Leaderboard — leader: {live_lb.iloc[-1]['Model']}", x=0.02, font=dict(size=14)),
            xaxis=dict(range=[0, max(0.7, live_lb["R2_test"].max() * 1.1)]),
        )
        leaderboard_box.plotly_chart(live_fig, use_container_width=True, key=f"live_lb_{i}")

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
    st.session_state["Xa_tr_raw"] = Xa_tr
    st.session_state["Xa_te_raw"] = Xa_te
    st.session_state["Xa_tr_scaled"] = Xa_tr_s
    st.session_state["Xa_te_scaled"] = Xa_te_s
    st.session_state["heritage_scaler"] = sca
    st.session_state["y_tr"] = y_tr
    st.session_state["y_te"] = y_te
    st.session_state["test_idx"] = test_idx

    st.markdown("---")
    st.markdown("### Results: Baseline vs Heritage-Enhanced")

    c1, c2 = st.columns(2)
    fmt_cols = ["R2_train", "R2_test", "MAE_dollars", "RMSE_dollars", "Time(s)"]

    with c1:
        st.markdown("**Baseline Model — structural variables only**")
        st.dataframe(
            dfb[fmt_cols].style.format(
                {
                    "R2_train": "{:.4f}",
                    "R2_test": "{:.4f}",
                    "MAE_dollars": "${:,.0f}",
                    "RMSE_dollars": "${:,.0f}",
                    "Time(s)": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    with c2:
        st.markdown("**Heritage-Enhanced Model — baseline + heritage variables**")
        st.dataframe(
            dfh[fmt_cols].style.format(
                {
                    "R2_train": "{:.4f}",
                    "R2_test": "{:.4f}",
                    "MAE_dollars": "${:,.0f}",
                    "RMSE_dollars": "${:,.0f}",
                    "Time(s)": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Heritage Feature Uplift")
    st.caption("This chart shows whether adding the selected heritage variables improves test performance.")

    up = pd.DataFrame(
        {
            "Model": sel,
            "Baseline R²": [dfb.loc[n, "R2_test"] for n in sel],
            "Heritage-Enhanced R²": [dfh.loc[n, "R2_test"] for n in sel],
        }
    )
    up["Uplift"] = up["Heritage-Enhanced R²"] - up["Baseline R²"]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Baseline", x=up["Model"], y=up["Baseline R²"], marker_color="steelblue"))
    fig.add_trace(go.Bar(name="Heritage-Enhanced", x=up["Model"], y=up["Heritage-Enhanced R²"], marker_color=T["accent"]))
    fig.update_layout(barmode="group", template=T["plotly"], height=400, yaxis_title="R² Test Score")
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
        labels={"x": "Actual log price", "y": "Predicted log price"},
    )
    fig.add_shape(type="line", x0=y_te.min(), x1=y_te.max(), y0=y_te.min(), y1=y_te.max(), line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Where the Model Disagrees with the Market")
    st.caption("Residuals compare actual price with predicted price. Positive means the market paid more than the model expected.")

    test_rows = mdf.iloc[test_idx].copy().reset_index(drop=True)
    test_rows["actual_price"] = np.expm1(y_te)
    test_rows["pred_price"] = np.expm1(yp)
    test_rows["residual_log"] = y_te - yp
    test_rows["residual_pct"] = (test_rows["actual_price"] / test_rows["pred_price"] - 1) * 100
    test_rows["abs_dollar_diff"] = test_rows["actual_price"] - test_rows["pred_price"]

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
            ticktext=["−80%<br>Below model", "−40%", "0%<br>On model", "+40%", "+80%<br>Above model"],
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### Biggest Model Misses")
    show_cols = ["display_name", "construction_era", "actual_price", "pred_price", "residual_pct", "neighborhood"]
    rename = {
        "display_name": "Property",
        "construction_era": "Era",
        "actual_price": "Actual",
        "pred_price": "Predicted",
        "residual_pct": "Diff %",
        "neighborhood": "Neighborhood",
    }
    fmt = {"Actual": "${:,.0f}", "Predicted": "${:,.0f}", "Diff %": "{:+.1f}%"}

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 10 Above Model Prediction**")
        top_above = test_rows.sort_values("residual_pct", ascending=False).head(10)[show_cols].rename(columns=rename)
        st.dataframe(top_above.style.format(fmt), use_container_width=True, height=380)
    with c2:
        st.markdown("**Top 10 Below Model Prediction**")
        top_below = test_rows.sort_values("residual_pct", ascending=True).head(10)[show_cols].rename(columns=rename)
        st.dataframe(top_below.style.format(fmt), use_container_width=True, height=380)


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

    sel = st.selectbox("Model", list(trained.keys()))
    model_info = trained[sel]
    model = model_info["model"]
    needs_scale = model_info["needs_scale"]

    Xa_te = st.session_state["Xa_te_scaled"] if needs_scale else st.session_state["Xa_te_raw"]
    Xa_tr = st.session_state["Xa_tr_scaled"] if needs_scale else st.session_state["Xa_tr_raw"]

    section = st.radio("View", ["Feature Importance", "SHAP Analysis"], horizontal=True)

    if section == "Feature Importance":
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_)
        else:
            st.warning("This model does not expose importance values.")
            return

        fi = pd.DataFrame(
            {
                "Feature": [flabel(f) for f in fnames],
                "Importance": imp,
                "Type": ["Baseline" if f in FB else "Heritage" for f in fnames],
            }
        ).sort_values("Importance", ascending=True)

        fig = px.bar(
            fi,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Type",
            color_discrete_map={"Baseline": "steelblue", "Heritage": T["accent"]},
            template=T["plotly"],
            height=560,
            labels={"Importance": "Importance", "Feature": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

        total = fi["Importance"].sum()
        heritage_share = fi[fi["Type"] == "Heritage"]["Importance"].sum()
        if total > 0:
            st.info(f"Heritage-related features account for **{heritage_share / total * 100:.1f}%** of this model's total importance.")

        if hasattr(model, "coef_"):
            st.markdown("---")
            st.markdown("### Coefficient Direction")
            coef = pd.DataFrame(
                {
                    "Feature": [flabel(f) for f in fnames],
                    "Coefficient": model.coef_,
                    "Type": ["Baseline" if f in FB else "Heritage" for f in fnames],
                }
            ).sort_values("Coefficient", key=abs, ascending=True)

            fig = px.bar(
                coef,
                x="Coefficient",
                y="Feature",
                orientation="h",
                color="Type",
                color_discrete_map={"Baseline": "steelblue", "Heritage": T["accent"]},
                template=T["plotly"],
                height=560,
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


# =================================================================
# PAGE 5. HYPERPARAMETER TUNING
# =================================================================
def page5():
    title(
        "Hyperparameter Tuning",
        "Testing different model settings to improve prediction performance"
    )

    with st.expander("📚 What is hyperparameter tuning?", expanded=False):
        st.markdown(
            """
Hyperparameters are the settings chosen before training a model.

Examples:
- the number of trees in a random forest
- the max depth of a tree
- the learning rate in boosting models

This page tries different combinations and compares their performance on the heritage-enhanced feature set.
"""
        )

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
    c1, c2 = st.columns(2)
    with c1:
        model_name = st.selectbox("Model", tune_choices)
    with c2:
        test_sz = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="hp_ts")

    st.markdown("---")
    st.markdown("### Hyperparameter Grid")
    grid = []

    if model_name == "Ridge Regression":
        alphas = st.multiselect("Alpha", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], default=[0.01, 0.1, 1.0, 10.0, 100.0])
        grid = [{"alpha": a} for a in alphas]

    elif model_name == "Lasso Regression":
        alphas = st.multiselect("Alpha", [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0], default=[0.001, 0.01, 0.1, 0.5])
        grid = [{"alpha": a} for a in alphas]

    elif model_name == "Elastic Net":
        alphas = st.multiselect("Alpha", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0], default=[0.01, 0.05, 0.1])
        l1s = st.multiselect("L1 ratio", [0.1, 0.3, 0.5, 0.7, 0.9], default=[0.3, 0.5, 0.7])
        grid = [{"alpha": a, "l1_ratio": l} for a in alphas for l in l1s]

    elif model_name == "Decision Tree":
        depths = st.multiselect("Max Depth", [2, 3, 5, 7, 10, 15, None], default=[3, 5, 10])
        mins = st.multiselect("Min Samples Split", [2, 5, 10, 20], default=[2, 5, 10])
        grid = [{"max_depth": d, "min_samples_split": s} for d in depths for s in mins]

    elif model_name == "Random Forest":
        nes = st.multiselect("N Estimators", [50, 100, 200, 300], default=[50, 100, 200])
        deps = st.multiselect("Max Depth", [5, 10, 15, None], default=[5, 10, 15])
        grid = [{"n_estimators": n, "max_depth": d} for n in nes for d in deps]

    else:
        lrs = st.multiselect("Learning Rate", [0.01, 0.05, 0.1, 0.2], default=[0.01, 0.1])
        nes = st.multiselect("N Estimators", [50, 100, 200], default=[50, 100])
        deps = st.multiselect("Max Depth", [3, 5, 7], default=[3, 5])
        grid = [{"learning_rate": lr, "n_estimators": n, "max_depth": d} for lr in lrs for n in nes for d in deps]

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
        wb_key = st.text_input("API key optional", "", type="password", help="Leave empty if you already ran wandb login.")

    st.markdown("---")
    if st.button("Run Tuning", type="primary"):
        if not grid:
            st.error("Configure at least one hyperparameter combination.")
            return

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_sz, random_state=42)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        needs_scale = model_name in ["Ridge Regression", "Lasso Regression", "Elastic Net"]

        wb_ok = False
        wandb_mod = None
        if use_wb:
            try:
                import wandb as wandb_mod

                if wb_key.strip():
                    wandb_mod.login(key=wb_key.strip(), relogin=True)

                api_key_present = bool(wandb_mod.api.api_key)
                if not api_key_present:
                    st.error("W&B has no API key in this session. Run wandb login or paste your key.")
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
            else:
                m = GradientBoostingRegressor(**params, random_state=42)

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
                        config={"model": model_name, "test_size": test_sz, **params},
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
            res_df.style.highlight_max(subset=["r2_test"], color="#667eea")
            .highlight_min(subset=["rmse_dollars", "mae_dollars"], color="#2ecc71")
            .format(
                {
                    "r2_train": "{:.4f}",
                    "r2_test": "{:.4f}",
                    "mae_dollars": "${:,.0f}",
                    "rmse_dollars": "${:,.0f}",
                }
            ),
            use_container_width=True,
        )

        best = res_df.iloc[0]
        pcols = [c for c in res_df.columns if c not in ["r2_train", "r2_test", "mae_dollars", "rmse_dollars"]]
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
        fig2.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
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

    train_idx, _ = train_test_split(np.arange(len(mdf)), test_size=0.2, random_state=42)

    Xb_tr, y_tr = X_b[train_idx], y[train_idx]
    Xa_tr = X_a[train_idx]

    mb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42).fit(Xb_tr, y_tr)
    mh = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42).fit(Xa_tr, y_tr)
    return mb, mh, Xa_tr


@st.cache_resource
def get_quantile_models():
    X_a = mdf[FALL].values
    y = mdf["log_price"].values
    train_idx, _ = train_test_split(np.arange(len(mdf)), test_size=0.2, random_state=42)

    Xa_tr, y_tr = X_a[train_idx], y[train_idx]

    out = {}
    for q in (0.1, 0.5, 0.9):
        m = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        ).fit(Xa_tr, y_tr)
        out[q] = m
    return out


def page6():
    title(
        "Property Valuator",
        "Compare a structural prediction with a heritage-enhanced prediction for one property"
    )

    with st.spinner("Warming up valuation models..."):
        mb, mh, _ = get_valuation_models()

    pool = mdf.dropna(subset=FALL).copy()

    st.markdown("### Find a property")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        query = st.text_input("Search address, building name, architect, or style", "").strip().lower()
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
        mask = (
            subset["address"].astype(str).str.lower().str.contains(query, na=False)
            | subset["building_name"].astype(str).str.lower().str.contains(query, na=False)
            | subset["architect"].astype(str).str.lower().str.contains(query, na=False)
            | subset["style_primary"].astype(str).str.lower().str.contains(query, na=False)
        )
        subset = subset[mask]

    subset = subset.sort_values("sale_price", ascending=False)

    if len(subset) == 0:
        st.info("No matches. Try a different search.")
        return

    st.caption(f"{len(subset):,} matching properties. Showing top 200 by price.")

    def label(i):
        r = subset.loc[i]
        nm = r.get("display_name") or r.get("address", "Unknown")
        arch = r["architect"] if pd.notna(r.get("architect")) and str(r["architect"]).strip() else "Unknown architect"
        return f"{nm} · {arch} · ${r['sale_price']:,.0f}"

    pick_idx = st.selectbox("Pick a property", subset.index[:200].tolist(), format_func=label)
    row = subset.loc[pick_idx]

    x_base = row[FB].values.reshape(1, -1).astype(float)
    x_all = row[FALL].values.reshape(1, -1).astype(float)

    pred_base = float(np.expm1(mb.predict(x_base)[0]))
    pred_her = float(np.expm1(mh.predict(x_all)[0]))
    actual = float(row["sale_price"])

    heritage_adjustment = pred_her - pred_base
    heritage_adjustment_pct = heritage_adjustment / pred_base * 100 if pred_base else 0
    gap = actual - pred_her
    gap_pct = gap / pred_her * 100 if pred_her else 0

    st.markdown("---")
    adjustment_color = T["accent"] if heritage_adjustment >= 0 else "#ef4444"

    cols = st.columns(4)
    cols[0].markdown(
        f'<div class="card"><p class="val">${actual:,.0f}</p><p class="lbl">Actual sale price</p></div>',
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        f'<div class="card"><p class="val">${pred_base:,.0f}</p><p class="lbl">Baseline prediction</p></div>',
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        f'<div class="card"><p class="val">${pred_her:,.0f}</p><p class="lbl">Heritage-enhanced prediction</p></div>',
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        f'<div class="card"><p class="val" style="color:{adjustment_color}">{heritage_adjustment:+,.0f}</p>'
        f'<p class="lbl">Heritage feature adjustment ({heritage_adjustment_pct:+.1f}%)</p></div>',
        unsafe_allow_html=True,
    )

    if abs(gap_pct) < 10:
        st.success(f"Actual price is close to the heritage-enhanced prediction ({gap_pct:+.1f}%).")
    elif gap_pct > 0:
        st.info(f"Actual sale price is **above** the heritage-enhanced prediction by {gap_pct:+.1f}% (${abs(gap):,.0f}).")
    else:
        st.warning(f"Actual sale price is **below** the heritage-enhanced prediction by {gap_pct:+.1f}% (${abs(gap):,.0f}).")

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
        fig.add_trace(
            go.Scatter(
                x=[q_low, q_high],
                y=[0, 0],
                mode="lines",
                line=dict(color=T["accent"], width=18),
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[q_low, q_high],
                y=[0, 0],
                mode="markers+text",
                marker=dict(size=10, color=T["accent"]),
                text=[f"P10<br>${q_low:,.0f}", f"P90<br>${q_high:,.0f}"],
                textposition=["top left", "top right"],
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[q_med],
                y=[0],
                mode="markers+text",
                marker=dict(size=20, color=T["accent"], line=dict(color="white", width=2)),
                text=[f"Model median<br>${q_med:,.0f}"],
                textposition="bottom center",
                name="Median",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[actual],
                y=[0],
                mode="markers+text",
                marker=dict(size=22, color=actual_color, symbol="diamond", line=dict(color="white", width=2)),
                text=[f"Actual<br>${actual:,.0f}"],
                textposition="bottom center",
                name="Actual",
            )
        )
        pad = max(q_high - q_low, 1) * 0.15
        fig.update_layout(
            template=T["plotly"],
            height=230,
            xaxis=dict(tickformat="$,.0f", range=[min(q_low, actual) - pad, max(q_high, actual) + pad]),
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
    wiki = None
    if bname and bname.lower() not in ("0", "none", "nan"):
        wiki = wiki_lookup(bname)

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
                fmt(row.get("alteration_year"), "year") if row.get("is_altered") == 1 else "Original",
            ),
        ]
        info_df = pd.DataFrame(info, columns=["Field", "Value"])
        st.dataframe(info_df, use_container_width=True, hide_index=True, height=480)

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
            center={"lat": row["latitude"], "lon": row["longitude"]},
            height=480,
            template=T["plotly"],
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=[row["latitude"]],
                lon=[row["longitude"]],
                mode="markers",
                marker=dict(size=24, color=T["accent"]),
                hovertext=[row.get("display_name", "This property")],
                name="Selected",
                showlegend=False,
            )
        )
        fig.update_layout(mapbox_style=T["map_style"], margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Why This Price? Feature Contributions")
    st.caption("This explains how the selected heritage variables and baseline variables push the prediction up or down.")

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
