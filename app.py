"""
Manhattan Heritage Property Valuation
Financial Premium of Historic Preservation in Manhattan

Central Question: Does architectural heritage create measurable market value?
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

# Optional: LightGBM. If missing, gracefully disable its models.
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# Optional: CatBoost. Native categorical handling, works great on smallish datasets.
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
# Three modern looks. Porcelain is pure Apple light mode.
# ─────────────────────────────────────────────────────────────
THEMES = {
    # Apple light. System white, SF blue, crisp and quiet.
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
    # Apple dark. Deep black, electric blue into purple.
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
    # Linear.app vibe. Midnight navy + cyan into violet.
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

# Sidebar
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

# Inject CSS. Inter font + glassy cards + tight vertical rhythm.
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

  /* Gradient page title */
  .page-title {{
    font-size: 2.6rem; font-weight: 800; line-height: 1.1;
    background: {T["header_grad"]};
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem; letter-spacing: -0.03em;
  }}
  .page-subtitle {{
    font-size: 1.02rem; color: {T["muted"]};
    margin-bottom: 1.6rem; font-weight: 400; max-width: 52rem;
  }}

  /* Glassy KPI card */
  .card {{
    background: {T["card_bg"]};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {T["card_border"]};
    border-radius: 14px; padding: 1.1rem 1.2rem;
    box-shadow: {T["shadow"]};
    transition: transform .18s ease, box-shadow .18s ease;
  }}
  .card:hover {{ transform: translateY(-2px); }}
  .card .val {{font-size: 1.65rem; font-weight: 700; margin: 0; color: {T["text"]}; letter-spacing: -0.02em;}}
  .card .lbl {{font-size: 0.78rem; color: {T["muted"]}; margin: 0; text-transform: uppercase; letter-spacing: 0.05em;}}

  /* Buttons feel like Apple */
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

  /* Tighten dataframes */
  [data-testid="stDataFrame"] {{border-radius: 12px; overflow: hidden;}}

  /* Radio chips */
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

    # Strip "0" placeholders from text cols (not real values)
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
    # Size & shape
    "gross_sqft", "land_sqft", "num_floors", "lot_area", "lot_depth",
    "lot_frontage", "building_depth", "building_frontage",
    # Unit mix (residential / commercial)
    "residential_units", "commercial_units", "total_units",
    # Tax & assessment
    "assess_total", "assess_land", "exempt_total",
    # Zoning (FAR family)
    "built_far", "resid_far", "comm_far", "facil_far",
    # Encoded categoricals (zoning + building class)
    "zoning_encoded", "building_class_code_encoded",
    # When it sold (month-of-year seasonality; year is all 2025 so dropped)
    "sale_month",
]
HERITAGE = [
    # Age & era
    "building_age", "construction_era_encoded",
    # Architect signal (prestige score + raw portfolio count)
    "architect_prestige_score", "architect_building_count",
    # Style rarity (rare score + raw frequency)
    "rare_style_score", "style_frequency",
    # Protection status
    "is_landmark", "in_historic_district",
    # Alteration history
    "is_altered", "years_since_alteration",
    # Facade + style
    "material_primary_encoded", "style_primary_encoded",
]

@st.cache_data
def prepare_features(dataframe):
    # Filter non-market transactions (donations, estate transfers, internal transfers)
    d = dataframe[dataframe["sale_price"] >= 100_000].copy()
    d["log_price"] = np.log1p(d["sale_price"])

    # Label-encode string categoricals we want in the model
    cat_cols = [
        "construction_era", "material_primary", "style_primary",
        "zoning", "building_class_code",
    ]
    for cat in cat_cols:
        if cat in d.columns:
            le = LabelEncoder()
            d[f"{cat}_encoded"] = le.fit_transform(d[cat].fillna("Unknown").astype(str))

    # Unit columns: missing usually means "none recorded", so impute zero not median
    for c in ["residential_units", "commercial_units", "total_units"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    all_f = BASELINE + HERITAGE
    avail = [f for f in all_f if f in d.columns]
    d[avail] = d[avail].apply(pd.to_numeric, errors="coerce")
    # Loose dropna: allow up to ~1/3 of features to be missing before we drop the row
    d = d.dropna(subset=avail, thresh=len(avail) - 6)
    for c in avail:
        d[c] = d[c].fillna(d[c].median())
    return d, [f for f in BASELINE if f in avail], [f for f in HERITAGE if f in avail]

mdf, FB, FH = prepare_features(df)
FALL = FB + FH


# ─────────────────────────────────────────────────────────────
# FRIENDLY LABELS
# Plain-English names for raw column names. Dropdowns/charts use these.
# ─────────────────────────────────────────────────────────────
FRIENDLY_LABELS = {
    # Structural / size
    "gross_sqft": "Gross Floor Area",
    "land_sqft": "Lot Size",
    "num_floors": "Number of Floors",
    "lot_area": "Lot Area",
    "lot_depth": "Lot Depth",
    "lot_frontage": "Lot Frontage",
    "building_depth": "Building Depth",
    "building_frontage": "Building Frontage",
    # Units
    "residential_units": "Residential Units",
    "commercial_units": "Commercial Units",
    "total_units": "Total Units",
    # Tax
    "assess_total": "Total Assessment Value",
    "assess_land": "Land Assessment Value",
    "exempt_total": "Tax Exemption Amount",
    # Zoning
    "built_far": "Built FAR",
    "resid_far": "Residential FAR Cap",
    "comm_far": "Commercial FAR Cap",
    "facil_far": "Facility FAR Cap",
    "zoning_encoded": "Zoning District",
    "building_class_code_encoded": "Building Class",
    # Time
    "sale_month": "Sale Month",
    "sale_price": "Sale Price",
    "price_per_sqft": "Price per Sqft",
    # Heritage
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

def flabel(var: str) -> str:
    """Return 'Plain English (raw_name)' for a variable, or just the raw name."""
    pretty = FRIENDLY_LABELS.get(var)
    return f"{pretty} ({var})" if pretty else var


# Manhattan trivia shown one at a time during model training. Keeps the wait fun.
MANHATTAN_FACTS = [
    "🏙️ The Empire State Building (1931) was built in just 410 days.",
    "🎨 Greenwich Village Historic District was designated in 1969 — NYC's largest.",
    "🏛️ NYC has 38,000+ landmark buildings across all five boroughs.",
    "🧱 'Brownstone' refers to a specific Triassic sandstone, mined in NJ.",
    "💡 The Flatiron Building (1902) was one of the first steel-framed skyscrapers.",
    "📜 The Landmarks Preservation Commission was created in 1965.",
    "🏠 Pre-war buildings (built before WWII) often command 10-20% premiums.",
    "🗽 The Plaza Hotel (1907) is a designated National Historic Landmark.",
    "🌆 Manhattan has 114 historic districts protecting 33,000+ buildings.",
    "🏗️ The first Manhattan landmark designated: Pieter Claesen Wyckoff House (1965).",
    "✏️ McKim, Mead & White designed over 940 buildings in their career.",
    "🏛️ Beaux-Arts style dominated Manhattan from 1885 to 1925.",
    "🎭 Carnegie Hall (1891) was funded entirely by Andrew Carnegie himself.",
    "⛪ St. Patrick's Cathedral took 21 years to build (1858-1879).",
    "🌃 The Chrysler Building (1930) held 'world's tallest' for just 11 months.",
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
    return {
        "R2_train": r2_score(ytr, model.predict(Xtr)),
        "R2_test": r2_score(yte, model.predict(Xte)),
        "MAE": mean_absolute_error(yte, model.predict(Xte)),
        "RMSE": np.sqrt(mean_squared_error(yte, model.predict(Xte))),
        "preds": model.predict(Xte),
        "model": model,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def wiki_lookup(query: str):
    """Best-effort Wikipedia hit: returns {title, url, extract, image} or None."""
    if not query or not query.strip():
        return None
    try:
        # Step 1: search for a matching page
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 1},
            timeout=4,
        ).json()
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]

        # Step 2: pull intro + image for that page
        p = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "prop": "pageimages|extracts",
                "exintro": 1, "explaintext": 1, "piprop": "original",
                "titles": title, "redirects": 1, "format": "json",
            },
            timeout=4,
        ).json()
        page = next(iter(p.get("query", {}).get("pages", {}).values()), {})
        img = page.get("original", {}).get("source")
        extract = (page.get("extract") or "").strip()
        return {
            "title": page.get("title", title),
            "url": f"https://en.wikipedia.org/wiki/{page.get('title', title).replace(' ', '_')}",
            "extract": extract[:500] + ("..." if len(extract) > 500 else ""),
            "image": img,
        }
    except Exception:
        return None


# =================================================================
# PAGE 1. BUSINESS CASE & DATA
# =================================================================
def page1():
    title("Manhattan Heritage Valuation",
          "How much does a building's architectural character affect its market price?")

    st.markdown("---")

    # ── Problem framing ──
    st.markdown("### The Research Question")
    st.markdown("""
Most real estate pricing models rely on **structural features** like square footage,
lot size, number of floors, zoning, and assessment values. These matter,
but they ignore *why certain buildings in Manhattan command extraordinary premiums*.

**Our approach.** We add a second layer of **architectural and preservation features**
(construction era, architect prestige, facade material, architectural style, landmark
status, historic district membership) to see if these create a measurable
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
# PAGE 2. VISUALIZATIONS & MAPS
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

    # ── 3D MAP ──
    if section == "Interactive Map":
        st.markdown("### 3D Property Map")
        st.caption("Each column is a historic property. Height and color encode whatever you pick. Drag to pan, scroll to zoom, right-drag to tilt.")

        c1, c2, c3 = st.columns(3)
        with c1:
            color_var = st.selectbox("Color by", [
                "sale_price", "price_per_sqft", "building_age",
                "architect_prestige_score", "num_floors", "assess_total",
            ], format_func=flabel,
               help="Each property's color encodes this value. The legend below the map shows the scale.")
        with c2:
            height_var = st.selectbox("Height by", [
                "sale_price", "gross_sqft", "num_floors", "building_age", "price_per_sqft",
            ], format_func=flabel,
               help="Taller column = higher value of this variable.")
        with c3:
            # Magma = dark purple to bright yellow, super readable on dark backgrounds.
            # Reordered so the most legible options come first.
            cmap_name = st.selectbox(
                "Color palette",
                ["magma", "inferno", "viridis", "plasma", "turbo", "cividis"],
                help="Magma & Inferno read best on dark mode; Viridis is colorblind-friendly.",
            )

        map_d = viz.dropna(subset=[color_var, height_var, "latitude", "longitude"]).copy()

        # Height: clip the top 3% so outliers don't dominate, scale to ~800m max
        h_vals = pd.to_numeric(map_d[height_var], errors="coerce")
        h_cap = h_vals.quantile(0.97)
        map_d["_h"] = (h_vals.clip(lower=0, upper=h_cap) / h_cap * 800).fillna(0)

        # Color: matplotlib colormap -> RGB list
        c_vals = pd.to_numeric(map_d[color_var], errors="coerce").fillna(map_d[color_var].median())
        norm = mcolors.Normalize(vmin=c_vals.quantile(0.02), vmax=c_vals.quantile(0.98))
        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(norm(c_vals.values))
        map_d["_r"] = (rgba[:, 0] * 255).astype(int)
        map_d["_g"] = (rgba[:, 1] * 255).astype(int)
        map_d["_b"] = (rgba[:, 2] * 255).astype(int)

        # Tooltip needs pre-formatted strings; pydeck can't format numbers
        map_d["price_fmt"] = map_d["sale_price"].apply(lambda v: f"${v:,.0f}")
        map_d["sqft_fmt"] = map_d["gross_sqft"].apply(
            lambda v: f"{v:,.0f} sqft" if pd.notna(v) else "unknown sqft")
        map_d["arch_fmt"] = map_d["architect"].fillna("Unknown architect")
        map_d["style_fmt"] = map_d["style_primary"].fillna("")
        map_d["era_fmt"] = map_d["construction_era"].fillna("")

        # Carto base layer, no mapbox token needed
        is_dark = T["plotly"] == "plotly_dark"
        pdk_style = "dark" if is_dark else "light"

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
            latitude=40.754, longitude=-73.987,
            zoom=12.2, pitch=50, bearing=20,
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
            layers=[layer], initial_view_state=view,
            map_provider="carto", map_style=pdk_style,
            tooltip=tooltip,
        )
        st.pydeck_chart(deck, use_container_width=True)

        # ── Color legend ──
        # pydeck won't render a colorbar on its own, so build one manually in CSS.
        # Format min/max in human units depending on which variable we colored by.
        c_min, c_max = c_vals.quantile(0.02), c_vals.quantile(0.98)
        def _fmt(v, varname):
            if "price" in varname or "assess" in varname:
                return f"${v:,.0f}"
            if "age" in varname:
                return f"{v:,.0f} yrs"
            return f"{v:,.0f}"

        # Build a CSS gradient string from the chosen matplotlib colormap (10 stops)
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
        st.caption(f"Low {pretty_color} on the left, high on the right. "
                   f"Taller columns = higher {pretty_height}.")

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

        # ── Architect Value Leaderboard ──
        st.markdown("---")
        st.markdown("### Architect Value Leaderboard")
        st.caption("Which architects' buildings hold the most value? Sortable table — click any column header.")

        min_n = st.slider("Minimum buildings per architect", 3, 25, 5, key="arch_minN")

        # Drop non-market transactions (internal transfers, easements) that would
        # tank the baseline. Manhattan reality: anything under $50/sqft isn't a real sale.
        market = viz[(viz["sale_price"] >= 100_000) & (viz["price_per_sqft"] >= 50)].copy()
        city_med_psf = market["price_per_sqft"].median()

        # Build per-architect stats
        g = (market.dropna(subset=["architect"])
                .groupby("architect")
                .agg(buildings=("BBL", "count"),
                     median_price=("sale_price", "median"),
                     median_psf=("price_per_sqft", "median"),
                     landmark_share=("is_landmark", "mean"),
                     prestige=("architect_prestige_score", "mean"),
                     hist_dist_share=("in_historic_district", "mean"))
                .reset_index())
        g = g[g["buildings"] >= min_n].copy()
        # Heritage premium = how much above city-wide $/sqft, in percent
        g["heritage_premium_pct"] = (g["median_psf"] / city_med_psf - 1) * 100
        g = g.sort_values("heritage_premium_pct", ascending=False)
        g.insert(0, "Rank", range(1, len(g) + 1))

        leaderboard = g.rename(columns={
            "architect": "Architect",
            "buildings": "# Buildings",
            "median_price": "Median Sale Price",
            "median_psf": "Median $/SqFt",
            "heritage_premium_pct": "Heritage Premium %",
            "landmark_share": "% Landmarked",
            "hist_dist_share": "% in Hist. Dist.",
            "prestige": "Prestige Score",
        })

        st.dataframe(
            leaderboard.style.format({
                "Median Sale Price": "${:,.0f}",
                "Median $/SqFt": "${:,.0f}",
                "Heritage Premium %": "{:+.1f}%",
                "% Landmarked": "{:.0%}",
                "% in Hist. Dist.": "{:.0%}",
                "Prestige Score": "{:.2f}",
            }).background_gradient(subset=["Heritage Premium %"], cmap="RdYlGn"),
            use_container_width=True,
            height=460,
        )

        # Quick top-5 highlight cards
        top5 = g.head(5)
        st.markdown("#### Top 5 Value Premium Architects")
        ccols = st.columns(5)
        for col, (_, r) in zip(ccols, top5.iterrows()):
            col.markdown(
                f'<div class="card">'
                f'<p class="lbl">#{int(r["Rank"])} &middot; {int(r["buildings"])} bldgs</p>'
                f'<p class="val" style="font-size:1.05rem; line-height:1.2;">{r["architect"]}</p>'
                f'<p class="lbl" style="margin-top:0.4rem;">+{r["heritage_premium_pct"]:.0f}% vs city</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

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

        # ── Sankey: Era → Style → Price Tier ──
        # How architectural choices in each era translate to modern price brackets
        st.markdown("---")
        st.markdown("### Era → Style → Price Tier Flow")
        st.caption("Follow each construction era through its dominant styles into today's price tiers. Width = number of properties.")

        sk = viz.dropna(subset=["construction_era", "style_primary", "sale_price"]).copy()
        # Keep only top 8 styles by frequency to avoid a hairball
        top_styles = sk["style_primary"].value_counts().head(8).index
        sk = sk[sk["style_primary"].isin(top_styles)]
        # Bucket price into 4 quartile tiers with readable labels
        q = sk["sale_price"].quantile([0.25, 0.5, 0.75]).values
        def tier(p):
            if p < q[0]: return "Tier 4: Budget"
            if p < q[1]: return "Tier 3: Mid"
            if p < q[2]: return "Tier 2: High"
            return "Tier 1: Top"
        sk["price_tier"] = sk["sale_price"].apply(tier)

        eras = sorted(sk["construction_era"].unique().tolist())
        styles = sk["style_primary"].value_counts().index.tolist()
        tiers = ["Tier 1: Top", "Tier 2: High", "Tier 3: Mid", "Tier 4: Budget"]
        nodes = eras + styles + tiers
        idx = {n: i for i, n in enumerate(nodes)}

        # Era -> Style edges
        es = sk.groupby(["construction_era", "style_primary"]).size().reset_index(name="n")
        # Style -> Tier edges
        st_t = sk.groupby(["style_primary", "price_tier"]).size().reset_index(name="n")

        sources = ([idx[e] for e in es["construction_era"]]
                   + [idx[s] for s in st_t["style_primary"]])
        targets = ([idx[s] for s in es["style_primary"]]
                   + [idx[t] for t in st_t["price_tier"]])
        values = list(es["n"]) + list(st_t["n"])

        # Color tier nodes from red->green by tier rank
        tier_colors = {"Tier 1: Top": "#16a34a", "Tier 2: High": "#84cc16",
                       "Tier 3: Mid": "#facc15", "Tier 4: Budget": "#ef4444"}
        node_colors = (["#94a3b8"] * len(eras)
                       + [T["accent"]] * len(styles)
                       + [tier_colors[t] for t in tiers])

        fig_sk = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(label=nodes, pad=14, thickness=16,
                      line=dict(color="rgba(0,0,0,0.1)", width=0.5),
                      color=node_colors),
            link=dict(source=sources, target=targets, value=values,
                      color="rgba(148,163,184,0.35)"),
        ))
        fig_sk.update_layout(template=T["plotly"], height=560,
                             margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_sk, use_container_width=True)

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
# PAGE 3. PREDICTION MODELS
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
    idx_all = np.arange(len(mdf))  # row index lookup so we can recover original rows

    Xb_tr, Xb_te, y_tr, y_te, _, te_idx = train_test_split(
        X_b, y, idx_all, test_size=test_sz, random_state=rs)
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
        "Elastic Net":        (ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=rs), True),
        "Decision Tree":      (DecisionTreeRegressor(max_depth=10, random_state=rs), False),
        "Random Forest":      (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1), False),
        "Gradient Boosting":  (GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=rs), False),
    }
    if HAS_LGBM:
        # Cut from 300 to 150 trees + bump learning_rate to keep R2 the same
        # Roughly halves training time on our 2.8K-row dataset
        MODELS["LightGBM"] = (
            lgb.LGBMRegressor(n_estimators=150, num_leaves=31, learning_rate=0.08,
                              random_state=rs, n_jobs=-1, verbose=-1),
            False,
        )
    if HAS_CATBOOST:
        # CatBoost: trim iterations + parallelize for snappy training
        MODELS["CatBoost"] = (
            CatBoostRegressor(iterations=300, depth=6, learning_rate=0.08,
                              random_seed=rs, verbose=0, allow_writing_files=False,
                              thread_count=-1),
            False,
        )

    st.markdown("---")
    st.markdown("### Select models to train")
    # 4 per row so the checkbox list stays tidy as the list grows
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

    # Pick a different fact per model so user has something fun to read
    import random
    facts_pool = random.sample(MANHATTAN_FACTS, k=min(len(sel), len(MANHATTAN_FACTS)))

    # Live UI: progress bar + status + rotating fact + live leaderboard chart
    progress = st.progress(0, text="Starting…")
    status = st.empty()
    fact_box = st.empty()
    leaderboard_box = st.empty()
    total_t0 = time.perf_counter()

    for i, nm in enumerate(sel, start=1):
        # Status line + a fun Manhattan fact while user waits
        status.markdown(
            f"&nbsp;&nbsp;**Training {i}/{len(sel)}: {nm}**"
            f" &middot; <span style='color:#6E6E73'>baseline + heritage</span>",
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

        base_rows.append({"Model": nm, **{k: v for k, v in rb.items() if k not in ("preds", "model")}, "Time(s)": round(elapsed, 2)})
        herit_rows.append({"Model": nm, **{k: v for k, v in rh.items() if k not in ("preds", "model")}, "Time(s)": round(elapsed, 2)})
        herit_rows[-1]["preds"] = rh["preds"]
        trained_h[nm] = mh

        progress.progress(i / len(sel), text=f"Done {i}/{len(sel)} · {nm} took {elapsed:.1f}s")

        # Live mini-leaderboard: refreshes after every model, shows current standings
        live_lb = pd.DataFrame(herit_rows)[["Model", "R2_test"]].sort_values("R2_test", ascending=True)
        is_winner = live_lb["R2_test"] == live_lb["R2_test"].max()
        bar_colors = [T["accent"] if w else "#94a3b8" for w in is_winner]
        live_fig = go.Figure(go.Bar(
            x=live_lb["R2_test"], y=live_lb["Model"], orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in live_lb["R2_test"]],
            textposition="outside",
        ))
        live_fig.update_layout(
            template=T["plotly"], height=max(220, 38 * len(live_lb) + 80),
            margin=dict(l=10, r=30, t=30, b=10),
            xaxis_title="R² (Test, Heritage features)",
            title=dict(text=f"🏆 Live Leaderboard — leader: {live_lb.iloc[-1]['Model']}",
                       x=0.02, font=dict(size=14)),
            xaxis=dict(range=[0, max(0.7, live_lb["R2_test"].max() * 1.1)]),
        )
        leaderboard_box.plotly_chart(live_fig, use_container_width=True, key=f"live_lb_{i}")

    total_elapsed = time.perf_counter() - total_t0
    progress.empty()
    fact_box.empty()
    status.success(f"✅ Trained {len(sel)} model{'s' if len(sel)>1 else ''} in {total_elapsed:.1f}s "
                   f"(slowest: {max(timings, key=timings.get)} at {max(timings.values()):.1f}s)")
    # A tiny celebration so finishing feels satisfying
    st.balloons()

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
    fmt_cols = ["R2_train", "R2_test", "MAE", "RMSE", "Time(s)"]
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
        f"Best Heritage Model: **{best['Model']}**. "
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

    # ── Residual Analysis ──
    # Where does the model agree with the market, and where doesn't it?
    st.markdown("---")
    st.markdown("### Where the Model Disagrees with the Market")
    st.caption("Residuals = actual log-price minus predicted log-price. "
               "Positive = market paid more than model expected (potentially undervalued by the model, or 'trophy premium'). "
               "Negative = market paid less than model expected (potentially overvalued by the model, or distressed sale).")

    # Recover the original row info for each test sample using te_idx
    test_rows = mdf.iloc[te_idx].copy().reset_index(drop=True)
    test_rows["actual_price"] = np.expm1(y_te)
    test_rows["pred_price"] = np.expm1(yp)
    test_rows["residual_log"] = y_te - yp
    test_rows["residual_pct"] = (test_rows["actual_price"] / test_rows["pred_price"] - 1) * 100
    test_rows["abs_dollar_diff"] = test_rows["actual_price"] - test_rows["pred_price"]

    # Map: each property colored by residual sign and sized by magnitude.
    # Using a custom diverging palette: vivid red for discount, vivid green for premium,
    # neutral cream in the middle. Clearer than RdBu on dark backgrounds.
    map_d = test_rows.dropna(subset=["latitude", "longitude"]).copy()
    diverging = [
        [0.0, "#dc2626"],   # market paid less (model thinks it's worth more)
        [0.25, "#f87171"],
        [0.5, "#fef3c7"],   # right on prediction
        [0.75, "#4ade80"],
        [1.0, "#15803d"],   # market paid more (trophy premium)
    ]
    fig_map = px.scatter_mapbox(
        map_d,
        lat="latitude", lon="longitude",
        color="residual_pct",
        color_continuous_scale=diverging, color_continuous_midpoint=0,
        range_color=[-80, 80],
        size=map_d["residual_pct"].abs().clip(5, 150),
        size_max=24,
        hover_name="display_name",
        hover_data={
            "actual_price": ":$,.0f", "pred_price": ":$,.0f",
            "residual_pct": ":+.1f", "architect": True,
            "construction_era": True,
            "latitude": False, "longitude": False,
        },
        zoom=11.5, height=520, template=T["plotly"],
        mapbox_style=T["map_style"],
        labels={"residual_pct": "Residual %"},
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Residual %",
            ticksuffix="%",
            tickvals=[-80, -40, 0, 40, 80],
            ticktext=["−80%<br>Discount", "−40%", "0%<br>On model", "+40%", "+80%<br>Premium"],
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("🔴 **Red = Discount** (market paid less than model expected) &nbsp;·&nbsp; "
               "🟢 **Green = Premium** (market paid more) &nbsp;·&nbsp; "
               "Bigger dot = bigger miss.")

    # ── Top 10 over/under priced ──
    st.markdown("### Biggest Misses by the Model")
    show_cols = ["display_name", "architect", "construction_era",
                 "actual_price", "pred_price", "residual_pct", "neighborhood"]
    rename = {
        "display_name": "Property", "architect": "Architect",
        "construction_era": "Era", "actual_price": "Actual",
        "pred_price": "Predicted", "residual_pct": "Diff %",
        "neighborhood": "Neighborhood",
    }
    fmt = {"Actual": "${:,.0f}", "Predicted": "${:,.0f}", "Diff %": "{:+.1f}%"}

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 10 'Trophy Premium'** (market paid more than model)")
        st.caption("Likely heritage/architect/view premium not fully captured.")
        top_under = (test_rows.sort_values("residual_pct", ascending=False)
                              .head(10)[show_cols].rename(columns=rename))
        st.dataframe(top_under.style.format(fmt), use_container_width=True, height=380)
    with c2:
        st.markdown("**Top 10 'Discount Sales'** (market paid less than model)")
        st.caption("Likely distressed sales, internal transfers, or condition issues.")
        top_over = (test_rows.sort_values("residual_pct", ascending=True)
                             .head(10)[show_cols].rename(columns=rename))
        st.dataframe(top_over.style.format(fmt), use_container_width=True, height=380)


# =================================================================
# PAGE 4. FEATURE IMPORTANCE
# =================================================================
def page4():
    title("Feature Importance & Explainability",
          "Which features drive the Heritage model: baseline structural or preservation characteristics?")

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

        # Use friendly names in the chart so non-technical reader gets it instantly
        fi = pd.DataFrame({
            "Feature": [flabel(f) for f in fnames],
            "Importance": imp,
            "Type": ["Baseline" if f in FB else "Heritage" for f in fnames],
        }).sort_values("Importance", ascending=True)

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
                "Feature": [flabel(f) for f in fnames],
                "Coefficient": model.coef_,
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
            shap.summary_plot(sv, Xa_te[:250], feature_names=[flabel(f) for f in fnames], show=False, max_display=15)
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
                                           data=Xa_te[idx], feature_names=[flabel(f) for f in fnames])
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
    title("Hyperparameter Tuning",
          "Grid search with optional Weights & Biases experiment tracking")

    # ── Beginner intro: what's hyperparameter tuning, in plain English ──
    with st.expander("📚 New here? What is hyperparameter tuning?", expanded=False):
        st.markdown("""
**TL;DR:** Hyperparameters are the dials *you* set before training. The model learns
its own weights *from* the data, but it doesn't pick its own dials. Tuning means
trying many dial combinations and keeping the one that works best on held-out data.

- **Grid search** = try every combination you list. Simple and exhaustive.
- **Test size** = how much of the data to hold out for scoring (20% is standard).
- **R²** = "how much of the price variance does the model explain?" Higher is better, max is 1.
- **MAE** = average dollar error. Lower is better.

**Picking a model:**
- *Linear (Ridge / Lasso / Elastic Net)* → fast, interpretable, but caps out at R² ~0.20 on this data.
- *Decision Tree* → great for explaining a single decision path, weak alone.
- *Random Forest / Gradient Boosting* → strong all-rounders, usually R² ~0.55-0.60.
- *LightGBM* → the fastest of the heavy hitters, usually wins.

Start with **LightGBM** if you want best results fast, or **Ridge** if you want to
understand which features matter linearly.
        """)

    X = mdf[FALL].values
    y = mdf["log_price"].values

    tune_choices = [
        "Ridge Regression", "Lasso Regression", "Elastic Net",
        "Decision Tree", "Random Forest", "Gradient Boosting",
    ]
    if HAS_LGBM:
        tune_choices.append("LightGBM")

    c1, c2 = st.columns(2)
    with c1:
        model_name = st.selectbox("Model", tune_choices,
                                  help="Pick a model family. Different models expose different dials.")
    with c2:
        test_sz = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="hp_ts",
                            help="Fraction of data held out for scoring. 20% is the standard. "
                                 "Smaller = more training data but noisier scores.")

    # ── Per-model tutorial. Subtle but always there for someone new. ──
    MODEL_TUTORIALS = {
        "Ridge Regression": {
            "what": "Linear regression with **L2 penalty** — shrinks coefficients toward zero so big features don't dominate. Good when many features are mildly useful.",
            "params": [
                ("Alpha", "Regularization strength. ↑ alpha = simpler model (more shrinkage, less overfitting). ↓ alpha = closer to plain linear regression. Try `0.01, 0.1, 1, 10, 100` to see the curve."),
            ],
            "watch": "Train R² and Test R² both stable across alpha → model isn't overfitting much. Diverging → too little regularization.",
        },
        "Lasso Regression": {
            "what": "Linear regression with **L1 penalty** — actually drives some coefficients to **exactly zero**, doing automatic feature selection.",
            "params": [
                ("Alpha", "Regularization strength. ↑ alpha = more features killed off. ↓ alpha = more features survive. Watch how many coefficients hit zero as alpha grows."),
            ],
            "watch": "If R² stays fine even at high alpha, the surviving features carry all the signal — the rest were noise.",
        },
        "Elastic Net": {
            "what": "**Mix of Ridge + Lasso.** Ridge for stability, Lasso for sparsity.",
            "params": [
                ("Alpha", "Total regularization strength."),
                ("L1 ratio", "0 = pure Ridge, 1 = pure Lasso. 0.5 = even mix. Use this when you can't decide between Ridge and Lasso."),
            ],
            "watch": "Best when Ridge and Lasso both give decent but imperfect scores — Elastic Net often beats both.",
        },
        "Decision Tree": {
            "what": "Splits data into yes/no questions, ending in price predictions. Easy to interpret as a flow chart.",
            "params": [
                ("Max Depth", "How many questions deep the tree can go. ↑ depth = more complex (risks overfitting). ↓ depth = simpler. `None` = unlimited."),
                ("Min Samples Split", "Minimum samples needed to split a node. ↑ value = more conservative tree, less overfitting. Try 2, 10, 20."),
            ],
            "watch": "Big gap between Train R² (high) and Test R² (low) = overfit. Lower max_depth or raise min_samples_split.",
        },
        "Random Forest": {
            "what": "Many decision trees voting together. Each tree sees a random slice of data and features. Robust by design.",
            "params": [
                ("N Estimators", "Number of trees. More trees = more stable predictions but slower. Diminishing returns past ~200."),
                ("Max Depth", "How deep each tree can grow. Deeper = capture more, but trees memorize. `None` lets each tree grow to perfect fit on its sample."),
            ],
            "watch": "Random Forest rarely overfits even with deep trees, because the randomness averages out. Tune n_estimators for stability, max_depth for speed.",
        },
        "Gradient Boosting": {
            "what": "Trees added **one at a time**, each new tree fixing what the previous trees got wrong. Powerful but slower and more sensitive than Random Forest.",
            "params": [
                ("Learning Rate", "How much each new tree gets to fix. ↓ rate + ↑ trees = slow but accurate. ↑ rate + ↓ trees = fast but rough. Classic combos: `(0.01, 500)` or `(0.1, 100)`."),
                ("N Estimators", "Number of boosting rounds (trees added)."),
                ("Max Depth", "Each tree's depth. 3-7 is the sweet spot — boosting prefers many shallow trees over a few deep ones."),
            ],
            "watch": "If train R² ≫ test R², you're overfitting — drop max_depth, lower learning rate, or fewer estimators.",
        },
        "LightGBM": {
            "what": "Modern gradient boosting library. Faster and usually more accurate than sklearn's Gradient Boosting.",
            "params": [
                ("N Estimators", "Number of boosting rounds. Pair with learning rate."),
                ("Num Leaves", "Max leaves per tree. Controls tree complexity. ↑ leaves = more flexible, risks overfit. Sweet spot 31-63 for most data."),
                ("Learning Rate", "Step size per tree. Lower = more careful learning. Combine 0.05 + 200 trees for best results, or 0.1 + 100 for speed."),
            ],
            "watch": "LightGBM trains in seconds. Try wide grids without fear. The default num_leaves=31 is usually fine.",
        },
    }

    if model_name in MODEL_TUTORIALS:
        tut = MODEL_TUTORIALS[model_name]
        with st.expander(f"💡 How **{model_name}** works & how to tune it", expanded=False):
            st.markdown(f"**What it does:** {tut['what']}")
            st.markdown("**Parameters:**")
            for pname, ptext in tut["params"]:
                st.markdown(f"- **{pname}** — {ptext}")
            st.info(f"**What to watch:** {tut['watch']}")

    st.markdown("---")
    st.markdown("### Hyperparameter Grid")
    grid = []

    if model_name == "Ridge Regression":
        alphas = st.multiselect("Alpha", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                                default=[0.01, 0.1, 1.0, 10.0, 100.0],
                                help="Higher alpha = more shrinkage on coefficients = simpler model.")
        grid = [{"alpha": a} for a in alphas]

    elif model_name == "Lasso Regression":
        alphas = st.multiselect("Alpha", [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                                default=[0.001, 0.01, 0.1, 0.5],
                                help="Higher alpha = more coefficients pushed exactly to zero.")
        grid = [{"alpha": a} for a in alphas]

    elif model_name == "Elastic Net":
        alphas = st.multiselect("Alpha", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                                default=[0.01, 0.05, 0.1],
                                help="Total regularization strength.")
        l1s = st.multiselect("L1 ratio", [0.1, 0.3, 0.5, 0.7, 0.9],
                             default=[0.3, 0.5, 0.7],
                             help="0 = pure Ridge, 1 = pure Lasso, 0.5 = even mix.")
        grid = [{"alpha": a, "l1_ratio": l} for a in alphas for l in l1s]

    elif model_name == "LightGBM":
        nes = st.multiselect("N Estimators", [100, 200, 400, 800], default=[200, 400],
                             help="Number of boosting trees. Pair with learning rate.")
        leaves = st.multiselect("Num Leaves", [15, 31, 63, 127], default=[31, 63],
                                help="Max leaves per tree. Higher = more flexible, risks overfit.")
        lrs = st.multiselect("Learning Rate", [0.01, 0.05, 0.1], default=[0.05, 0.1],
                             help="Step size per tree. Lower = more careful, needs more trees.")
        grid = [{"n_estimators": n, "num_leaves": lv, "learning_rate": lr}
                for n in nes for lv in leaves for lr in lrs]

    elif model_name == "Decision Tree":
        depths = st.multiselect("Max Depth", [2, 3, 5, 7, 10, 15, None], default=[3, 5, 10],
                                help="How many yes/no splits deep the tree can go. None = unlimited.")
        mins = st.multiselect("Min Samples Split", [2, 5, 10, 20], default=[2, 5, 10],
                              help="Min samples needed to split a node. Higher = more conservative.")
        grid = [{"max_depth": d, "min_samples_split": s} for d in depths for s in mins]

    elif model_name == "Random Forest":
        nes = st.multiselect("N Estimators", [50, 100, 200, 300], default=[50, 100, 200],
                             help="Number of trees in the forest. More = stable, slower.")
        deps = st.multiselect("Max Depth", [5, 10, 15, None], default=[5, 10, 15],
                              help="Each tree's depth. None = grow until each leaf is pure.")
        grid = [{"n_estimators": n, "max_depth": d} for n in nes for d in deps]

    else:  # Gradient Boosting
        lrs = st.multiselect("Learning Rate", [0.01, 0.05, 0.1, 0.2], default=[0.01, 0.1],
                             help="How much each tree fixes. Lower + more trees usually wins.")
        nes = st.multiselect("N Estimators", [50, 100, 200], default=[50, 100],
                             help="Number of boosting rounds.")
        deps = st.multiselect("Max Depth", [3, 5, 7], default=[3, 5],
                              help="Per-tree depth. Boosting likes shallow trees (3-7).")
        grid = [{"learning_rate": lr, "n_estimators": n, "max_depth": d}
                for lr in lrs for n in nes for d in deps]

    st.caption(f"Total experiments: **{len(grid)}** — each combination trains a fresh model and is logged separately.")

    # ── W&B setup ──
    st.markdown("---")
    st.markdown("### Weights & Biases (Optional)")
    use_wb = st.checkbox("Log experiments to W&B", False)
    wb_proj = "manhattan-heritage-pricing"
    wb_entity = ""
    wb_key = ""
    if use_wb:
        wc1, wc2 = st.columns(2)
        with wc1:
            wb_proj = st.text_input("Project name", "manhattan-heritage-pricing")
        with wc2:
            wb_entity = st.text_input("Entity / team (optional)", "",
                                      placeholder="e.g. mumu031122-new-york-university")
        wb_key = st.text_input(
            "API key (only if `wandb login` is not set up)", "",
            type="password",
            help="Leave empty if you already ran `wandb login` in your terminal. Otherwise paste from wandb.ai/authorize.",
        )
        st.caption(
            "If you already ran `wandb login` in your terminal, leave the key empty. "
            "Pasting here just runs `wandb.login(key=...)` once for this session."
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
        needs_scale = model_name in ["Ridge Regression", "Lasso Regression", "Elastic Net"]

        # Try to set up W&B once before the loop, not inside it
        wb_ok = False
        wandb_mod = None
        if use_wb:
            try:
                import wandb as wandb_mod
                if wb_key.strip():
                    wandb_mod.login(key=wb_key.strip(), relogin=True)
                # Probe: does this session have credentials?
                api_key_present = bool(wandb_mod.api.api_key)
                if not api_key_present:
                    st.error("W&B has no API key in this session. Run `wandb login` in your terminal, "
                             "or paste your key in the field above.")
                else:
                    wb_ok = True
            except ImportError:
                st.warning("wandb not installed. Logging locally only.")
            except Exception as e:
                st.error(f"W&B login failed: {e}")

        # Short tag for run names: model + first param values
        def short_tag(p):
            bits = []
            for k, v in p.items():
                if v is None: v = "auto"
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

            if model_name == "Ridge Regression":       m = Ridge(**params)
            elif model_name == "Lasso Regression":     m = Lasso(**params)
            elif model_name == "Elastic Net":          m = ElasticNet(**params, random_state=42)
            elif model_name == "Decision Tree":        m = DecisionTreeRegressor(**params, random_state=42)
            elif model_name == "Random Forest":        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            elif model_name == "Gradient Boosting":    m = GradientBoostingRegressor(**params, random_state=42)
            else:                                      m = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)

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
                    if wb_errors <= 2:  # only show first 2 errors so we don't spam
                        st.warning(f"W&B logging failed for run {i}: {e}")

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
            f"RMSE = {best['rmse']:.4f}. Params: {dict(best[pcols])}"
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
            ent_part = f"{wb_entity.strip()}/" if wb_entity.strip() else ""
            url = f"https://wandb.ai/{ent_part}{wb_proj}"
            st.success(f"Logged {len(grid) - wb_errors}/{len(grid)} runs. [Open dashboard ↗]({url})")


# =================================================================
# PAGE 6. PROPERTY VALUATOR
# Pick a building, see baseline vs heritage prediction, and why.
# =================================================================
@st.cache_resource
def get_valuation_models():
    """Train baseline + heritage GBR once, keep in memory across reruns."""
    X_b = mdf[FB].values
    X_a = mdf[FALL].values
    y = mdf["log_price"].values
    Xb_tr, _, y_tr, _ = train_test_split(X_b, y, test_size=0.2, random_state=42)
    Xa_tr, _, _, _   = train_test_split(X_a, y, test_size=0.2, random_state=42)
    mb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42).fit(Xb_tr, y_tr)
    mh = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42).fit(Xa_tr, y_tr)
    return mb, mh, Xa_tr


@st.cache_resource
def get_quantile_models():
    """Train LGBM at 10/50/90 percentiles for prediction intervals. None if lgb missing."""
    if not HAS_LGBM:
        return None
    X_a = mdf[FALL].values
    y   = mdf["log_price"].values
    Xa_tr, _, y_tr, _ = train_test_split(X_a, y, test_size=0.2, random_state=42)
    out = {}
    for q in (0.1, 0.5, 0.9):
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbose=-1,
        ).fit(Xa_tr, y_tr)
        out[q] = m
    return out


def page6():
    title("Property Valuator",
          "Pick a historic building. Compare what basic structural features alone predict versus the full heritage model. Then see which features actually drove that number.")

    with st.spinner("Warming up valuation models..."):
        mb, mh, bg_ref = get_valuation_models()

    pool = mdf.dropna(subset=FALL).copy()

    # ── Search & filter ──
    st.markdown("### Find a property")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        query = st.text_input("Search address, building name, or architect", "").strip().lower()
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
        return f"{nm}  ·  {arch}  ·  ${r['sale_price']:,.0f}"

    pick_idx = st.selectbox("Pick a property", subset.index[:200].tolist(), format_func=label)
    row = subset.loc[pick_idx]

    # ── Predict (log space, then back to dollars) ──
    x_base = row[FB].values.reshape(1, -1).astype(float)
    x_all  = row[FALL].values.reshape(1, -1).astype(float)
    pred_base = float(np.expm1(mb.predict(x_base)[0]))
    pred_her  = float(np.expm1(mh.predict(x_all)[0]))
    actual    = float(row["sale_price"])
    premium = pred_her - pred_base
    premium_pct = premium / pred_base * 100 if pred_base else 0
    gap = actual - pred_her
    gap_pct = gap / pred_her * 100 if pred_her else 0

    # ── Hero cards ──
    st.markdown("---")
    good = T["accent"]
    bad = "#ef4444"
    premium_color = good if premium >= 0 else bad

    cols = st.columns(4)
    cols[0].markdown(
        f'<div class="card"><p class="val">${actual:,.0f}</p>'
        f'<p class="lbl">Actual sale price</p></div>', unsafe_allow_html=True)
    cols[1].markdown(
        f'<div class="card"><p class="val">${pred_base:,.0f}</p>'
        f'<p class="lbl">Baseline prediction</p></div>', unsafe_allow_html=True)
    cols[2].markdown(
        f'<div class="card"><p class="val">${pred_her:,.0f}</p>'
        f'<p class="lbl">Heritage prediction</p></div>', unsafe_allow_html=True)
    cols[3].markdown(
        f'<div class="card"><p class="val" style="color:{premium_color}">'
        f'{premium:+,.0f}</p>'
        f'<p class="lbl">Heritage premium ({premium_pct:+.1f}%)</p></div>',
        unsafe_allow_html=True)

    # ── Verdict line ──
    if abs(gap_pct) < 10:
        verdict = f"Close to heritage fair value ({gap_pct:+.1f}%)."
        st.success(verdict + f" Market paid ${abs(gap):,.0f} {'above' if gap > 0 else 'below'} model.")
    elif gap_pct > 0:
        st.info(f"Sold **above** heritage model by {gap_pct:+.1f}% (${abs(gap):,.0f}). Could be unique features the model misses, or a hot market moment.")
    else:
        st.warning(f"Sold **below** heritage model by {gap_pct:+.1f}% (${abs(gap):,.0f}). Could be a distressed sale, condition issues, or a bargain.")

    # ── LGBM quantile interval (80% band) ──
    qm = get_quantile_models()
    if qm is not None:
        q_low  = float(np.expm1(qm[0.1].predict(x_all)[0]))
        q_med  = float(np.expm1(qm[0.5].predict(x_all)[0]))
        q_high = float(np.expm1(qm[0.9].predict(x_all)[0]))
        inside = q_low <= actual <= q_high

        st.markdown("---")
        st.markdown("### Prediction interval")
        st.caption("LightGBM quantile regression at the 10th, 50th, and 90th percentiles. The band covers where 80% of buildings with these features sell.")

        # Horizontal range chart: low to high band + median dot + actual marker
        dot_color = T["accent"]
        actual_color = "#10b981" if inside else "#ef4444"

        fig = go.Figure()
        # The band
        fig.add_trace(go.Scatter(
            x=[q_low, q_high], y=[0, 0], mode="lines",
            line=dict(color=T["accent"], width=18),
            opacity=0.35, hoverinfo="skip", showlegend=False,
        ))
        # Endpoint ticks
        fig.add_trace(go.Scatter(
            x=[q_low, q_high], y=[0, 0], mode="markers+text",
            marker=dict(size=10, color=T["accent"]),
            text=[f"P10<br>${q_low:,.0f}", f"P90<br>${q_high:,.0f}"],
            textposition=["top left", "top right"],
            hoverinfo="skip", showlegend=False,
        ))
        # Median
        fig.add_trace(go.Scatter(
            x=[q_med], y=[0], mode="markers+text",
            marker=dict(size=20, color=dot_color, line=dict(color="white", width=2)),
            text=[f"Model median<br>${q_med:,.0f}"], textposition="bottom center",
            name="Median (P50)",
        ))
        # Actual
        fig.add_trace(go.Scatter(
            x=[actual], y=[0], mode="markers+text",
            marker=dict(size=22, color=actual_color, symbol="diamond",
                        line=dict(color="white", width=2)),
            text=[f"Actual<br>${actual:,.0f}"], textposition="bottom center",
            name="Actual sale",
        ))
        pad = max(q_high - q_low, 1) * 0.15
        fig.update_layout(
            template=T["plotly"], height=230,
            xaxis=dict(tickformat="$,.0f",
                       range=[min(q_low, actual) - pad, max(q_high, actual) + pad]),
            yaxis=dict(visible=False, range=[-1, 1]),
            margin=dict(l=20, r=20, t=40, b=20), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        if inside:
            st.caption(f"Actual price sits inside the 80% band. Consistent with comparable buildings.")
        elif actual > q_high:
            st.caption(f"Actual price is above the P90 by ${actual - q_high:,.0f}. Top decile outcome.")
        else:
            st.caption(f"Actual price is below the P10 by ${q_low - actual:,.0f}. Bottom decile outcome.")

    # ── Wikipedia spotlight (best-effort) ──
    bname = str(row.get("building_name") or "").strip()
    arch  = str(row.get("architect") or "").strip()
    wiki = None
    if bname and bname.lower() not in ("0", "none", "nan"):
        wiki = wiki_lookup(bname)
    if (not wiki or not wiki.get("image")) and arch and arch.lower() not in ("unknown", "0"):
        arch_wiki = wiki_lookup(arch)
        if arch_wiki and arch_wiki.get("image"):
            wiki = arch_wiki

    if wiki and (wiki.get("image") or wiki.get("extract")):
        st.markdown("---")
        st.markdown("### Heritage spotlight")
        wc1, wc2 = st.columns([1, 2])
        with wc1:
            if wiki.get("image"):
                st.image(wiki["image"], use_container_width=True)
            else:
                st.caption("No photo found on Wikipedia.")
        with wc2:
            st.markdown(f"**{wiki['title']}**  ·  [Open on Wikipedia ↗]({wiki['url']})")
            if wiki.get("extract"):
                st.markdown(f"<p style='color:{T['muted']};line-height:1.55;'>{wiki['extract']}</p>",
                            unsafe_allow_html=True)

    # ── Building profile + mini-map ──
    st.markdown("---")
    st.markdown("### Building profile")
    pc1, pc2 = st.columns([1, 1])

    with pc1:
        def fmt(v, kind="str"):
            if pd.isna(v) or v == "" or v == 0:
                return "Unknown"
            if kind == "int":   return f"{int(v):,}"
            if kind == "year":  return f"{int(v)}"
            if kind == "sqft":  return f"{v:,.0f}"
            return str(v)

        info = [
            ("Address",          fmt(row.get("address"))),
            ("Building name",    fmt(row.get("building_name"))),
            ("Architect",        fmt(row.get("architect"))),
            ("Style",            fmt(row.get("style_primary"))),
            ("Facade material",  fmt(row.get("material_primary"))),
            ("Era",              fmt(row.get("construction_era"))),
            ("Year built",       fmt(row.get("construction_year"), "year")),
            ("Floors",           fmt(row.get("num_floors"), "int")),
            ("Gross sqft",       fmt(row.get("gross_sqft"), "sqft")),
            ("Neighborhood",     fmt(row.get("neighborhood"))),
            ("Landmark",         "Yes" if row.get("is_landmark") == 1 else "No"),
            ("Historic district","Yes" if row.get("in_historic_district") == 1 else "No"),
            ("Altered since",    fmt(row.get("last_alteration_year"), "year") if row.get("is_altered") == 1 else "Original"),
        ]
        info_df = pd.DataFrame(info, columns=["Field", "Value"])
        st.dataframe(info_df, use_container_width=True, hide_index=True, height=480)

    with pc2:
        # Neighborhood dot field + this building as accent marker
        near = pool[
            (pool["latitude"].between(row["latitude"] - 0.01, row["latitude"] + 0.01))
            & (pool["longitude"].between(row["longitude"] - 0.012, row["longitude"] + 0.012))
        ].copy()
        fig = px.scatter_mapbox(
            near, lat="latitude", lon="longitude", opacity=0.35,
            color_discrete_sequence=[T["muted"]], hover_name="display_name",
            zoom=14.5, center={"lat": row["latitude"], "lon": row["longitude"]},
            height=480, template=T["plotly"],
        )
        fig.add_trace(go.Scattermapbox(
            lat=[row["latitude"]], lon=[row["longitude"]],
            mode="markers",
            marker=dict(size=24, color=T["accent"]),
            hovertext=[row.get("display_name", "This property")],
            name="Selected", showlegend=False,
        ))
        fig.update_layout(mapbox_style=T["map_style"], margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    # ── SHAP waterfall ──
    st.markdown("---")
    st.markdown("### Why this price? Feature contributions")
    st.caption("Each bar shows how much a feature pushed the heritage prediction up or down from the typical Manhattan historic building. Red drags price down, the theme color pushes it up.")

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
            f"Model baseline (typical building): ~${base_dollars:,.0f}. "
            f"Sum of feature pushes takes it to ${pred_her:,.0f}."
        )

    except ImportError:
        st.error("SHAP is not installed. Run `pip install shap`.")
    except Exception as e:
        st.warning(f"SHAP unavailable for this sample: {e}")


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────
if   "1" in page: page1()
elif "2" in page: page2()
elif "3" in page: page3()
elif "4" in page: page4()
elif "5" in page: page5()
elif "6" in page: page6()
