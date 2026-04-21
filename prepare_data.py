"""
Manhattan Heritage Property Analysis — Data Preparation
Merges Landmark, Sales, and PLUTO datasets on BBL,
cleans data, and engineers architectural/preservation features.
"""

import pandas as pd
import numpy as np

# ── 1. Read raw datasets ─────────────────────────────────────────────
print("Reading datasets...")
landmark = pd.read_csv(
    "Individual_Landmark_and_Historic_District_Building_Database_20260420.csv",
    low_memory=False,
)
sales = pd.read_csv(
    "NYC_Citywide_Rolling_Calendar_Sales_20260420.csv",
    low_memory=False,
)
pluto = pd.read_csv(
    "Primary_Land_Use_Tax_Lot_Output_(PLUTO)_20260420.csv",
    low_memory=False,
)
print(f"  Landmark : {landmark.shape}")
print(f"  Sales    : {sales.shape}")
print(f"  PLUTO    : {pluto.shape}")

# ── 2. Filter Manhattan ──────────────────────────────────────────────
landmark_mn = landmark[landmark["Borough"] == "MN"].copy()
sales_mn = sales[sales["BOROUGH"].astype(str).str.strip() == "1"].copy()
pluto_mn = pluto[pluto["borough"] == "MN"].copy()
print(f"\nManhattan: Landmark={len(landmark_mn)}, Sales={len(sales_mn)}, PLUTO={len(pluto_mn)}")

# ── 3. Construct BBL for Sales ────────────────────────────────────────
for col in ["BOROUGH", "BLOCK", "LOT"]:
    sales_mn[col] = sales_mn[col].astype(str).str.replace(",", "").str.strip()

sales_mn["BBL"] = (
    sales_mn["BOROUGH"].apply(lambda x: str(int(float(x))))
    + sales_mn["BLOCK"].apply(lambda x: f"{int(float(x)):05d}")
    + sales_mn["LOT"].apply(lambda x: f"{int(float(x)):04d}")
)
landmark_mn["BBL"] = landmark_mn["BBL"].astype(str)
pluto_mn["BBL"] = pluto_mn["BBL"].astype(str)

# ── 4. Merge on BBL ──────────────────────────────────────────────────
print("Merging...")
merged = landmark_mn.merge(sales_mn, on="BBL", how="inner")
merged = merged.merge(pluto_mn, on="BBL", how="inner")
print(f"  Merged shape: {merged.shape}")


# ── 5. Helper: clean numeric columns ─────────────────────────────────
def clean_numeric(series, zero_as_nan=False):
    """Remove $, commas and convert to float."""
    s = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan})
    )
    if zero_as_nan:
        s = s.replace({"0": np.nan})
    return s.astype(float)


# ── 6. Build analysis DataFrame ──────────────────────────────────────
df = pd.DataFrame()

# --- A. Core Pricing / Structural Controls ---
df["BBL"] = merged["BBL"]
df["sale_price"] = clean_numeric(merged["SALE PRICE"], zero_as_nan=True)
df["sale_date"] = pd.to_datetime(merged["SALE DATE"], errors="coerce")
# Use Sales gross sqft, fall back to PLUTO building area
sales_sqft = clean_numeric(merged["GROSS SQUARE FEET"])
pluto_sqft = clean_numeric(merged["bldgarea"])
df["gross_sqft"] = sales_sqft.fillna(pluto_sqft).replace(0, np.nan)
land_sqft_s = clean_numeric(merged["LAND SQUARE FEET"])
land_sqft_p = clean_numeric(merged["lotarea"])
df["land_sqft"] = land_sqft_s.fillna(land_sqft_p).replace(0, np.nan)
df["residential_units"] = clean_numeric(merged["RESIDENTIAL UNITS"])
df["commercial_units"] = clean_numeric(merged["COMMERCIAL UNITS"])
df["total_units"] = clean_numeric(merged["TOTAL UNITS"])
df["num_floors"] = pd.to_numeric(merged["numfloors"], errors="coerce")
df["building_depth"] = clean_numeric(merged["bldgdepth"])
df["lot_depth"] = clean_numeric(merged["lotdepth"])
df["lot_frontage"] = clean_numeric(merged["lotfront"])
df["building_frontage"] = clean_numeric(merged["bldgfront"])
df["building_area"] = clean_numeric(merged["bldgarea"])
df["lot_area"] = clean_numeric(merged["lotarea"])
df["assess_land"] = pd.to_numeric(merged["assessland"], errors="coerce")
df["assess_total"] = pd.to_numeric(merged["assesstot"], errors="coerce")
df["exempt_total"] = pd.to_numeric(merged["exempttot"], errors="coerce")
df["built_far"] = pd.to_numeric(merged["builtfar"], errors="coerce")
df["resid_far"] = pd.to_numeric(merged["residfar"], errors="coerce")
df["comm_far"] = pd.to_numeric(merged["commfar"], errors="coerce")
df["facil_far"] = pd.to_numeric(merged["facilfar"], errors="coerce")

# --- B. Location Controls ---
df["neighborhood"] = merged["NEIGHBORHOOD"].str.strip()
df["zip_code"] = merged["ZIP CODE"].astype(str).str.strip()
df["latitude"] = pd.to_numeric(merged["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(merged["longitude"], errors="coerce")
df["zoning"] = merged["zonedist1"].str.strip()
df["building_class"] = merged["BUILDING CLASS CATEGORY"].str.strip()
df["building_class_code"] = merged["BUILDING CLASS AT PRESENT"].str.strip()

# --- C. Historical Preservation Variables ---
df["historic_district"] = merged["Hist_Dist"].fillna("").str.strip()
df["in_historic_district"] = (df["historic_district"] != "").astype(int)
df["landmark_orig"] = merged["LM_Orig"].astype(str).str.strip()
df["landmark_new"] = merged["LM_New"].astype(str).str.strip()
df["is_landmark"] = (
    (df["landmark_orig"].isin(["1", "1.0"])) | (df["landmark_new"].isin(["1", "1.0"]))
).astype(int)
df["pluto_landmark"] = merged["landmark"].fillna("").str.strip()
df["pluto_histdist"] = merged["histdist"].fillna("").str.strip()

# --- D. Architectural Aesthetic Variables ---
df["architect"] = merged["Arch_Build"].fillna("Not determined").str.strip()
df["style_primary"] = merged["Style_Prim"].fillna("Not determined").str.strip()
df["style_secondary"] = merged["Style_Sec"].fillna("").str.strip()
df["material_primary"] = merged["Mat_Prim"].fillna("Not determined").str.strip()
df["material_secondary"] = merged["Mat_Sec"].fillna("").str.strip()
df["building_type"] = merged["Build_Type"].fillna("").str.strip()
df["original_use"] = merged["Use_Orig"].fillna("").str.strip()
df["building_name"] = merged["Build_Nme"].fillna("").str.strip()
df["address"] = merged["Des_Addres"].fillna("").str.strip()

# Construction year — try landmark Date_Low first, fall back to PLUTO yearbuilt
lm_year = pd.to_numeric(merged["Date_Low"], errors="coerce")
pluto_year = pd.to_numeric(merged["yearbuilt"], errors="coerce")
df["construction_year"] = lm_year.fillna(pluto_year)
df["circa_flag"] = merged["Circa"].astype(str).str.strip().isin(["1", "1.0"]).astype(int)

# Alteration variables
df["altered"] = merged["Altered"].astype(str).str.strip().isin(["1", "1.0"]).astype(int)
alt1 = pd.to_numeric(merged["Alt_Date_1"], errors="coerce")
alt2 = pd.to_numeric(merged["yearalter1"].astype(str).str.replace(",", ""), errors="coerce")
df["alteration_year"] = alt1.fillna(alt2)
df["is_altered"] = (df["altered"] == 1).astype(int) | (df["alteration_year"] > 0).astype(int)

# Owner / Developer
df["owner_developer"] = merged["Own_Devel"].fillna("").str.strip()

# --- E. Engineered Features ---

# Building Age
current_year = 2026
df["building_age"] = current_year - df["construction_year"]
df.loc[df["building_age"] < 0, "building_age"] = np.nan
df.loc[df["building_age"] > 400, "building_age"] = np.nan

# Construction Era Bins
def era_bin(year):
    if pd.isna(year) or year <= 0:
        return "Unknown"
    elif year < 1850:
        return "Pre-1850"
    elif year < 1900:
        return "1850–1899"
    elif year < 1920:
        return "1900–1919"
    elif year < 1940:
        return "1920–1939 (Art Deco)"
    elif year < 1970:
        return "1940–1969 (Mid-Century)"
    else:
        return "1970+"

df["construction_era"] = df["construction_year"].apply(era_bin)

# Price per square foot
df["price_per_sqft"] = df["sale_price"] / df["gross_sqft"]
df.loc[df["price_per_sqft"] == np.inf, "price_per_sqft"] = np.nan

# Time since alteration
df["years_since_alteration"] = current_year - df["alteration_year"]
df.loc[df["years_since_alteration"] < 0, "years_since_alteration"] = np.nan

# Architect Prestige Score (frequency-based)
arch_counts = df["architect"].value_counts()
df["architect_building_count"] = df["architect"].map(arch_counts)
# Normalize: higher score = more prolific architect
max_count = df["architect_building_count"].max()
df["architect_prestige_score"] = df["architect_building_count"] / max_count if max_count > 0 else 0

# Rare Style Premium (inverse frequency)
style_counts = df["style_primary"].value_counts()
total_styles = len(df)
df["style_frequency"] = df["style_primary"].map(style_counts) / total_styles
df["rare_style_score"] = 1 - df["style_frequency"]  # rarer = higher score

# Preservation status composite
df["preservation_level"] = "None"
df.loc[df["in_historic_district"] == 1, "preservation_level"] = "Historic District"
df.loc[df["is_landmark"] == 1, "preservation_level"] = "Individual Landmark"
df.loc[(df["in_historic_district"] == 1) & (df["is_landmark"] == 1), "preservation_level"] = "Both"

# Sale year
df["sale_year"] = df["sale_date"].dt.year
df["sale_month"] = df["sale_date"].dt.month

# ── 7. Filter ─────────────────────────────────────────────────────────
print(f"\nBefore filtering: {len(df)} rows")

# Remove $0 sales (non-market transactions)
df = df[df["sale_price"] > 0].copy()
print(f"After removing $0 sales: {len(df)} rows")

# For price_per_sqft, need gross_sqft > 0; but keep rows even if sqft missing
# Just recalc price_per_sqft with NaN where sqft is missing
df["price_per_sqft"] = np.where(
    df["gross_sqft"] > 0, df["sale_price"] / df["gross_sqft"], np.nan
)
print(f"Rows with valid price_per_sqft: {df['price_per_sqft'].notna().sum()}")

# ── 8. Save ───────────────────────────────────────────────────────────
output = "Manhattan_Heritage_Analysis.csv"
df.to_csv(output, index=False)
print(f"\n✅ Saved: {output}")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n=== Column Summary ===")
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    non_null = df[col].notna().sum()
    print(f"  {i+1:3d}. {col:35s} | {str(dtype):10s} | {non_null} non-null")

# Quick stats on key variables
print(f"\n=== Key Statistics ===")
print(f"  Sale Price: mean=${df['sale_price'].mean():,.0f}, median=${df['sale_price'].median():,.0f}")
print(f"  Price/SqFt: mean=${df['price_per_sqft'].mean():,.0f}, median=${df['price_per_sqft'].median():,.0f}")
print(f"  Building Age: mean={df['building_age'].mean():.0f} years")
print(f"  In Historic District: {df['in_historic_district'].sum()} ({df['in_historic_district'].mean()*100:.1f}%)")
print(f"  Is Landmark: {df['is_landmark'].sum()} ({df['is_landmark'].mean()*100:.1f}%)")
print(f"  Unique Architects: {df['architect'].nunique()}")
print(f"  Unique Styles: {df['style_primary'].nunique()}")
print(f"  Construction Eras: {df['construction_era'].value_counts().to_dict()}")
