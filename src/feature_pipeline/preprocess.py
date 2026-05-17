"""
⚡ Preprocessing Script for Housing Regression MLE

- Reads train/eval/holdout CSVs from data/raw/.
- Cleans and normalizes city names.
- Maps cities to metros and merges lat/lng.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.

"""

"""
Preprocessing: city normalization + (optional) lat/lng merge, duplicate drop, outlier removal.

- Production defaults read from data/raw/ and write to data/processed/
- Tests can override `raw_dir`, `processed_dir`, and pass `metros_path=None`
  to skip merge safely without touching disk assets.
"""

import pandas as pd
import re
from typing import Dict, Optional
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def normalize_city(s: str) -> str:
    """
    Aggressive normalization for matching:
    - Remove state abbreviations and county suffixes
    - Standardize separators
    - Remove common words
    """
    if pd.isna(s):
        return s
    
    s = str(s).lower().strip()
    
    # Remove state patterns: ", ca", "-ca", ", ny-nj", ", dc-va-md-wv", etc.
    s = re.sub(r',\s*[a-z]{2}(?:-[a-z]{2})*', '', s)  # , CA or , NY-NJ
    s = re.sub(r'-\s*[a-z]{2}(?:-[a-z]{2})*', '', s)  # -CA or -NY-NJ
    
    # Remove county/parish suffixes
    s = re.sub(r',\s*[a-z]+\s+(?:county|parish)', '', s)
    
    # Standardize separators
    s = re.sub(r'[–—-]', '-', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\s_]+', '-', s)  # Convert spaces to dashes for consistency
    
    # Remove trailing punctuation
    s = s.rstrip(',.-')
    
    return s

# Expand your mapping to cover all 30 cities
CITY_MAPPING = {
    # Your existing mappings (already good)
    "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
    "denver-aurora-lakewood": "denver-aurora-centennial",
    "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
    "austin-round rock-georgetown": "austin-round rock-san marcos",
    "miami-fort lauderdale-pompano beach": "miami-fort lauderdale-west palm beach",
    "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
    "dc_metro": "washington-arlington-alexandria",
    "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
    
    # Additional mappings needed (examples)
    "pittsburgh": "pittsburgh",  # Already matches after normalization
    "boston-cambridge-newton": "boston-cambridge-newton",
    "tampa-st-petersburg-clearwater": "tampa-st-petersburg-clearwater",
    "baltimore-columbia-towson": "baltimore-columbia-towson",
    "portland-vancouver-hillsboro": "portland-vancouver-hillsboro",
    "philadelphia-camden-wilmington": "philadelphia-camden-wilmington",
    "new york-newark-jersey city": "new york-newark-jersey city",
    "chicago-naperville-elgin": "chicago-naperville-elgin",
    "orlando-kissimmee-sanford": "orlando-kissimmee-sanford",
    "seattle-tacoma-bellevue": "seattle-tacoma-bellevue",
    "san diego-chula vista-carlsbad": "san diego-chula vista-carlsbad",
    "st louis": "st louis",
    "sacramento-roseville-folsom": "sacramento-roseville-folsom",
    "phoenix-mesa-chandler": "phoenix-mesa-chandler",
    "riverside-san bernardino-ontario": "riverside-san bernardino-ontario",
    "san antonio-new braunfels": "san antonio-new braunfels",
    "detroit-warren-dearborn": "detroit-warren-dearborn",
    "cincinnati": "cincinnati",
    "charlotte-concord-gastonia": "charlotte-concord-gastonia",
    "los angeles-long beach-anaheim": "los angeles-long beach-anaheim",
    "dallas-fort worth-arlington": "dallas-fort worth-arlington",
    "minneapolis-st paul-bloomington": "minneapolis-st paul-bloomington",
}

def clean_and_merge(df: pd.DataFrame, metros_path: str | None = "data/raw/usmetros.csv") -> pd.DataFrame:
    """Fixed version with proper matching."""
    
    if "city_full" not in df.columns:
        print("⚠️ Skipping city merge: no 'city_full' column present.")
        return df
    
    # Step 1: Normalize df cities aggressively
    df["city_normalized"] = df["city_full"].apply(normalize_city)
    
    # Step 2: Apply mapping (normalize mapping keys too)
    norm_mapping = {normalize_city(k): normalize_city(v) 
                    for k, v in CITY_MAPPING.items()}
    df["city_normalized"] = df["city_normalized"].replace(norm_mapping)
    
    # Step 3: Load and normalize usmetros
    if not metros_path or not Path(metros_path).exists():
        print("⚠️ Skipping lat/lng merge: metros file not provided or not found.")
        return df
    
    metros = pd.read_csv(metros_path)
    if "metro_full" not in metros.columns or not {"lat", "lng"}.issubset(metros.columns):
        print("⚠️ Skipping lat/lng merge: metros file missing required columns.")
        return df
    
    metros["metro_normalized"] = metros["metro_full"].apply(normalize_city)
    
    # Step 4: Merge on normalized keys
    df = df.merge(
        metros[["metro_normalized", "lat", "lng"]],
        how="left", 
        left_on="city_normalized", 
        right_on="metro_normalized"
    )
    
    # Step 5: Check results
    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print(f"⚠️ Still missing lat/lng for {len(missing)} cities: {list(missing)[:5]}...")
        # Debug: Show normalized versions of missing
        for city in missing[:3]:
            norm_city = df[df["city_full"] == city]["city_normalized"].iloc[0]
            print(f"  - '{city}' -> normalized: '{norm_city}'")
    else:
        print("✅ All cities matched with metros dataset.")
    
    # Clean up temporary columns
    df.drop(columns=["city_normalized", "metro_normalized"], inplace=True, errors="ignore")
    
    return df



def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates while keeping different dates/years."""
    before = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
    after = df.shape[0]
    print(f"✅ Dropped {before - after} duplicate rows (excluding date/year).")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in median_list_price (> 19M)."""
    if "median_list_price" not in df.columns:
        return df
    before = df.shape[0]
    df = df[df["median_list_price"] <= 19_000_000].copy()
    after = df.shape[0]
    print(f"✅ Removed {before - after} rows with median_list_price > 19M.")
    return df


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/usmetros.csv",
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)

    df = clean_and_merge(df, metros_path=metros_path)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    out_path = processed_dir / f"cleaning_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Preprocessed {split} saved to {out_path} ({df.shape})")
    return df


def run_preprocess(
    splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/usmetros.csv",
):
    for s in splits:
        preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir, metros_path=metros_path)


if __name__ == "__main__":
    run_preprocess()