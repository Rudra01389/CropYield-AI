"""Shared settings for CropYield AI. Paths are relative to this file, so the project is portable."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"
EDA_DIR = REPORT_DIR / "eda"

# --- inputs supplied by the team (never modified) ---
CROP_CSV = DATA_DIR / "India Agriculture Crop Production.csv"
PUNE_REFERENCE_CSV = DATA_DIR / "POWER_Point_Daily_20000101_20201231_018d52N_073d86E_LST.csv"

# --- files produced by the scripts ---
COORD_CSV = RAW_DIR / "district_coordinates.csv"          # fetch_coordinates.py
WEATHER_CACHE_DIR = RAW_DIR / "weather_cache"             # one file per district (download cache)
WEATHER_DAILY_CSV = RAW_DIR / "weather_daily_all_districts.csv"
WEATHER_KHARIF_CSV = PROC_DIR / "weather_kharif_district_year.csv"
CROP_CLEAN_CSV = PROC_DIR / "maharashtra_rice_kharif_crop_clean.csv"
MERGED_CSV = PROC_DIR / "maharashtra_rice_kharif_merged.csv"
ANN_READY_CSV = PROC_DIR / "ann_ready_dataset.csv"

# --- crop filters ---
STATE, CROP, SEASON = "maharashtra", "rice", "kharif"

# --- weather ---
WEATHER_START, WEATHER_END = "19970101", "20201231"   # daily endpoint wants YYYYMMDD
KHARIF_MONTHS = [6, 7, 8, 9]                          # June-September
MIN_VALID_DAY_FRACTION = 0.90                         # else the season aggregate is set to NaN

# --- data-quality rules ---
MAX_PLAUSIBLE_YIELD = 10.0     # t/ha; rows above are excluded and reported, never edited
SMALL_AREA_HA = 1000           # rows below are KEPT but flagged (production is rounded to 100 t)

TARGET = "yield_t_ha"
FEATURES = ["district", "year", "area_ha", "rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"]
