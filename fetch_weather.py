"""Download DAILY weather per district from the NASA POWER API.

Needs data/raw/district_coordinates.csv  (run fetch_coordinates.py first).
Output: data/raw/weather_daily_all_districts.csv
        (district, date, year, doy, PRECTOTCORR mm/day, T2M C, RH2M %)
Each district is cached in data/raw/weather_cache/, so re-running only downloads what is missing.

API docs: https://power.larc.nasa.gov/docs/   (daily endpoint, dates as YYYYMMDD)
Data is MERRA-2 reanalysis (0.5 x 0.625 deg grid), a model product rather than gauge data.
Fill value -999 is converted to NaN; nothing is interpolated.
"""
import time

import pandas as pd
import requests

import config as C

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "PRECTOTCORR,T2M,RH2M"


def download(lat, lon, retries=5):
    q = {"parameters": PARAMS, "community": "AG", "latitude": round(lat, 4), "longitude": round(lon, 4),
         "start": C.WEATHER_START, "end": C.WEATHER_END, "format": "JSON", "time-standard": "LST"}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(URL, params=q, timeout=120)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            p = r.json()["properties"]["parameter"]
            df = pd.DataFrame(p).rename_axis("date").reset_index()
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df = df.replace(-999, float("nan"))
            df.insert(1, "year", df["date"].dt.year)
            df.insert(2, "doy", df["date"].dt.dayofyear)
            return df
        except Exception as exc:
            wait = 5 * attempt
            print(f"   attempt {attempt}/{retries} failed ({exc}); waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"NASA POWER download failed for ({lat}, {lon})")


def check_against_pune_reference():
    """Sanity check: our API parsing must reproduce the team's Pune file for the same point."""
    ref = C.PUNE_REFERENCE_CSV
    if not ref.exists():
        return
    lines = ref.read_text().splitlines()
    skip = next(i for i, l in enumerate(lines) if "-END HEADER-" in l) + 1
    r = pd.read_csv(ref, skiprows=skip).replace(-999, float("nan"))
    r = r.rename(columns={"YEAR": "year", "DOY": "doy"})
    cache = C.WEATHER_CACHE_DIR / "_pune_reference_point.csv"
    if cache.exists():
        a = pd.read_csv(cache, parse_dates=["date"])
    else:
        a = download(18.5204, 73.8567)
        a.to_csv(cache, index=False)
    m = r.merge(a, on=["year", "doy"], suffixes=("_ref", "_api"))
    print(f"\nPune reference check on {len(m)} overlapping days (API vs your file):")
    for c in ["PRECTOTCORR", "T2M", "RH2M"]:
        print(f"  {c}: max abs difference = {(m[c + '_ref'] - m[c + '_api']).abs().max():.4f}")


def main():
    if not C.COORD_CSV.exists():
        raise SystemExit(f"Missing {C.COORD_CSV} - run: python fetch_coordinates.py")
    coords = pd.read_csv(C.COORD_CSV)
    C.WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts, failed = [], []
    for row in coords.itertuples():
        f = C.WEATHER_CACHE_DIR / f"{row.district}.csv"
        if f.exists():
            df = pd.read_csv(f, parse_dates=["date"])
            print(f"cached   {row.district}")
        else:
            print(f"download {row.district} ({row.lat:.3f}, {row.lon:.3f})")
            try:
                df = download(row.lat, row.lon)
            except RuntimeError as exc:
                print("  FAILED:", exc)
                failed.append(row.district)
                continue
            df.to_csv(f, index=False)
            time.sleep(1)
        df.insert(0, "district", row.district)
        parts.append(df)
    if not parts:
        raise SystemExit("No weather downloaded - API unreachable? Not generating substitute data.")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(C.WEATHER_DAILY_CSV, index=False)
    print(f"\nSaved {C.WEATHER_DAILY_CSV}  shape={out.shape}")
    print("Districts with weather:", out.district.nunique(), "| failed:", failed)
    print("Missing values:\n", out[["PRECTOTCORR", "T2M", "RH2M"]].isna().sum().to_string())
    check_against_pune_reference()


if __name__ == "__main__":
    main()
