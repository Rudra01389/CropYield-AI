"""Clean Maharashtra rice-Kharif yield data, aggregate Kharif weather, merge, write reports.

Run order:  python fetch_coordinates.py -> python fetch_weather.py -> python preprocess.py -> python eda.py

Nothing is imputed or edited. Rows are either kept, kept-and-flagged, or excluded with a logged reason.
No scaling / encoding / splitting happens here (that belongs to ANN model preprocessing, train data only).
"""
import re

import numpy as np
import pandas as pd

import config as C

QA = {}   # numbers collected for the reports
STEPS = []  # human-readable processing log


def step(msg):
    print(msg)
    STEPS.append(msg)


def md_table(df, index=False):
    df = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = ["| " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return head + "\n".join(rows) + "\n"


def parse_start_year(s):
    """'1999-00' -> 1999. Indian crop years run e.g. 1999-00; the Kharif crop is sown in Jun-Jul and
    harvested Oct-Dec of the FIRST calendar year, so Kharif weather = Jun-Sep of that start year."""
    m = re.fullmatch(r"(\d{4})-(\d{2})", str(s).strip())
    if not m or (int(m.group(1)) + 1) % 100 != int(m.group(2)):
        return np.nan
    return int(m.group(1))


# ------------------------------------------------------------------ crop data
def load_crop():
    raw = pd.read_csv(C.CROP_CSV)
    QA["n_raw"] = len(raw)
    step(f"1. Loaded `{C.CROP_CSV.name}`: {raw.shape[0]:,} rows x {raw.shape[1]} columns {list(raw.columns)}")

    for col in ["State", "Crop", "Season"]:
        raw[col + "_n"] = raw[col].astype(str).str.strip().str.lower()
    a = raw[raw.State_n == C.STATE]
    b = a[a.Crop_n == C.CROP]
    k = b[b.Season_n == C.SEASON].drop(columns=["State_n", "Crop_n", "Season_n"]).copy()
    QA.update(n_state=len(a), n_crop=len(b), n_filtered=len(k))
    step(f"2. Filters: State=Maharashtra {len(a):,} -> Crop=Rice {len(b):,} -> Season=Kharif {len(k):,} rows "
         f"(Rice seasons in Maharashtra: {b.Season.value_counts().to_dict()})")

    k["district"] = k["District"].astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    k["year"] = k["Year"].map(parse_start_year)
    bad_year = int(k["year"].isna().sum())
    step(f"3. Year convention: all `Year` strings match 'YYYY-YY' with consecutive years: {bad_year == 0} "
         f"(range {k['Year'].min()} .. {k['Year'].max()}; Kharif year = start year, an assumption stated in the report)")
    k = k.dropna(subset=["year"]).astype({"year": int})
    k = k.rename(columns={"Area": "area_ha", "Production": "production_t", "Yield": C.TARGET})

    QA["units"] = k[["Area Units", "Production Units"]].drop_duplicates().to_dict("records")
    QA["missing"] = k[["district", "year", "area_ha", "production_t", C.TARGET]].isna().sum()
    QA["dup_full"] = int(k.duplicated().sum())
    QA["dup_key"] = int(k.duplicated(["district", "year"], keep=False).sum())
    QA["invalid"] = int(((k.area_ha <= 0) | (k.production_t < 0) | (k[C.TARGET] < 0)).sum())
    step(f"4. Checks: missing values {QA['missing'].to_dict()}; exact duplicates {QA['dup_full']}; "
         f"rows sharing district+year {QA['dup_key']}; non-positive area / negative values {QA['invalid']}; units {QA['units']}")

    calc = k.production_t / k.area_ha
    diff = (calc - k[C.TARGET]).abs()
    QA["yield_mismatch"] = int((diff > 1e-6 * k[C.TARGET].abs().clip(lower=1)).sum())
    QA["yield_checked"] = int(calc.notna().sum())
    step(f"5. Yield vs Production/Area (both already t and ha, so t/ha): {QA['yield_mismatch']} mismatches in "
         f"{QA['yield_checked']} rows with production")

    # ---- exclusions (each logged) ----
    excl = []
    miss = k[k.production_t.isna()]
    for r in miss.itertuples():
        excl.append(dict(district=r.district, year=r.Year, area_ha=r.area_ha, production_t=r.production_t,
                         reported_yield=getattr(r, C.TARGET),
                         reason="Production missing; reported Yield 0.0 is a placeholder, true yield unknown"))
    high = k[k[C.TARGET] > C.MAX_PLAUSIBLE_YIELD]
    for r in high.itertuples():
        excl.append(dict(district=r.district, year=r.Year, area_ha=r.area_ha, production_t=r.production_t,
                         reported_yield=getattr(r, C.TARGET),
                         reason=f"Yield > {C.MAX_PLAUSIBLE_YIELD} t/ha; area inconsistent with the district's other years "
                                f"(see data_quality_report). Not corrected because the true area is unknown"))
    QA["exclusions"] = pd.DataFrame(excl)
    QA["outlier_evidence"] = (k[k.district.isin(high.district.unique())]
                              .sort_values(["district", "year"])[["district", "Year", "area_ha", "production_t", C.TARGET]]
                              if len(high) else pd.DataFrame())
    drop_idx = miss.index.union(high.index)
    k = k.drop(drop_idx).reset_index(drop=True)
    step(f"6. Excluded {len(drop_idx)} rows ({len(miss)} missing production, {len(high)} implausible yield); "
         f"{len(k)} crop records retained")

    k["area_lt_1000ha"] = k.area_ha < C.SMALL_AREA_HA
    QA["small_area"] = int(k.area_lt_1000ha.sum())
    QA["zero_prod"] = int((k.production_t == 0).sum())
    step(f"7. Kept but flagged: {QA['small_area']} rows with area < {C.SMALL_AREA_HA} ha "
         f"({QA['zero_prod']} with production = 0); production is rounded to the nearest 100 t there, so their yield is coarse")

    # robust within-district outlier screen (descriptive only; nothing removed)
    g = k.groupby("district")[C.TARGET]
    med, mad = g.transform("median"), g.transform(lambda s: (s - s.median()).abs().median())
    k["mad_z"] = ((k[C.TARGET] - med) / (1.4826 * mad.replace(0, np.nan))).round(2)
    QA["mad_flagged"] = k[k.mad_z.abs() > 3.5][["district", "Year", "area_ha", C.TARGET, "mad_z"]]
    step(f"8. Within-district robust outlier screen (|z|>3.5): {len(QA['mad_flagged'])} rows flagged, all retained")

    out = k[["district", "Year", "year", "area_ha", "production_t", C.TARGET, "area_lt_1000ha"]].rename(
        columns={"Year": "crop_year"})
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.CROP_CLEAN_CSV, index=False)
    QA["district_counts"] = out.groupby("district").size().rename("records")
    QA["year_counts"] = out.groupby("year").size().rename("records")
    return out


# ---------------------------------------------------------------- weather data
def aggregate_weather():
    if not C.WEATHER_DAILY_CSV.exists():
        raise SystemExit(f"Missing {C.WEATHER_DAILY_CSV}. Run fetch_coordinates.py then fetch_weather.py.")
    d = pd.read_csv(C.WEATHER_DAILY_CSV, parse_dates=["date"])
    QA["weather_rows"] = len(d)
    QA["weather_missing"] = d[["PRECTOTCORR", "T2M", "RH2M"]].isna().sum()
    s = d[d.date.dt.month.isin(C.KHARIF_MONTHS)].copy()
    expected = int(sum(pd.Period(f"2001-{m:02d}").days_in_month for m in C.KHARIF_MONTHS))  # 122
    g = s.groupby(["district", "year"])
    w = pd.DataFrame({
        "rain_kharif_mm": g["PRECTOTCORR"].sum(min_count=1),
        "temp_kharif_c": g["T2M"].mean(),
        "rh_kharif_pct": g["RH2M"].mean(),
        "valid_days_rain": g["PRECTOTCORR"].count(),
        "valid_days_temp": g["T2M"].count(),
        "valid_days_rh": g["RH2M"].count(),
    }).reset_index()
    for col, vd in [("rain_kharif_mm", "valid_days_rain"), ("temp_kharif_c", "valid_days_temp"),
                    ("rh_kharif_pct", "valid_days_rh")]:
        w.loc[w[vd] < C.MIN_VALID_DAY_FRACTION * expected, col] = np.nan
    w["rain_kharif_mm"] = w["rain_kharif_mm"].round(1)
    w[["temp_kharif_c", "rh_kharif_pct"]] = w[["temp_kharif_c", "rh_kharif_pct"]].round(2)
    w.to_csv(C.WEATHER_KHARIF_CSV, index=False)
    QA["weather_districts"] = w.district.nunique()
    # districts whose Kharif rainfall series is identical share one MERRA-2 grid cell
    series = w.sort_values("year").groupby("district")["rain_kharif_mm"].apply(tuple)
    groups = series.groupby(series).apply(lambda s: list(s.index))
    QA["shared_cells"] = [g for g in groups if len(g) > 1]
    QA["distinct_series"] = len(groups)
    QA["weather_incomplete"] = int(w[["rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"]].isna().any(axis=1).sum())
    step(f"9. Aggregated daily weather ({len(d):,} rows) to Jun-Sep per district-year: {len(w)} rows, "
         f"{w.district.nunique()} districts, years {w.year.min()}-{w.year.max()}; seasons with <90% valid days set to NaN: "
         f"{QA['weather_incomplete']}")
    return w


# ------------------------------------------------------------------------ merge
def merge(crop, weather):
    assert not weather.duplicated(["district", "year"]).any(), "duplicate weather district-year"
    assert not crop.duplicated(["district", "year"]).any(), "duplicate crop district-year"
    m = crop.merge(weather, on=["district", "year"], how="left", validate="one_to_one", indicator=True)
    no_weather = m[m._merge == "left_only"]
    nan_weather = m[(m._merge == "both") & m[["rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"]].isna().any(axis=1)]
    QA["unmatched"] = no_weather[["district", "crop_year"]]
    QA["nan_weather"] = nan_weather[["district", "crop_year"]]
    step(f"10. Merge crop x weather on district + start year: {len(crop)} crop rows; "
         f"no weather row: {len(no_weather)}; weather incomplete: {len(nan_weather)}")
    m = m.drop(columns="_merge")
    m = m.dropna(subset=["rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"])
    m = m.sort_values(["year", "district"]).reset_index(drop=True)
    assert not m.duplicated(["district", "year"]).any()
    m.to_csv(C.MERGED_CSV, index=False)
    ann = m[C.FEATURES + [C.TARGET, "area_lt_1000ha"]]
    ann.to_csv(C.ANN_READY_CSV, index=False)
    QA["n_merged"], QA["n_districts_final"] = len(m), m.district.nunique()
    QA["final_missing"] = ann.isna().sum().sum()
    step(f"11. Final merged / ANN-ready rows: {len(m)} ({m.district.nunique()} districts, years {m.year.min()}-{m.year.max()}); "
         f"missing values in ANN file: {QA['final_missing']}")
    step(f"    Final feature names: {C.FEATURES}; target: {C.TARGET}; extra flag column (not a feature): area_lt_1000ha")
    return m


# ---------------------------------------------------------------------- reports
def write_reports(m):
    C.REPORT_DIR.mkdir(exist_ok=True)
    (C.REPORT_DIR / "preprocessing_log.md").write_text(
        "# Preprocessing log\n\nGenerated by `preprocess.py`; every number below was computed from the data files.\n\n"
        + "\n".join(f"- {s}" for s in STEPS) + "\n")

    desc = m[[C.TARGET, "area_ha", "rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"]].describe().T.round(3)
    r = ["# Data quality report\n",
         "## Record counts\n",
         f"- Original file: {QA['n_raw']:,} rows (all India, all crops)",
         f"- Maharashtra: {QA['n_state']:,} -> Rice: {QA['n_crop']:,} -> Kharif: {QA['n_filtered']:,}",
         f"- Excluded: {len(QA['exclusions'])}; retained crop records: {len(m) + len(QA['unmatched']) + len(QA['nan_weather'])}",
         f"- Final merged (crop + weather): **{QA['n_merged']}** rows, **{QA['n_districts_final']}** districts\n",
         "## Missing values, duplicates, validity (Kharif rice, Maharashtra, before exclusions)\n",
         md_table(QA["missing"].rename("missing").to_frame(), index=True),
         f"- Exact duplicate rows: {QA['dup_full']}; rows sharing district+year: {QA['dup_key']}",
         f"- Non-positive area or negative values: {QA['invalid']}",
         f"- Units: {QA['units']}",
         f"- Yield vs Production/Area: {QA['yield_mismatch']} mismatches in {QA['yield_checked']} rows with production\n",
         "## Excluded records\n", md_table(QA["exclusions"]) if len(QA["exclusions"]) else "None.\n",
         "### Evidence for the implausible-yield exclusion\n",
         "Reported yield equals Production/Area, so the yield is internally consistent; the problem is the *area* "
         "(1,100 ha) which is ~1% of the same district's area in every other year. The true area is not recoverable "
         "from the file, so the row is excluded rather than corrected.\n",
         md_table(QA["outlier_evidence"]) if len(QA["outlier_evidence"]) else "",
         "## Flagged but retained\n",
         f"- {QA['small_area']} rows have area < {C.SMALL_AREA_HA} ha ({QA['zero_prod']} with production 0). Production is "
         "rounded to the nearest 100 tonnes in these rows, so yield is coarse (e.g. 0.25 t/ha). They are kept and flagged by "
         "`area_lt_1000ha`; decide in Phase 3 whether to down-weight or drop them.",
         f"- {len(QA['mad_flagged'])} rows exceed |robust z| > 3.5 within their district (listed below, all retained):\n",
         md_table(QA["mad_flagged"]),
         "## District-wise record counts (clean crop data)\n", md_table(QA["district_counts"].to_frame(), index=True),
         "## Year-wise record counts\n", md_table(QA["year_counts"].to_frame(), index=True),
         "## Weather coverage\n",
         f"- Daily rows: {QA['weather_rows']:,}; districts with weather: {QA['weather_districts']}",
         f"- Missing daily values: {QA['weather_missing'].to_dict()}",
         f"- Distinct weather series: {QA['distinct_series']} for {QA['weather_districts']} districts. Districts sharing one "
         f"NASA grid cell (identical weather): {QA['shared_cells']}",
         f"- Crop rows without a weather row: {len(QA['unmatched'])}; with incomplete weather: {len(QA['nan_weather'])}\n",
         "## Final dataset summary\n", md_table(desc, index=True),
         "## Assumptions and limitations\n",
         "- **Year mapping**: crop year `YYYY-YY` is matched to calendar year YYYY (Kharif is sown in June-July and harvested "
         "Oct-Dec of that year). The CSV does not state this; it is the standard Indian agricultural-year convention.",
         "- **Weather window**: June-September only. Rice harvested in Oct-Nov is partly influenced by October weather, which is "
         "not included.",
         "- **Weather source**: NASA POWER / MERRA-2 reanalysis (0.5 x 0.625 deg), not rain-gauge data. One representative point "
         "per district (centre of the OpenStreetMap district boundary) cannot capture within-district variation, e.g. the Konkan "
         "coast vs the Sahyadri ranges.",
         "- **Coordinate source**: OpenStreetMap Nominatim (ODbL); see `data/raw/district_coordinates.csv`.",
         "- **District changes**: Palghar (2014) was split from Thane, Gondia from Bhandara (1999), etc. Early-year records of "
         "parent districts cover the old boundaries while the coordinates are for the current boundaries.",
         "- **No soil, irrigation, fertiliser or variety data**; area is the only management feature.",
         "- Rice here is the aggregate of all Kharif rice; yield differences between districts include rainfed vs irrigated mix.",
         "- Plot-level statistics (EDA figures) use all rows for description only; no scaling or splitting was done. Phase 3 must "
         "split chronologically and fit scalers/encoders on training data only."]
    (C.REPORT_DIR / "data_quality_report.md").write_text("\n".join(map(str, r)) + "\n")
    print("Wrote reports/preprocessing_log.md and reports/data_quality_report.md")


def main():
    crop = load_crop()
    w = aggregate_weather()
    m = merge(crop, w)
    write_reports(m)


if __name__ == "__main__":
    main()
