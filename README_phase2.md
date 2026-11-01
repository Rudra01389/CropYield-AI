# CropYield AI - Data preparation (Phase 2)

Inputs (never modified, in `data/`): `India Agriculture Crop Production.csv`, Pune NASA POWER reference file
(used only to validate the API download; its weather is NOT applied to other districts).

```
pip install -r requirements.txt
python fetch_coordinates.py   # OpenStreetMap Nominatim -> data/raw/district_coordinates.csv
python fetch_weather.py       # NASA POWER daily API    -> data/raw/weather_daily_all_districts.csv (cached per district)
python preprocess.py          # -> data/processed/*.csv, reports/data_quality_report.md, reports/preprocessing_log.md
python eda.py                 # -> reports/eda/*.png, reports/eda/eda_summary.md
```

Outputs: `data/processed/ann_ready_dataset.csv` (features: district, year, area_ha, rain_kharif_mm, temp_kharif_c,
rh_kharif_pct; target: yield_t_ha; flag column `area_lt_1000ha` is not a feature).

Obsolete from the first attempt (not used any more, safe to delete): `data/raw/weather_monthly.csv`.
