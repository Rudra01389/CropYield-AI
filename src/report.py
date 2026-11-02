"""Build the downloadable one-row prediction report (CSV) from the values the user entered and the ANN output."""
import re

import pandas as pd

COLUMNS = [
    "District",
    "Agricultural year",
    "Cultivated area (ha)",
    "Kharif rainfall Jun-Sep (mm)",
    "Average temperature (°C)",
    "Average relative humidity (%)",
    "Predicted rice yield (tonnes/hectare)",
    "Estimated production (tonnes)",
    "District historical average yield (tonnes/hectare, 1997-2015)",
]


def build_report(values: dict, predicted_yield: float, historical_average: float) -> pd.DataFrame:
    """values: the validated inputs (district, year, area_ha, rain_kharif_mm, temp_kharif_c, rh_kharif_pct)."""
    return pd.DataFrame([[
        str(values["district"]).title(),
        int(values["year"]),
        float(values["area_ha"]),
        float(values["rain_kharif_mm"]),
        float(values["temp_kharif_c"]),
        float(values["rh_kharif_pct"]),
        round(float(predicted_yield), 4),
        round(float(predicted_yield) * float(values["area_ha"]), 1),   # estimate = yield x entered area
        round(float(historical_average), 4),
    ]], columns=COLUMNS)


def to_csv_bytes(report: pd.DataFrame) -> bytes:
    """UTF-8 with BOM so Excel shows the degree sign correctly."""
    return report.to_csv(index=False).encode("utf-8-sig")


def file_name(values: dict) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(values["district"]).lower()).strip("_")
    return f"cropyield_prediction_{slug}_{int(values['year'])}.csv"
