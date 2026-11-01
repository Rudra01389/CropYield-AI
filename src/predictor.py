"""Load the saved Phase 3 ANN + preprocessing pipeline and make predictions.

Nothing is retrained here. Feature names and their ORDER come from models/model_metadata.json
(`input_features`), the same list used when the preprocessor was fitted in train_models.py.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

NUMERIC = ["year", "area_ha", "rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"]
LABELS = {
    "year": "Agricultural year (start year)", "area_ha": "Cultivated area (ha)",
    "rain_kharif_mm": "Kharif rainfall, Jun-Sep (mm)", "temp_kharif_c": "Average temperature (°C)",
    "rh_kharif_pct": "Average relative humidity (%)",
}


class ModelNotAvailable(RuntimeError):
    """Raised when saved model files are missing or cannot be loaded."""


class Predictor:
    def __init__(self):
        try:
            import tensorflow as tf
            self.meta = json.loads((MODEL_DIR / "model_metadata.json").read_text())
            self.pre = joblib.load(MODEL_DIR / "preprocessor.joblib")
            self.model = tf.keras.models.load_model(MODEL_DIR / "ann_model.keras", compile=False)
        except Exception as exc:  # missing file, version mismatch, ...
            raise ModelNotAvailable("The trained model could not be loaded. Run train_models.py first.") from exc
        self.features = self.meta["input_features"]          # order used in training
        self.baseline = self.meta["district_mean_baseline"]  # training-set district means
        expected = set(NUMERIC) | {"district"}
        if set(self.features) != expected:
            raise ModelNotAvailable("Model metadata does not match the app's input fields.")

    def predict(self, values: dict) -> float:
        row = pd.DataFrame([{f: values[f] for f in self.features}])[self.features]
        x = self.pre.transform(row).astype("float32")
        return float(self.model.predict(x, verbose=0).ravel()[0])

    def baseline_for(self, district: str) -> float:
        return float(self.baseline["means"].get(district, self.baseline["fallback_for_unseen_district"]))


def validate_inputs(values: dict, ref: pd.DataFrame, districts: list):
    """Return (errors, warnings). Errors block prediction; warnings flag extrapolation."""
    errors, warnings = [], []
    if values.get("district") not in districts:
        errors.append("Please choose a district from the list.")
    for col in NUMERIC:
        v = values.get(col)
        if v is None or not np.isfinite(v):
            errors.append(f"{LABELS[col]} is missing or not a number.")
    if errors:
        return errors, warnings
    if values["area_ha"] <= 0:
        errors.append("Cultivated area must be greater than 0 ha.")
    if not 0 <= values["rh_kharif_pct"] <= 100:
        errors.append("Relative humidity must be between 0 and 100 %.")
    if values["rain_kharif_mm"] < 0:
        errors.append("Rainfall cannot be negative.")
    if errors:
        return errors, warnings
    for col in NUMERIC:
        lo, hi = ref[col].min(), ref[col].max()
        if not lo <= values[col] <= hi:
            warnings.append(f"{LABELS[col]} = {values[col]:g} is outside the range seen in training/historical data "
                            f"({lo:g} to {hi:g}); the prediction is an extrapolation and less reliable.")
    return errors, warnings
