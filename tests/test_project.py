"""Final checks for CropYield AI. Run from the project folder:  python -m pytest -q

Nothing here trains or modifies models/data; it only loads and verifies the existing artifacts.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402
from src.predictor import ModelNotAvailable, Predictor, validate_inputs  # noqa: E402

DISTRICT_SAMPLE = ["PUNE", "GONDIA", "SOLAPUR", "KOLHAPUR", "NAGPUR", "THANE"]


@pytest.fixture(scope="module")
def df():
    return data.load_dataset()


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


# ------------------------------------------------------------------ data
def test_dataset_shape_and_columns(df):
    assert df.shape[0] == 642 and df.district.nunique() == 34
    assert (df.year.min(), df.year.max()) == (1997, 2019)
    assert df.drop(columns=["area_lt_1000ha"]).isna().sum().sum() == 0
    assert "production_t" not in df.columns and "Production" not in df.columns   # no target leakage


def test_chronological_split_counts(df):
    tr, va, te = df[df.year <= 2015], df[df.year.between(2016, 2017)], df[df.year >= 2018]
    assert (len(tr), len(va), len(te)) == (540, 51, 51)
    assert tr.year.max() < va.year.min() <= va.year.max() < te.year.min()
    assert set(va.district) | set(te.district) <= set(tr.district)


def test_metadata_matches_data(df):
    m = data.load_metadata()
    assert m["split"]["n_train"] == 540 and m["split"]["n_val"] == 51 and m["split"]["n_test"] == 51
    assert m["production_used"] is False
    assert set(m["input_features"]) == {"year", "area_ha", "rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct", "district"}


# ------------------------------------------------------------------ model
def test_model_and_preprocessor_load(predictor):
    assert predictor.features == predictor.meta["input_features"]
    names = predictor.pre.get_feature_names_out()
    assert len(names) == predictor.model.input_shape[1] == 5 + 34


def test_predictor_reproduces_phase3_predictions(df, predictor):
    preds = data.load_predictions()
    m = preds.merge(df, on=["district", "year"], suffixes=("", "_d"))
    assert len(m) == 642
    x = predictor.pre.transform(m[predictor.features]).astype("float32")
    out = predictor.model.predict(x, verbose=0).ravel()
    assert np.abs(out - m.pred_ann_all_features).max() < 1e-3
    # the single-row path used by the app gives the same answer as the batch path
    for i in range(0, len(m), 90):
        row = m.iloc[i][predictor.features].to_dict()
        assert abs(predictor.predict(row) - out[i]) < 1e-4


@pytest.mark.parametrize("district", DISTRICT_SAMPLE)
def test_prediction_is_finite_and_plausible(df, predictor, district):
    row = df[df.district == district].sort_values("year").iloc[-1][predictor.features].to_dict()
    y = predictor.predict(row)
    assert np.isfinite(y) and -0.5 < y < 6.0


def test_unknown_district_does_not_crash(df, predictor):
    row = df.iloc[0][predictor.features].to_dict()
    row["district"] = "ATLANTIS"
    assert np.isfinite(predictor.predict(row))   # one-hot ignores unseen category


# ------------------------------------------------------- reports use real data
def test_comparison_csv_matches_predictions_csv():
    comp, preds = data.load_comparison(), data.load_predictions()
    cols = {"District-mean baseline": "pred_district_mean_baseline", "Linear Regression": "pred_linear_regression",
            "ANN (all features)": "pred_ann_all_features", "ANN (no weather)": "pred_ann_no_weather"}
    for model, col in cols.items():
        for split in ("val", "test"):
            p = preds[preds.split == split]
            rmse = float(np.sqrt(((p.actual_yield_t_ha - p[col]) ** 2).mean()))
            reported = comp[(comp.model == model) & (comp.split == split)].RMSE.iloc[0]
            assert abs(rmse - reported) < 2e-3, (model, split, rmse, reported)


# ------------------------------------------------------------ validation
def test_input_validation(df):
    ds = sorted(df.district.unique())
    ok = dict(district="PUNE", year=2019, area_ha=1000.0, rain_kharif_mm=800.0, temp_kharif_c=26.0, rh_kharif_pct=80.0)
    assert validate_inputs(ok, df, ds) == ([], [])
    assert validate_inputs({**ok, "area_ha": 0}, df, ds)[0]
    assert validate_inputs({**ok, "rh_kharif_pct": 120}, df, ds)[0]
    assert validate_inputs({**ok, "rain_kharif_mm": -1}, df, ds)[0]
    assert validate_inputs({**ok, "district": "XYZ"}, df, ds)[0]
    assert validate_inputs({**ok, "temp_kharif_c": float("nan")}, df, ds)[0]
    assert validate_inputs({**ok, "rain_kharif_mm": 9000}, df, ds)[1]       # extrapolation warning


def test_missing_model_gives_friendly_error(monkeypatch, tmp_path):
    import src.predictor as p
    monkeypatch.setattr(p, "MODEL_DIR", tmp_path)
    with pytest.raises(ModelNotAvailable):
        p.Predictor()


# ---------------------------------------------------------------- Streamlit
@pytest.fixture()
def app():  # fresh app per test, so sidebar/page state never leaks between tests
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


@pytest.mark.parametrize("page", ["Predict yield", "Data & trends", "Model performance", "About & limitations"])
def test_every_page_renders(app, page):
    at = app.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert not at.error


@pytest.mark.parametrize("district", DISTRICT_SAMPLE)
def test_predict_button_for_districts(app, district):
    at = app.run()
    at.selectbox[0].set_value(district).run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    labels = {m.label: m.value for m in at.metric}
    assert labels["Predicted rice yield"].endswith("t/ha")
    assert any("Estimated production" in k for k in labels)


def test_app_rejects_zero_area(app):
    at = app.run()
    at.number_input[1].set_value(0.0)
    at.button[0].click().run()
    assert at.error and not at.exception and not at.metric


def test_app_warns_on_extrapolation(app):
    at = app.run()
    at.number_input[2].set_value(6000.0)
    at.button[0].click().run()
    assert at.warning and not at.exception


def test_performance_page_metrics_come_from_csv(app):
    at = app.run()
    at.sidebar.radio[0].set_value("Model performance").run()
    comp = data.load_comparison()
    ann = comp[(comp.model == "ANN (all features)") & (comp.split == "test")].iloc[0]
    shown = {m.label: m.value for m in at.metric}
    assert shown["ANN test RMSE"] == f"{ann.RMSE:.3f} t/ha"
    assert shown["ANN test R²"] == f"{ann.R2:.3f}"
