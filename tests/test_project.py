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


def _num(at, label):
    """Find a number input by its label (robust to layout changes)."""
    return next(n for n in at.number_input if n.label.startswith(label))


def test_app_rejects_zero_area(app):
    at = app.run()
    _num(at, "Cultivated area").set_value(0.0)
    at.button[0].click().run()
    assert at.error and not at.exception and not at.metric


def test_app_warns_on_extrapolation(app):
    at = app.run()
    _num(at, "Kharif rainfall").set_value(6000.0)
    at.button[0].click().run()
    assert at.warning and not at.exception


def test_performance_page_metrics_come_from_csv(app):
    at = app.run()
    at.sidebar.radio[0].set_value("Model performance").run()
    comp = data.load_comparison()
    ann = comp[(comp.model == "ANN (all features)") & (comp.split == "test")].iloc[0]
    shown = {m.label: m.value for m in at.metric}
    assert shown["ANN test RMSE"] == f"{ann.RMSE:.4f} t/ha"
    assert shown["ANN test R²"] == f"{ann.R2:.4f}"


# ======================================================================= Phase 6 features
# ---- 1. district map ----
def test_boundaries_cover_every_district(df):
    gj = data.load_boundaries()
    assert gj is not None and len(gj["features"]) == 34
    assert set(f["properties"]["district"] for f in gj["features"]) == set(df.district)


def test_boundary_vertices_unchanged_only_reordered():
    """The loader may only reverse ring order for the map renderer; no coordinate may change."""
    import json
    raw = json.loads((ROOT / "data" / "raw" / "maharashtra_district_boundaries.geojson").read_text())
    ready = data.load_boundaries()

    def pts(g):
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        return sorted(tuple(p[:2]) for poly in polys for ring in poly for p in ring)
    for a, b in zip(raw["features"], ready["features"]):
        assert pts(a["geometry"]) == pts(b["geometry"])
        polys = b["geometry"]["coordinates"] if b["geometry"]["type"] == "MultiPolygon" else [b["geometry"]["coordinates"]]
        assert all(data._signed_area([tuple(p[:2]) for p in poly[0]]) < 0 for poly in polys)   # exteriors clockwise


def test_boundaries_missing_file_is_handled(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    assert data.load_boundaries() is None and data.load_district_points() is None


def test_map_tab_renders_and_highlights(app):
    at = app.run()
    at.sidebar.radio[0].set_value("Data & trends").run()
    at.selectbox(key="map_pick").set_value("NAGPUR").run()
    assert not at.exception and not at.error
    assert any(m.label == "Nagpur mean yield" for m in at.metric)
    expected = df_mean("NAGPUR")
    assert next(m.value for m in at.metric if m.label == "Nagpur mean yield") == f"{expected:.2f} t/ha"


def df_mean(district):
    d = data.load_dataset()
    return d[d.district == district].yield_t_ha.mean()


# ---- 2. permutation importance ----
@pytest.fixture(scope="module")
def importance(df, predictor):
    from src import interpret
    return interpret.permutation_importance(predictor, interpret.evaluation_rows(df, data.load_metadata()))


def test_importance_structure_and_baseline(importance):
    assert set(importance.feature) == {"district", "year", "area_ha", "rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"}
    assert np.isfinite(importance[["importance", "std"]].to_numpy()).all()
    assert importance.attrs["n_rows"] == 51
    comp = data.load_comparison()
    saved = comp[(comp.model == "ANN (all features)") & (comp.split == "test")].RMSE.iloc[0]
    assert abs(importance.attrs["baseline_rmse"] - saved) < 1e-3     # uses the saved ANN on the saved test years


def test_importance_is_reproducible_and_uses_test_years_only(df, predictor, importance):
    from src import interpret
    rows = interpret.evaluation_rows(df, data.load_metadata())
    assert rows.year.min() == 2018 and rows.year.max() == 2019
    again = interpret.permutation_importance(predictor, rows)
    assert np.allclose(again.importance, importance.importance)


def test_predict_page_shows_understanding_section(app):
    at = app.run()
    assert any("Understanding the Prediction" in m.value for m in at.markdown)
    assert any("not causation" in m.value.lower() for m in at.markdown)


# ---- 3. downloadable report ----
def test_report_columns_values_and_units():
    from src import report
    vals = dict(district="PUNE", year=2019, area_ha=1000.0, rain_kharif_mm=1200.5, temp_kharif_c=25.5, rh_kharif_pct=80.0)
    r = report.build_report(vals, 2.0, 1.3)
    assert list(r.columns) == report.COLUMNS and len(r) == 1
    row = r.iloc[0]
    assert row["District"] == "Pune" and row["Agricultural year"] == 2019
    assert row["Predicted rice yield (tonnes/hectare)"] == 2.0 and row["Estimated production (tonnes)"] == 2000.0
    assert all(("(" in c) or c in ("District", "Agricultural year") for c in r.columns)    # units present
    text = report.to_csv_bytes(r).decode("utf-8-sig")
    assert "°C" in text and text.splitlines()[1].startswith("Pune,2019")
    assert report.file_name(vals) == "cropyield_prediction_pune_2019.csv"


@pytest.mark.parametrize("district", ["PUNE", "KOLHAPUR", "GONDIA"])
def test_download_button_serves_the_actual_prediction(app, district):
    at = app.run()
    at.selectbox[0].set_value(district).run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.get("download_button")) == 1
    shown = float(next(m.value for m in at.metric if m.label == "Predicted rice yield").split()[0])
    # the report is built from the same call as the metric card: rebuild it and compare to the card
    from src import report
    p = Predictor()
    d = data.load_dataset()
    sub = d[d.district == district]
    vals = dict(district=district, year=int(d.year.max()), area_ha=float(sub.area_ha.median()),
                rain_kharif_mm=float(round(sub.rain_kharif_mm.median(), 1)),
                temp_kharif_c=float(round(sub.temp_kharif_c.median(), 2)),
                rh_kharif_pct=float(round(sub.rh_kharif_pct.median(), 2)))
    rep = report.build_report(vals, max(p.predict(vals), 0.0), p.baseline_for(district))
    assert abs(rep.iloc[0]["Predicted rice yield (tonnes/hectare)"] - shown) < 0.006


def test_no_download_button_when_input_invalid(app):
    at = app.run()
    _num(at, "Cultivated area").set_value(0.0)
    at.button[0].click().run()
    assert len(at.get("download_button")) == 0
