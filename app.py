"""CropYield AI - Streamlit app.  Run:  streamlit run app.py"""
import altair as alt
import pandas as pd
import streamlit as st

from src import data
from src.predictor import LABELS, ModelNotAvailable, Predictor, validate_inputs

GREEN, LIGHT, DARK = "#2E7D32", "#A5D6A7", "#1B5E20"

st.set_page_config(page_title="CropYield AI", page_icon="🌾", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem;}
  .hero {background: linear-gradient(90deg, #2E7D32, #66BB6A); color: white; padding: 1.2rem 1.5rem;
         border-radius: 12px; margin-bottom: 1rem;}
  .hero h1 {margin: 0; color: white; font-size: 2rem;}
  .hero p {margin: .2rem 0 0 0; opacity: .95;}
  [data-testid="stMetric"] {background: #EEF6EC; border-radius: 10px; padding: .6rem .9rem;
                            border-left: 5px solid #2E7D32;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading trained model...")
def get_predictor():
    return Predictor()


@st.cache_data
def get_dataset():
    return data.load_dataset()


@st.cache_data
def get_meta():
    return data.load_metadata()


@st.cache_data
def get_comparison():
    return data.load_comparison()


@st.cache_data
def get_predictions():
    return data.load_predictions()


def pretty(d):
    return d.title()


try:
    df = get_dataset()
    meta = get_meta()
except Exception:
    st.error("Project data files were not found. Run the earlier pipeline steps (preprocess.py, train_models.py) first.")
    st.stop()

districts = sorted(df.district.unique())

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🌾 CropYield AI")
page = st.sidebar.radio("Navigate", ["Predict yield", "Data & trends", "Model performance", "About & limitations"])
st.sidebar.markdown("---")
st.sidebar.caption("Academic prototype - Soft Computing mini project.\n\nNot a production forecasting system.")

st.markdown("""
<div class="hero"><h1>🌾 CropYield AI</h1>
<p><b>ANN-Based Rice Yield Prediction for Maharashtra</b></p>
<p>Estimates Kharif rice yield (tonnes/hectare) for a Maharashtra district from the agricultural year, cultivated
area and June-September weather, using an Artificial Neural Network trained on district-level records from 1997-2019.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------ pages
def page_predict():
    st.subheader("1. Enter conditions")
    district = st.selectbox("District", districts, format_func=pretty,
                            help="Defaults below are this district's historical medians (1997-2019).")
    sub = df[df.district == district]
    ref = df  # ranges for validation warnings come from the actual dataset

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        year = c1.number_input(LABELS["year"], 1997, 2035, int(df.year.max()), 1, key=f"year_{district}",
                               help="2019 means crop year 2019-20 (Kharif sown June-July 2019).")
        area = c2.number_input(LABELS["area_ha"], 0.0, 1_000_000.0, float(sub.area_ha.median()), 100.0,
                               key=f"area_{district}", help="Area under Kharif rice, in hectares.")
        rain = c3.number_input(LABELS["rain_kharif_mm"], 0.0, 6000.0, float(round(sub.rain_kharif_mm.median(), 1)), 10.0,
                               key=f"rain_{district}", help="Total rainfall June-September, in millimetres.")
        c4, c5, _ = st.columns(3)
        temp = c4.number_input(LABELS["temp_kharif_c"], 0.0, 50.0, float(round(sub.temp_kharif_c.median(), 2)), 0.1,
                               key=f"temp_{district}", help="Mean temperature June-September, in °C.")
        rh = c5.number_input(LABELS["rh_kharif_pct"], 0.0, 100.0, float(round(sub.rh_kharif_pct.median(), 2)), 0.5,
                             key=f"rh_{district}", help="Mean relative humidity June-September, in %.")
        submitted = st.form_submit_button("🌱 Predict Yield", type="primary", width="stretch")

    if not submitted:
        st.info("Choose inputs and press **Predict Yield**.")
        return
    values = dict(district=district, year=year, area_ha=area, rain_kharif_mm=rain, temp_kharif_c=temp,
                  rh_kharif_pct=rh)
    errors, warns = validate_inputs(values, ref, districts)
    if errors:
        for e in errors:
            st.error(e)
        return
    try:
        predictor = get_predictor()
        pred = predictor.predict(values)
    except ModelNotAvailable as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("Prediction failed because of an unexpected error. Please check the inputs and try again.")
        return

    st.subheader("2. Result")
    shown = max(pred, 0.0)
    base = predictor.baseline_for(district)
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted rice yield", f"{shown:.2f} t/ha")
    m2.metric("Estimated production (estimate)", f"{shown * area:,.0f} t",
              help="Predicted yield x entered area. A rough estimate only.")
    m3.metric(f"{pretty(district)} historical average", f"{base:.2f} t/ha",
              delta=f"{shown - base:+.2f} t/ha vs average", help="Mean yield in the training years 1997-2015.")
    if pred < 0:
        st.warning("The model's raw output was below zero; it is shown as 0. Inputs are probably unrealistic.")
    for w in warns:
        st.warning(w)
    st.caption("This is a statistical estimate from a small academic model (see *Model performance* for its test "
               "accuracy). Weather inputs should be the observed June-September values.")


def page_trends():
    st.subheader("Historical Kharif rice yield")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(df)}")
    c2.metric("Districts", f"{df.district.nunique()}")
    c3.metric("Years", f"{df.year.min()}-{df.year.max()}")

    pick = st.multiselect("Overlay districts (optional)", districts, format_func=pretty, max_selections=5)
    state = df.groupby("year")["yield_t_ha"].mean().reset_index().assign(series="All districts (mean)")
    sel = df[df.district.isin(pick)][["year", "yield_t_ha", "district"]].rename(columns={"district": "series"})
    sel["series"] = sel["series"].map(pretty)
    lines = pd.concat([state, sel])
    chart = alt.Chart(lines).mark_line(point=True).encode(
        x=alt.X("year:O", title="Agricultural year (start)"), y=alt.Y("yield_t_ha:Q", title="Yield (t/ha)"),
        color=alt.Color("series:N", title="", scale=alt.Scale(range=[DARK, "#F9A825", "#1E88E5", "#8E24AA", "#E53935", "#00897B"])),
        tooltip=["series", "year", alt.Tooltip("yield_t_ha:Q", format=".2f")]).properties(height=330)
    st.altair_chart(chart, width="stretch")

    st.subheader("District-wise average yield")
    avg = df.groupby("district")["yield_t_ha"].agg(["mean", "count"]).reset_index()
    avg["district"] = avg["district"].map(pretty)
    bar = alt.Chart(avg).mark_bar(color=GREEN).encode(
        y=alt.Y("district:N", sort="-x", title=""), x=alt.X("mean:Q", title="Mean yield 1997-2019 (t/ha)"),
        tooltip=["district", alt.Tooltip("mean:Q", format=".2f"), alt.Tooltip("count:Q", title="records")]
    ).properties(height=620)
    st.altair_chart(bar, width="stretch")
    st.caption("Averages include districts with very small rice areas, where reported yield is coarsely rounded.")


def page_performance():
    comp = get_comparison()
    st.subheader("Model performance (from Phase 3 evaluation)")
    sp = meta["split"]
    st.caption(f"Chronological split: train {sp['train']} ({sp['n_train']} rows), validation {sp['val']} "
               f"({sp['n_val']}), test {sp['test']} ({sp['n_test']}). Saved ANN = best of 5 seeds on validation loss.")

    def row(model, split):
        return comp[(comp.model == model) & (comp.split == split)].iloc[0]

    ann, base, lin = row("ANN (all features)", "test"), row("District-mean baseline", "test"), row("Linear Regression", "test")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ANN test RMSE", f"{ann.RMSE:.3f} t/ha")
    c2.metric("ANN test R²", f"{ann.R2:.3f}")
    c3.metric("Baseline test RMSE", f"{base.RMSE:.3f} t/ha")
    c4.metric("Linear Reg. test RMSE", f"{lin.RMSE:.3f} t/ha")
    if ann.RMSE >= min(base.RMSE, lin.RMSE):
        st.warning("On the 2018-2019 test set the ANN did **not** outperform the simple district-mean baseline or Linear "
                   "Regression. It was better on the validation years. With only 51 test rows these results are noisy.")
    else:
        st.success("On the test set the ANN outperformed both simple baselines.")

    show = comp[["model", "split", "MAE", "MSE", "RMSE", "R2", "n"]].copy()
    show["model"] = show["model"].str.replace("District-mean baseline", "District-mean baseline (Model A)")
    st.dataframe(show, hide_index=True, width="stretch")

    long = comp[["model", "split", "RMSE"]]
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X("model:N", title="", axis=alt.Axis(labelAngle=-25)), y=alt.Y("RMSE:Q", title="RMSE (t/ha, lower is better)"),
        color=alt.Color("split:N", scale=alt.Scale(range=[LIGHT, DARK])), xOffset="split:N",
        tooltip=["model", "split", alt.Tooltip("RMSE:Q", format=".3f")]).properties(height=330)
    st.altair_chart(chart, width="stretch")

    st.subheader("Actual vs predicted yield (ANN)")
    preds = get_predictions()
    split = st.radio("Data subset", ["test", "val"], horizontal=True,
                     format_func=lambda s: {"test": "Test (2018-2019)", "val": "Validation (2016-2017)"}[s])
    p = preds[preds.split == split].copy()
    p["district"] = p["district"].map(pretty)
    hi = float(max(p.actual_yield_t_ha.max(), p.pred_ann_all_features.max())) * 1.05
    pts = alt.Chart(p).mark_circle(size=60, color=GREEN, opacity=.8).encode(
        x=alt.X("actual_yield_t_ha:Q", title="Actual yield (t/ha)", scale=alt.Scale(domain=[0, hi])),
        y=alt.Y("pred_ann_all_features:Q", title="Predicted yield (t/ha)", scale=alt.Scale(domain=[0, hi])),
        tooltip=["district", "year", alt.Tooltip("actual_yield_t_ha:Q", format=".2f"),
                 alt.Tooltip("pred_ann_all_features:Q", format=".2f")])
    diag = alt.Chart(pd.DataFrame({"x": [0, hi], "y": [0, hi]})).mark_line(color="gray", strokeDash=[5, 4]).encode(x="x", y="y")
    st.altair_chart((pts + diag).properties(height=380), width="stretch")
    st.caption("Dashed line = perfect prediction.")

    img = data.loss_curve_image()
    if img:
        with st.expander("Training vs validation loss"):
            st.image(img)


def page_about():
    st.subheader("Model")
    st.markdown(f"**Architecture:** `{meta['ann']['architecture']}`")
    st.markdown(f"Best epoch {meta['ann']['best_epoch']} of {meta['ann']['epochs_run']} run (early stopping on validation loss). "
                f"Library versions: TensorFlow {meta['versions']['tensorflow']}, scikit-learn {meta['versions']['sklearn']}.")
    st.markdown("**Preprocessing:** numeric inputs standardised (StandardScaler); district one-hot encoded; "
                "both fitted on training years only.")
    st.subheader("Input parameters")
    st.table(pd.DataFrame({
        "Input": ["District", LABELS["year"], LABELS["area_ha"], LABELS["rain_kharif_mm"], LABELS["temp_kharif_c"],
                  LABELS["rh_kharif_pct"]],
        "Observed range in data": ["34 districts", f"{df.year.min()}-{df.year.max()}",
                                   f"{df.area_ha.min():,.0f}-{df.area_ha.max():,.0f}",
                                   f"{df.rain_kharif_mm.min():,.0f}-{df.rain_kharif_mm.max():,.0f}",
                                   f"{df.temp_kharif_c.min():.1f}-{df.temp_kharif_c.max():.1f}",
                                   f"{df.rh_kharif_pct.min():.1f}-{df.rh_kharif_pct.max():.1f}"]}))
    st.caption("Target: rice yield in tonnes/hectare. Production is deliberately not an input (it would leak the answer).")
    st.subheader("Data sources")
    st.markdown("""
- **Crop data:** district-level Kharif rice area, production and yield for Maharashtra, 1997-98 to 2019-20
  (India Agriculture Crop Production dataset supplied by the project team). 642 usable records after cleaning.
- **Weather:** NASA POWER daily MERRA-2 reanalysis (precipitation, 2 m temperature, relative humidity), aggregated
  to June-September per district-year.
- **District locations:** OpenStreetMap Nominatim (centre of each district boundary).
""")
    st.subheader("Limitations")
    st.markdown("""
- Only 642 observations; test results rest on two years (51 rows).
- Four district pairs share one NASA grid cell (identical weather): Hingoli/Parbhani, Akola/Amravati, Nagpur/Wardha, Bhandara/Gondia.
- Each district's weather is a single geographic point, not a district average.
- Correlations in historical data do not establish causation.
- Weather inputs cover June-September and are known only after the season, so this is not an early forecast.
- No soil, irrigation or fertiliser information is used.
- The ANN did not beat simple baselines on the test set (see *Model performance*).
""")
    st.info("**Disclaimer:** academic prototype for a Soft Computing mini project. Do not use it for farming, "
            "insurance, pricing or policy decisions.")


{"Predict yield": page_predict, "Data & trends": page_trends, "Model performance": page_performance,
 "About & limitations": page_about}[page]()
