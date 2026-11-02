"""CropYield AI - Streamlit dashboard.  Run:  streamlit run app.py"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config as C
from src import auth, charts, data, interpret, login, report, ui
from src.predictor import LABELS, ModelNotAvailable, Predictor, validate_inputs
from src.ui import AMBER, BLUE, FOREST, GREEN, MINT, MODEL_COLORS, SLATE

st.set_page_config(page_title="CropYield AI", page_icon="🌱", layout="wide", initial_sidebar_state="auto")
ui.inject_css()

# --- demo authentication gate: nothing below (data, model, pages) runs for an unauthenticated visitor ---
if not auth.is_authenticated():
    login.render()
    st.stop()
profile = auth.current_user()

ORDER = ["District-mean baseline", "Linear Regression", "Linear Regression (no weather)",
         "ANN (all features)", "ANN (no weather)"]
PRED_COL = {"District-mean baseline": "pred_district_mean_baseline", "Linear Regression": "pred_linear_regression",
            "Linear Regression (no weather)": "pred_linear_regression_no_weather",
            "ANN (all features)": "pred_ann_all_features", "ANN (no weather)": "pred_ann_no_weather"}
SPLIT_NAME = {"val": "Validation (2016-17)", "test": "Test (2018-19)"}


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


@st.cache_data
def get_seeds():
    return data.load_seed_results()


def pretty(d):
    return d.title()


try:
    df = get_dataset()
    meta = get_meta()
    comp = get_comparison()
except Exception:
    st.error("Project data files were not found. Run the earlier pipeline steps (preprocess.py, train_models.py) first.")
    st.stop()

districts = sorted(df.district.unique())
split_info = meta["split"]
base_means = meta["district_mean_baseline"]["means"]
N_INPUT = len(meta["preprocessing"]["encoded_feature_names"])


@st.cache_data(show_spinner="Computing feature importance...")
def get_importance():
    """Permutation importance of the saved ANN on the held-out test years (cached for the whole session)."""
    res = interpret.permutation_importance(get_predictor(), interpret.evaluation_rows(df, meta))
    return res.copy(), dict(res.attrs)


def crow(model, split):
    return comp[(comp.model == model) & (comp.split == split)].iloc[0]


# --------------------------------------------------------------------------- sidebar
NAV = {"Dashboard": ":material/dashboard: Dashboard",
       "Predict Yield": ":material/eco: Predict Yield",
       "Data & Trends": ":material/insights: Data & Trends",
       "Model Performance": ":material/model_training: Model Performance"}


def go_predict():
    st.session_state["nav"] = "Predict Yield"


with st.sidebar:
    ui.sidebar_brand()
    ui.sidebar_profile(profile)
    page = st.radio("Navigate", list(NAV), format_func=NAV.get, label_visibility="collapsed", key="nav")
    ui.sidebar_footer([("Project", "Soft Computing mini project"), ("Focus", f"{C.CROP.title()} · {C.SEASON.title()} · Maharashtra"),
                       ("Data", f"{df.year.min()}–{df.year.max()} · {len(df)} records"),
                       ("Model", f"ANN {N_INPUT} → 64 → 32 → 1")],
                      "Academic prototype. Not for real farming, insurance or policy decisions.")
    st.button("Logout", icon=":material/logout:", key="logout_btn", on_click=auth.logout, width="stretch",
              help="Sign out and return to the login screen.")


# ============================================================================= DASHBOARD
def page_dashboard(profile):
    latest = int(df.year.max())
    ly = df[df.year == latest]
    ann, base, lin = crow("ANN (all features)", "test"), crow("District-mean baseline", "test"), crow("Linear Regression", "test")

    hl, hr = st.columns([4, 1], vertical_alignment="center")
    with hl:
        ui.greeting(f"Welcome, {profile['name']}!", "Welcome to CropYield AI · AI-powered rice yield prediction for Maharashtra")
    with hr:
        st.button("🌱  Start a prediction", type="primary", on_click=go_predict, width="stretch", key="cta_predict")

    ui.profile_card(profile)
    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    ui.kpi_cards([("📄", "Total historical records", f"{len(df)}", "district-year observations", "green"),
                  ("🗺️", "Districts", f"{df.district.nunique()}", "across Maharashtra", "blue"),
                  ("🌾", "Crop", C.CROP.title(), "yield in tonnes per hectare", "amber"),
                  ("🌧️", "Season", C.SEASON.title(), "June–September weather", "purple")])
    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3, gap="medium")
    with b1:
        with st.container(key="card_season"):
            ui.card_title(f"{latest} season", "latest year")
            st.markdown(ui.pill("Maharashtra · Kharif rice") +
                        f'<div class="bigv">{ly.yield_t_ha.mean():.2f} t/ha</div>'
                        f'<div class="bigs">Mean rice yield across {ly.district.nunique()} districts</div>'
                        + ui.stat_row([("Rainfall", f"{ly.rain_kharif_mm.mean():,.0f} mm"),
                                       ("Temperature", f"{ly.temp_kharif_c.mean():.1f} °C"),
                                       ("Humidity", f"{ly.rh_kharif_pct.mean():.0f} %")]), unsafe_allow_html=True)
    with b2:
        with st.container(key="card_gauge"):
            ui.card_title("Model accuracy", "test R², 2018-19")
            ui.gauge(float(ann.R2), 1.0, f"{ann.R2:.2f}", "ANN test R²",
                     [("ANN", f"{ann.R2:.3f}", FOREST), ("Linear Regression", f"{lin.R2:.3f}", BLUE),
                      ("District mean", f"{base.R2:.3f}", AMBER)])
            st.caption("The ANN did not outperform the simpler models on the test years (see Model Performance).")
    with b3:
        with st.container(key="card_info"):
            ui.card_title("Project information")
            ui.info_list([("🗓️", "Dataset coverage", f"{df.year.min()}–{df.year.max()} · {len(df)} records · "
                                                      f"{df.district.nunique()} districts", "green"),
                          ("🌦️", "Weather parameters", "Rainfall, temperature and humidity (NASA POWER, June–September)", "blue"),
                          ("🧠", "Prediction model", f"Artificial Neural Network ({N_INPUT} → 64 → 32 → 1)", "purple")])

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        with st.container(key="card_dash_trend"):
            ui.card_title("Historical rice yield trend", "mean across districts")
            ui.show(charts.yield_trend_smooth(df), 350)
    with c2:
        with st.container(key="card_dash_map"):
            ui.card_title("District yield map", "mean yield 1997-2019")
            chart, _ = charts.district_map(df, labels=False, legend=False)
            if chart is None:
                st.info("District outlines not downloaded yet (run fetch_boundaries.py).")
            else:
                ui.show(chart, 350)
                st.caption("Darker green = higher average yield. Open Data & Trends for details.")

    d1, d2 = st.columns([3, 2], gap="medium")
    with d1:
        with st.container(key="card_dash_bars"):
            ui.card_title("District-wise average rice yield", "t/ha")
            ui.show(charts.district_bars(df), 340)
            st.caption("Amber bars: districts with fewer than 15 records (less reliable averages).")
    with d2:
        with st.container(key="card_dash_rain"):
            ui.card_title("Rainfall vs rice yield", "June–September")
            chart, r = charts.rain_vs_yield(df)
            ui.show(chart, 340)
            st.caption(f"Pearson r = {r:+.2f} across all records. Much of this reflects differences between districts, "
                       "not year-to-year effects.")

    stats = df.groupby("district")["yield_t_ha"].agg(["mean", "min", "max", "count"])
    t1, t2 = st.columns(2, gap="medium")
    for col, key, title, sub in [(t1, "card_top", "Highest-yielding districts", stats.sort_values("mean", ascending=False).head(5)),
                                 (t2, "card_low", "Lowest-yielding districts", stats.sort_values("mean").head(5))]:
        with col:
            with st.container(key=key):
                ui.card_title(title, "mean yield")
                ui.html_table(["District", "Mean (t/ha)", "Range (t/ha)", "Records"],
                              [(pretty(d), f"{r['mean']:.2f}", f"{r['min']:.2f}–{r['max']:.2f}", int(r["count"]))
                               for d, r in sub.iterrows()])


# ============================================================================ PREDICT
def history_chart(district, year=None, estimate=None):
    sub = df[df.district == district].sort_values("year")
    base = float(base_means.get(district, np.nan))
    x = alt.X("year:Q", title="Agricultural year", axis=alt.Axis(format="d", tickMinStep=1),
              scale=alt.Scale(zero=False, nice=False, padding=8))
    line = alt.Chart(sub).mark_line(color=GREEN, strokeWidth=2.5, point=alt.OverlayMarkDef(color=GREEN, size=50)).encode(
        x=x, y=alt.Y("yield_t_ha:Q", title="Yield (t/ha)", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("year:Q", format="d"), alt.Tooltip("yield_t_ha:Q", title="Reported yield", format=".2f"),
                 alt.Tooltip("area_ha:Q", title="Area (ha)", format=",.0f")])
    layers = [line]
    if not np.isnan(base):
        layers.append(alt.Chart(pd.DataFrame({"y": [base]})).mark_rule(color=AMBER, strokeDash=[6, 4], strokeWidth=2)
                      .encode(y="y:Q"))
    if estimate is not None:
        pt = pd.DataFrame({"year": [year], "yield_t_ha": [estimate]})
        layers.append(alt.Chart(pt).mark_point(shape="diamond", size=300, filled=True, color=FOREST, opacity=1,
                                               stroke="white", strokeWidth=2).encode(x=x, y="yield_t_ha:Q",
                                               tooltip=[alt.Tooltip("year:Q", format="d"),
                                                        alt.Tooltip("yield_t_ha:Q", title="ANN estimate", format=".2f")]))
    return alt.layer(*layers)


def understanding_section(values=None):
    ui.section("Understanding the Prediction",
               "What each input represents, and how much the saved ANN's accuracy depends on it.")
    num = ["year", "area_ha", "rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"]
    corr = df[num].corrwith(df.yield_t_ha)
    dm = df.groupby("district").yield_t_ha.mean()
    roles = [
        ("District", f"One of {len(districts)} districts. District average yields in the data range from {dm.min():.2f} "
                     f"to {dm.max():.2f} t/ha, so this input carries a lot of information."),
        ("Agricultural year", f"Start year of the Kharif season (correlation with yield {corr.year:+.2f}). Years after "
                              f"{df.year.max()} are outside the data."),
        ("Cultivated area", f"Hectares under Kharif rice (correlation {corr.area_ha:+.2f}). Very small areas have "
                            "coarsely rounded yields."),
        ("Rainfall", f"Total June–September rainfall at one NASA POWER grid point (correlation {corr.rain_kharif_mm:+.2f})."),
        ("Temperature", f"Mean June–September temperature (correlation {corr.temp_kharif_c:+.2f})."),
        ("Humidity", f"Mean June–September relative humidity (correlation {corr.rh_kharif_pct:+.2f})."),
    ]
    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.container(key="card_roles"):
            ui.card_title("What each input is")
            ui.definition_list(roles)
            st.caption("Correlations are over all 642 records and mostly reflect differences between districts.")
    with right:
        with st.container(key="card_importance"):
            ui.card_title("How much the ANN relies on each input", "permutation importance")
            try:
                imp, info = get_importance()
            except Exception:
                st.info("Feature importance is not available because the saved model could not be loaded.")
                imp = None
            if imp is not None:
                order = imp.label.tolist()
                bars = alt.Chart(imp).mark_bar(color=GREEN, cornerRadiusEnd=4).encode(
                    y=alt.Y("label:N", sort=order, title=""),
                    x=alt.X("importance:Q", title="Increase in test RMSE when the input is shuffled (t/ha)"),
                    tooltip=[alt.Tooltip("label:N", title="Input"),
                             alt.Tooltip("importance:Q", title="RMSE increase (t/ha)", format="+.3f"),
                             alt.Tooltip("std:Q", title="Std over shuffles", format=".3f")])
                err = alt.Chart(imp.assign(lo=imp.importance - imp["std"], hi=imp.importance + imp["std"])).mark_rule(
                    color="#14281D", strokeWidth=1.5).encode(y=alt.Y("label:N", sort=order), x="lo:Q", x2="hi:Q")
                zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=SLATE).encode(x="x:Q")
                ui.show(bars + err + zero, 270)
                st.caption(f"Computed with the saved ANN on the {info['n_rows']} held-out test rows (2018-19); baseline test "
                           f"RMSE {info['baseline_rmse']:.4f} t/ha; {info['n_repeats']} random shuffles per input; "
                           "whiskers show ± 1 standard deviation over shuffles.")
    ui.callout("warn", "Importance is not causation. ",
               "A bar shows how much the ANN's predictions on these test rows get worse when an input is scrambled, so it "
               "indicates predictive contribution only; it does not show that the input causes yield changes, and it does not "
               "explain the network's internal reasoning. Inputs are correlated (humidity and rainfall differ mostly between "
               "districts), so they share credit: a short bar for District does not mean districts do not matter. Bars near "
               "or below zero mean shuffling that input did not hurt accuracy on these 51 rows. The test set is small, so "
               "the ranking is approximate.")
    if values:
        sub = df[df.district == values["district"]]
        decimals = {"year": 0, "area_ha": 0, "rain_kharif_mm": 1, "temp_kharif_c": 2, "rh_kharif_pct": 2}
        rows = []
        for c in num:
            k = decimals[c]
            mine, typical = round(float(values[c]), k), round(float(sub[c].median()), k)
            diff = round(mine - typical, k)
            rows.append((LABELS[c], f"{mine:,.{k}f}", f"{typical:,.{k}f}", "0" if diff == 0 else f"{diff:+,.{k}f}"))
        with st.container(key="card_compare"):
            ui.card_title("Your inputs vs this district's typical values")
            st.dataframe(pd.DataFrame(rows, columns=["Input", "Your value", f"{pretty(values['district'])} median", "Difference"]),
                         hide_index=True, width="stretch")
            st.caption("Large differences mean the estimate is based on unusual conditions for this district.")


def page_predict(profile):
    ui.page_header("", "Predict Yield", "Enter a district and the season's conditions to get an ANN estimate of Kharif rice yield.",
                   f"ANN · {N_INPUT}→64→32→1")
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(key="card_inputs"):
            ui.section("1 · Location", "Defaults below are this district's historical medians.")
            farm_district = profile["district"].upper()
            district = st.selectbox("District", districts, index=districts.index(farm_district) if farm_district in districts else 0,
                                    format_func=pretty, help="Maharashtra districts with Kharif rice records (1997-2019). "
                                    "Prefilled with your registered district.")
            sub = df[df.district == district]
            land = float(profile["land_area_ha"])
            ui.chips([f"{len(sub)} records", f"{int(sub.year.min())}-{int(sub.year.max())}",
                      f"typical area {sub.area_ha.median():,.0f} ha"], muted=True)
            in_scope = profile["crop"].lower() == C.CROP and profile["season"].lower() == C.SEASON
            ui.chips([f"👤 {profile['name']}", f"🌾 {profile['crop']} · {profile['season']}", f"📐 your land {land:g} ha"])
            if not in_scope:
                st.warning(f"The model only covers {C.CROP.title()} in the {C.SEASON.title()} season.")
            use_land = st.checkbox(f"Use my land area ({land:g} ha) as the model's area input", value=False, key="use_land",
                                   help="Off by default: the ANN was trained on district-level rice areas "
                                        f"({df.area_ha.min():,.0f}–{df.area_ha.max():,.0f} ha), so a single farm is outside its range.")
            if use_land:
                st.caption("The model's area input expects district-level hectares; a single farm is far smaller than anything it "
                           "was trained on, so expect an extrapolation warning and a less reliable yield estimate.")
            ui.section("2 · Season conditions", "Rainfall, temperature and humidity are June–September values.")
            with st.form("predict_form", border=False):
                r1a, r1b = st.columns(2)
                year = r1a.number_input(LABELS["year"], 1997, 2035, int(df.year.max()), 1, key=f"year_{district}",
                                        help="2019 means crop year 2019-20 (Kharif sown June-July 2019).")
                area = r1b.number_input(LABELS["area_ha"], 0.0, 1_000_000.0, land if use_land else float(sub.area_ha.median()),
                                        0.5 if use_land else 100.0, key=f"area_{district}_{int(use_land)}",
                                        help="Area under Kharif rice, in hectares.")
                r2a, r2b = st.columns(2)
                rain = r2a.number_input(LABELS["rain_kharif_mm"], 0.0, 6000.0,
                                        float(round(sub.rain_kharif_mm.median(), 1)), 10.0, key=f"rain_{district}",
                                        help="Total rainfall June-September, in millimetres.")
                temp = r2b.number_input(LABELS["temp_kharif_c"], 0.0, 50.0, float(round(sub.temp_kharif_c.median(), 2)),
                                        0.1, key=f"temp_{district}", help="Mean temperature June-September, in °C.")
                r3a, _ = st.columns(2)
                rh = r3a.number_input(LABELS["rh_kharif_pct"], 0.0, 100.0, float(round(sub.rh_kharif_pct.median(), 2)),
                                      0.5, key=f"rh_{district}", help="Mean relative humidity June-September, in %.")
                st.markdown("&nbsp;", unsafe_allow_html=True)
                submitted = st.form_submit_button("🌱  Predict Yield", type="primary", width="stretch")

    estimate_pt = None
    last_values = None
    with right:
        with st.container(key="card_result"):
            ui.section("3 · Estimate")
            if not submitted:
                ui.callout("info", "Ready when you are. ", "Choose inputs and press Predict Yield to see the estimated "
                           "yield, an estimated production and how it compares with the district's history.")
                ui.steps(["Pick a district", "Adjust the season conditions", "Press Predict Yield"])
            else:
                values = dict(district=district, year=year, area_ha=area, rain_kharif_mm=rain, temp_kharif_c=temp,
                              rh_kharif_pct=rh)
                errors, warns = validate_inputs(values, df, districts)
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    try:
                        predictor = get_predictor()
                        pred = predictor.predict(values)
                    except ModelNotAvailable as exc:
                        st.error(str(exc))
                        pred = None
                    except Exception:
                        st.error("Prediction failed because of an unexpected error. Please check the inputs and try again.")
                        pred = None
                    if pred is not None:
                        shown = max(pred, 0.0)
                        base = predictor.baseline_for(district)
                        estimate_pt = (year, shown)
                        for w in warns:
                            st.warning(w)
                        if pred < 0:
                            st.warning("The model's raw output was below zero; it is shown as 0. Inputs are probably unrealistic.")
                        st.metric("Predicted rice yield", f"{shown:.2f} t/ha")
                        st.metric("Estimated production (estimate)", f"{shown * area:,.0f} t",
                                  help="Predicted yield × entered area. A rough estimate only.")
                        if not use_land:
                            st.metric(f"Your {land:g} ha: est. production", f"{shown * land:,.2f} t",
                                      help="Predicted district-level yield × your registered land area. A rough estimate only.")
                        st.metric(f"{pretty(district)} historical average", f"{base:.2f} t/ha",
                                  delta=f"{shown - base:+.2f} t/ha vs average",
                                  help="Mean yield in the training years 1997-2015.")
                        st.caption("Statistical estimate from a small academic model. See *Model Performance* for its "
                                   "test accuracy.")
                        last_values = values
                        rep = report.build_report(values, shown, base)
                        st.download_button("⬇  Download Prediction Report (CSV)", data=report.to_csv_bytes(rep),
                                           file_name=report.file_name(values), mime="text/csv", key="dl_report",
                                           on_click="ignore", width="stretch",
                                           help="One-row CSV with your inputs, the ANN estimate and the district average.")

    ui.section(f"Historical yield · {pretty(district)}",
               "Reported Kharif rice yield by year, with the district's training-period average.")
    with st.container(key="card_history"):
        ui.show(history_chart(district, *(estimate_pt or (None, None))) if estimate_pt else history_chart(district), 300)
        st.caption("Green line: reported yield · dashed amber line: district average 1997-2015"
                   + (" · diamond: your ANN estimate." if estimate_pt else "."))
    understanding_section(last_values)


# ============================================================================== TRENDS
def district_map_tab():
    ui.section("District-wise average rice yield", "Mean reported Kharif rice yield per district, 1997-2019. Hover a district "
               "for its name and value.")
    pick = st.selectbox("Highlight a district (optional)", ["All districts"] + districts,
                        format_func=lambda d: d if d == "All districts" else pretty(d), key="map_pick")
    chart, no_shape = charts.district_map(df, pick)
    if chart is None:
        ui.callout("warn", "Boundary data not found. ", "Run `python fetch_boundaries.py` to download the district outlines "
                   "(OpenStreetMap). The charts in the other tabs still work.")
        return
    with st.container(key="card_map"):
        ui.show(chart, 560)
        st.caption("Darker green = higher average yield. Outlines: current OpenStreetMap district boundaries (ODbL), simplified.")
    if pick != "All districts":
        stats = df.groupby("district")["yield_t_ha"].agg(["mean", "min", "max", "count"])
        yrs = df.groupby("district")["year"].agg(["min", "max"])
        r, y = stats.loc[pick], yrs.loc[pick]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{pretty(pick)} mean yield", f"{r['mean']:.2f} t/ha")
        m2.metric("Range (t/ha)", f"{r['min']:.2f}–{r['max']:.2f}")
        m3.metric("Records", f"{int(r['count'])}")
        m4.metric("Years", f"{int(y['min'])}-{int(y['max'])}")
    notes = ["Districts are coloured by their historical mean, so districts with few records (e.g. Akola has 2) are less reliable.",
             "Outlines show today's district boundaries; older records of districts later split (e.g. Thane before 2014, "
             "Bhandara before 1999) refer to their former, larger areas.",
             "Mumbai City and Mumbai Suburban have no rice records and are not shown."]
    if no_shape:
        notes.append("No boundary available for: " + ", ".join(pretty(d) for d in no_shape) + " (listed in other tabs only).")
    st.caption(" ".join(notes))


def page_trends():
    ui.page_header("", "Data & Trends", "Explore the Kharif rice records, weather and district patterns behind the model.",
                   f"{df.year.min()}–{df.year.max()}")
    ui.kpi_cards([("📄", "Records", f"{len(df)}", "district-year observations", "green"),
                  ("🗺️", "Districts", f"{df.district.nunique()}", "Maharashtra", "blue"),
                  ("📅", "Years", f"{df.year.min()}-{df.year.max()}", f"{df.year.nunique()} Kharif seasons", "purple"),
                  ("🌾", "Mean yield", f"{df.yield_t_ha.mean():.2f} t/ha", "all records", "amber")])
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    to, t1, t2, tm, tw = st.tabs(["Dataset overview", "Yield trends", "District comparison", "Yield map", "Weather"])

    with to:
        o1, o2 = st.columns([3, 2], gap="medium")
        with o1:
            with st.container(key="card_summary"):
                ui.card_title("Variables in the final dataset", "642 rows")
                spec = [("Rice yield", "yield_t_ha", "t/ha", 2), ("Cultivated area", "area_ha", "ha", 0),
                        ("Rainfall Jun-Sep", "rain_kharif_mm", "mm", 0), ("Mean temperature", "temp_kharif_c", "°C", 1),
                        ("Mean humidity", "rh_kharif_pct", "%", 1)]
                ui.html_table(["Variable", "Unit", "Mean", "Min", "Max"],
                              [(n, u, f"{df[c].mean():,.{k}f}", f"{df[c].min():,.{k}f}", f"{df[c].max():,.{k}f}")
                               for n, c, u, k in spec])
                st.caption("Yield is reported production ÷ area. Weather comes from NASA POWER (MERRA-2) at one point per "
                           "district; four district pairs share a grid cell.")
        with o2:
            with st.container(key="card_coverage"):
                ui.card_title("Records per year")
                ui.show(charts.records_per_year(df), 250)
        with st.container(key="card_rec_district"):
            ui.card_title("Records per district", "years with data")
            ui.show(charts.records_per_district(df), 620)
            st.caption("Some districts have short histories (Akola 2, Palghar 6, Buldhana 7) because of missing rows or "
                       "district reorganisation.")

    with t1:
        ui.section("Yield over time", "Yearly mean across districts, with the middle 50% of districts shaded.")
        pick = st.multiselect("Overlay districts (optional, up to 5)", districts, format_func=pretty, max_selections=5)
        with st.container(key="card_trend"):
            ui.show(charts.yield_trend(df, pick), 360)
            st.caption("Averages include districts with very small rice areas, where reported yield is coarsely rounded.")

    with t2:
        ui.section("District comparison", "Mean yield (dot) and min-max range (line) per district, 1997-2019.")
        s = df.groupby("district")["yield_t_ha"].agg(["mean", "min", "max", "count"]).reset_index()
        s["district"] = s["district"].map(pretty)
        s["history"] = np.where(s["count"] >= 15, "15+ records", "fewer than 15 records")
        y = alt.Y("district:N", sort=alt.EncodingSortField("mean", order="descending"), title="")
        rng = alt.Chart(s).mark_rule(strokeWidth=3, opacity=.45).encode(
            y=y, x=alt.X("min:Q", title="Yield (t/ha)"), x2="max:Q",
            color=alt.Color("history:N", title="", scale=alt.Scale(domain=["15+ records", "fewer than 15 records"],
                                                                    range=[GREEN, AMBER])))
        dot = alt.Chart(s).mark_circle(size=90, opacity=1).encode(
            y=y, x="mean:Q", color=alt.Color("history:N", legend=None, scale=alt.Scale(
                domain=["15+ records", "fewer than 15 records"], range=[FOREST, AMBER])),
            tooltip=["district", alt.Tooltip("mean:Q", format=".2f", title="Mean"), alt.Tooltip("min:Q", format=".2f"),
                     alt.Tooltip("max:Q", format=".2f"), alt.Tooltip("count:Q", title="Records")])
        with st.container(key="card_districts"):
            ui.show(rng + dot, 760)
            st.caption("Amber = districts with fewer than 15 records (e.g. Akola has only 2), so their averages are less reliable.")

    with tm:
        district_map_tab()

    with tw:
        ui.section("Weather vs yield", "Each dot is one district-year; the line is a straight-line fit.")
        within = st.checkbox("Compare years within each district (remove district averages)", value=False,
                             help="Subtracts each district's own mean, so only year-to-year differences remain.")
        d = df.copy()
        cols = ["rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"]
        if within:
            for c in cols + ["yield_t_ha"]:
                d[c] = d[c] - d.groupby("district")[c].transform("mean")
        info = {"rain_kharif_mm": "Rainfall Jun-Sep (mm)", "temp_kharif_c": "Mean temperature (°C)",
                "rh_kharif_pct": "Mean humidity (%)"}
        for col, c in zip(st.columns(3, gap="medium"), cols):
            r = d[c].corr(d["yield_t_ha"])
            xlab = info[c] + (" - change vs district avg" if within else "")
            ylab = "Yield (t/ha)" + (" - change vs district avg" if within else "")
            pts = alt.Chart(d.assign(district=d.district.map(pretty))).mark_circle(size=45, opacity=.55, color=GREEN).encode(
                x=alt.X(f"{c}:Q", title=xlab, scale=alt.Scale(zero=False)), y=alt.Y("yield_t_ha:Q", title=ylab),
                tooltip=["district", alt.Tooltip("year:Q", format="d"), alt.Tooltip(f"{c}:Q", format=".1f"),
                         alt.Tooltip("yield_t_ha:Q", format=".2f")])
            fit = pts.transform_regression(c, "yield_t_ha").mark_line(color=AMBER, strokeWidth=3)
            with col:
                with st.container(key=f"card_sc_{c}"):
                    ui.card_title(info[c], f"r = {r:+.2f}")
                    ui.show(pts + fit, 260)
        if within:
            ui.callout("info", "Within-district view. ", "After removing district averages the relationships are much "
                       "weaker. Most of the apparent weather effect comes from differences between districts.")
        else:
            ui.callout("warn", "Be careful reading these correlations. ", "Wet, humid districts (e.g. in the Konkan) also "
                       "have higher yields for other reasons. Tick the box above to see year-to-year effects within districts.")

        ui.section("Weather over time", "Average across districts (NASA POWER, June-September).")
        yr = df.groupby("year")[["rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"]].mean().reset_index()
        xw = alt.X("year:Q", title="Agricultural year", axis=alt.Axis(format="d", tickMinStep=1),
                   scale=alt.Scale(zero=False, nice=False, padding=8))
        w1, w2 = st.columns(2, gap="medium")
        with w1:
            with st.container(key="card_w_rain"):
                ui.card_title("Rainfall, Jun-Sep (mm)", "mean of districts")
                ui.show(alt.Chart(yr).mark_bar(color=BLUE, opacity=.85).encode(
                    x=xw, y=alt.Y("rain_kharif_mm:Q", title="Mean rainfall (mm)"),
                    tooltip=[alt.Tooltip("year:Q", format="d"), alt.Tooltip("rain_kharif_mm:Q", format=".0f")]), 280)
        with w2:
            with st.container(key="card_w_temp"):
                ui.card_title("Temperature (°C) and humidity (%)", "mean of districts")
                tl = yr.melt("year", ["temp_kharif_c", "rh_kharif_pct"], "var", "value")
                tl["var"] = tl["var"].map({"temp_kharif_c": "Temperature (°C)", "rh_kharif_pct": "Humidity (%)"})
                base_ch = alt.Chart(tl).encode(x=xw, color=alt.Color("var:N", title="", scale=alt.Scale(
                    domain=["Temperature (°C)", "Humidity (%)"], range=[AMBER, GREEN])))
                ui.show(alt.layer(
                    base_ch.transform_filter(alt.datum["var"] == "Temperature (°C)").mark_line(point=True, strokeWidth=2.5)
                    .encode(y=alt.Y("value:Q", title="Temperature (°C)", scale=alt.Scale(zero=False))),
                    base_ch.transform_filter(alt.datum["var"] == "Humidity (%)").mark_line(point=True, strokeWidth=2.5)
                    .encode(y=alt.Y("value:Q", title="Humidity (%)", scale=alt.Scale(zero=False)))
                ).resolve_scale(y="independent"), 280)
        st.caption("Note: Akola/Amravati, Hingoli/Parbhani, Nagpur/Wardha and Bhandara/Gondia share one weather grid cell.")


# =========================================================================== PERFORMANCE
def best_style(t):
    css = pd.DataFrame("", index=t.index, columns=t.columns)
    for _, g in t.groupby("Split"):
        for c in ["MAE", "MSE", "RMSE"]:
            css.loc[g[c].idxmin(), c] = "background-color:#D8EEDC;font-weight:700"
        css.loc[g["R²"].idxmax(), "R²"] = "background-color:#D8EEDC;font-weight:700"
    return t.style.apply(lambda _: css, axis=None).format({"MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"})


def performance_overview(ann, base, lin):
    sp = split_info
    p1 = N_INPUT * 64 + 64
    p2 = 64 * 32 + 32
    p3 = 32 + 1
    o1, o2 = st.columns(2, gap="medium")
    with o1:
        with st.container(key="card_arch"):
            ui.card_title("ANN architecture", f"{p1 + p2 + p3:,} parameters")
            ui.html_table(["Layer", "Units", "Activation", "Parameters"],
                          [("Input (5 numeric + district one-hot)", N_INPUT, "–", "–"), ("Dense 1", 64, "ReLU", f"{p1:,}"),
                           ("Dense 2", 32, "ReLU", f"{p2:,}"), ("Output", 1, "Linear", f"{p3}")])
            st.caption(f"{meta['ann']['architecture']}. Saved model = best of 5 seeds by validation loss (seed "
                       f"{meta['ann']['selected_seed']}, best epoch {meta['ann']['best_epoch']} of {meta['ann']['epochs_run']}).")
    with o2:
        with st.container(key="card_split"):
            ui.card_title("Chronological train / validation / test split")
            ui.html_table(["Set", "Years", "Rows", "Used for"],
                          [("Training", sp["train"], sp["n_train"], "fit weights, scaler, encoder"),
                           ("Validation", sp["val"], sp["n_val"], "early stopping, seed choice"),
                           ("Test", sp["test"], sp["n_test"], "final metrics only")])
            st.caption("No random splitting: later years are never used to predict earlier ones. Scaler and encoder are fitted "
                       "on training rows only.")
    with st.container(key="card_metrics"):
        ui.card_title("RMSE, MAE and R² for the three main models")
        rows = []
        for name in ["ANN (all features)", "Linear Regression", "District-mean baseline"]:
            t, v = crow(name, "test"), crow(name, "val")
            rows.append((name, f"{t.MAE:.4f}", f"{t.RMSE:.4f}", f"{t.R2:.4f}", f"{v.MAE:.4f}", f"{v.RMSE:.4f}", f"{v.R2:.4f}"))
        ui.html_table(["Model", "Test MAE", "Test RMSE", "Test R²", "Val MAE", "Val RMSE", "Val R²"], rows)
        st.caption("Errors are in tonnes/hectare (lower is better); R² higher is better.")

    full, noweather = crow("ANN (all features)", "test"), crow("ANN (no weather)", "test")
    diff = full.RMSE_mean - noweather.RMSE_mean
    with st.container(key="card_observations"):
        ui.card_title("Evaluation observations")
        ui.bullets([
            f"On the test years the ANN's RMSE ({ann.RMSE:.4f}) is {'higher' if ann.RMSE > base.RMSE else 'lower'} than the "
            f"district-mean baseline ({base.RMSE:.4f}) and {'higher' if ann.RMSE > lin.RMSE else 'lower'} than Linear "
            f"Regression ({lin.RMSE:.4f}) t/ha, so the ANN {'did not outperform' if ann.RMSE >= min(base.RMSE, lin.RMSE) else 'outperformed'} "
            "the simpler models.",
            f"On the validation years the ANN was the best model ({crow('ANN (all features)', 'val').RMSE:.4f} vs "
            f"{crow('District-mean baseline', 'val').RMSE:.4f} baseline), but that advantage did not carry over to the test years.",
            f"Averaged over 5 seeds, the ANN's test RMSE is {full.RMSE_mean:.4f} ± {full.RMSE_std:.4f}; the saved model is one "
            "seed picked on validation data.",
            f"Weather inputs did not help: seed-averaged test RMSE is {full.RMSE_mean:.4f} with weather vs "
            f"{noweather.RMSE_mean:.4f} without ({'worse' if diff > 0 else 'better'} by {abs(diff):.3f} t/ha).",
            f"The test set has only {sp['n_test']} rows (2 years), so small differences between models are within noise.",
        ])


def page_performance():
    ann, base, lin = crow("ANN (all features)", "test"), crow("District-mean baseline", "test"), crow("Linear Regression", "test")
    under = ann.RMSE >= min(base.RMSE, lin.RMSE)
    ui.page_header("", "Model Performance", "How the ANN compares with two simple reference models on unseen years.",
                   "ANN below baselines on test" if under else "ANN best on test")
    sp = split_info

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ANN test RMSE", f"{ann.RMSE:.4f} t/ha")
    c2.metric("ANN test R²", f"{ann.R2:.4f}")
    c3.metric("Baseline test RMSE", f"{base.RMSE:.4f} t/ha")
    c4.metric("Linear Reg. test RMSE", f"{lin.RMSE:.4f} t/ha")

    va, vb, vl = crow("ANN (all features)", "val"), crow("District-mean baseline", "val"), crow("Linear Regression", "val")
    sd = comp[(comp.model == "ANN (all features)") & (comp.split == "test")].RMSE_std.iloc[0]
    if under:
        ui.callout("warn", "The ANN currently underperforms the simpler models on the test years. ",
                   f"Test RMSE: ANN {ann.RMSE:.4f} vs district-mean baseline {base.RMSE:.4f} and Linear Regression "
                   f"{lin.RMSE:.4f} t/ha (lower is better). The ANN was best on the validation years "
                   f"({va.RMSE:.4f} vs {vb.RMSE:.4f} / {vl.RMSE:.4f}), but that advantage did not carry over. With only "
                   f"{sp['n_test']} test rows and seed-to-seed spread of about ±{sd:.3f}, differences between models are small. "
                   f"Note that the saved ANN is one of five seeds; the average test RMSE over all five seeds is "
                   f"{ann.RMSE_mean:.4f}, worse than the saved one.")
    else:
        ui.callout("good", "The ANN had the lowest test error. ", f"ANN {ann.RMSE:.4f} vs baseline {base.RMSE:.4f} and Linear "
                   f"Regression {lin.RMSE:.4f} t/ha.")

    to, t1, t2, t3, t4 = st.tabs(["Overview", "Comparison", "Actual vs predicted", "Training curves", "Seed variability"])

    with to:
        performance_overview(ann, base, lin)

    with t1:
        ui.section("All models", "Green cells mark the best value in each column within a split.")
        tab = comp[["model", "split", "MAE", "MSE", "RMSE", "R2"]].copy()
        tab["model"] = pd.Categorical(tab.model, ORDER, ordered=True)
        tab = tab.sort_values(["split", "model"], ascending=[False, True]).reset_index(drop=True)
        tab["split"] = tab.split.map({"val": "Validation", "test": "Test"})
        tab["model"] = tab.model.astype(str)
        tab.columns = ["Model", "Split", "MAE", "MSE", "RMSE", "R²"]
        st.dataframe(best_style(tab), hide_index=True, width="stretch")

        ch1, ch2 = st.columns(2, gap="medium")
        long = comp.assign(split=comp.split.map({"val": "Validation", "test": "Test"}))
        for col, metric, title in [(ch1, "RMSE", "RMSE (t/ha, lower is better)"), (ch2, "R2", "R² (higher is better)")]:
            base_c = alt.Chart(long).encode(
                y=alt.Y("model:N", sort=ORDER, title="", axis=alt.Axis(labelLimit=220)),
                yOffset=alt.YOffset("split:N", sort=["Validation", "Test"]),
                color=alt.Color("split:N", title="", scale=alt.Scale(domain=["Validation", "Test"], range=[MINT, FOREST])))
            bars = base_c.mark_bar().encode(x=alt.X(f"{metric}:Q", title=title),
                                            tooltip=["model", "split", alt.Tooltip(f"{metric}:Q", format=".4f")])
            labels = alt.Chart(long).mark_text(align="left", dx=4, fontSize=10, color="#14281D").encode(
                y=alt.Y("model:N", sort=ORDER), yOffset=alt.YOffset("split:N", sort=["Validation", "Test"]),
                x=f"{metric}:Q", text=alt.Text(f"{metric}:Q", format=".3f"))
            with col:
                with st.container(key=f"card_cmp_{metric}"):
                    ui.show(bars + labels, 360)

    with t2:
        preds = get_predictions()
        ui.section("Actual vs predicted", "Points on the dashed line would be perfect predictions.")
        f1, f2 = st.columns([1, 1])
        split = f1.radio("Data subset", ["test", "val"], horizontal=True, format_func=SPLIT_NAME.get)
        model = f2.selectbox("Model", ORDER, index=ORDER.index("ANN (all features)"))
        p = preds[preds.split == split].copy()
        p["district"] = p["district"].map(pretty)
        p["pred"] = p[PRED_COL[model]]
        p["resid"] = p.actual_yield_t_ha - p.pred
        sel_row = crow(model, split)  # metrics from the saved evaluation file
        ui.chips([f"{model}", f"RMSE {sel_row.RMSE:.4f} t/ha", f"R² {sel_row.R2:.4f}", f"{len(p)} rows"])
        hi = float(max(p.actual_yield_t_ha.max(), p.pred.max())) * 1.05
        color = MODEL_COLORS[model]
        a1, a2 = st.columns(2, gap="medium")
        pts = alt.Chart(p).mark_circle(size=70, opacity=.75, color=color).encode(
            x=alt.X("actual_yield_t_ha:Q", title="Actual yield (t/ha)", scale=alt.Scale(domain=[0, hi])),
            y=alt.Y("pred:Q", title="Predicted yield (t/ha)", scale=alt.Scale(domain=[0, hi])),
            tooltip=["district", alt.Tooltip("year:Q", format="d"), alt.Tooltip("actual_yield_t_ha:Q", format=".2f", title="Actual"),
                     alt.Tooltip("pred:Q", format=".2f", title="Predicted")])
        diag = alt.Chart(pd.DataFrame({"x": [0, hi], "y": [0, hi]})).mark_line(color=SLATE, strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
        with a1:
            with st.container(key="card_avp"):
                ui.card_title("Actual vs predicted")
                ui.show(pts + diag, 340)
        res = alt.Chart(p).mark_circle(size=70, opacity=.75, color=color).encode(
            x=alt.X("pred:Q", title="Predicted yield (t/ha)"), y=alt.Y("resid:Q", title="Residual = actual − predicted (t/ha)"),
            tooltip=["district", alt.Tooltip("year:Q", format="d"), alt.Tooltip("resid:Q", format="+.2f", title="Residual")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=SLATE, strokeDash=[5, 4]).encode(y="y:Q")
        with a2:
            with st.container(key="card_resid"):
                ui.card_title("Residuals")
                ui.show(res + zero, 340)

    with t3:
        ui.section("Training vs validation loss", "Mean squared error per epoch for the saved ANN and the no-weather ANN.")
        m = meta["ann"]
        ui.chips([f"best epoch {m['best_epoch']}", f"{m['epochs_run']} epochs run", "early stopping (patience 30)",
                  "Adam · MSE · batch 32"])
        img = data.loss_curve_image()
        if img:
            st.image(img, width="stretch")
        st.caption("The validation loss flattens after about 14 epochs while training loss keeps falling: early stopping "
                   "keeps the weights from the best validation epoch. The model overfits quickly because the dataset is small.")

    with t4:
        seeds = get_seeds()
        ui.section("Does the ANN result depend on luck?", "Test RMSE for each of the 5 random seeds, against the reference models.")
        s = seeds[(seeds.split == "test") & seeds.model.isin(["ANN (all features)", "ANN (no weather)"])].copy()
        refs = pd.DataFrame({"model": ["District-mean baseline", "Linear Regression"], "RMSE": [base.RMSE, lin.RMSE]})
        dots = alt.Chart(s).mark_circle(size=140, opacity=.9).encode(
            x=alt.X("seed:O", title="Random seed", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("RMSE:Q", title="Test RMSE (t/ha)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="", scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))),
            xOffset="model:N", tooltip=["model", "seed", alt.Tooltip("RMSE:Q", format=".3f")])
        rules = alt.Chart(refs).mark_rule(strokeDash=[6, 4], strokeWidth=2.5).encode(
            y="RMSE:Q", color=alt.Color("model:N", title="", scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))),
            tooltip=["model", alt.Tooltip("RMSE:Q", format=".3f")])
        with st.container(key="card_seeds"):
            ui.show(dots + rules, 360)
            beat = {m: int((s[s.model == m].RMSE < base.RMSE).sum()) for m in ["ANN (all features)", "ANN (no weather)"]}
            n_runs = int(s.seed.nunique())
            st.caption(f"Dashed lines: district-mean baseline and Linear Regression. Dots: the ANN trained with each seed. "
                       f"Runs with a lower test RMSE than the district-mean baseline: full ANN {beat['ANN (all features)']}/{n_runs}, "
                       f"no-weather ANN {beat['ANN (no weather)']}/{n_runs}. The saved model is one seed chosen on validation data.")
        agg = (s.groupby("model")["RMSE"].agg(["mean", "std", "min", "max"]).round(3).reset_index()
               .rename(columns={"model": "Model", "mean": "Mean", "std": "Std", "min": "Best seed", "max": "Worst seed"}))
        st.dataframe(agg, hide_index=True, width="stretch")


{"Dashboard": lambda: page_dashboard(profile), "Predict Yield": lambda: page_predict(profile), "Data & Trends": page_trends,
 "Model Performance": page_performance}[page]()
ui.footer()
