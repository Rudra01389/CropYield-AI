"""CropYield AI - Streamlit dashboard.  Run:  streamlit run app.py"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import data, interpret, report, ui
from src.predictor import LABELS, ModelNotAvailable, Predictor, validate_inputs
from src.ui import AMBER, BLUE, FOREST, GREEN, LEAF, MINT, MODEL_COLORS, SLATE

st.set_page_config(page_title="CropYield AI", page_icon="🌾", layout="wide")
ui.inject_css()

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


@st.cache_data(show_spinner="Computing feature importance...")
def get_importance():
    """Permutation importance of the saved ANN on the held-out test years (cached for the whole session)."""
    res = interpret.permutation_importance(get_predictor(), interpret.evaluation_rows(df, meta))
    return res.copy(), dict(res.attrs)


split_info = meta["split"]
base_means = meta["district_mean_baseline"]["means"]


def crow(model, split):
    return comp[(comp.model == model) & (comp.split == split)].iloc[0]


# --------------------------------------------------------------------------- sidebar
NAV = {"Predict yield": ":material/agriculture: Predict yield", "Data & trends": ":material/insights: Data & trends",
       "Model performance": ":material/analytics: Model performance",
       "About & limitations": ":material/info: About & limitations"}
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🌾</div><div><div class="name">CropYield AI</div>'
                '<div class="tag">Rice yield · Maharashtra</div></div></div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    page = st.radio("Navigate", list(NAV), format_func=NAV.get, label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    ann_t = crow("ANN (all features)", "test")
    st.markdown(
        f'<div class="snap"><div class="lbl">Model</div><b>ANN</b> 39 → 64 → 32 → 1<br>'
        f'<div class="lbl" style="margin-top:.5rem">Test RMSE (2018-19)</div><b>{ann_t.RMSE:.4f} t/ha</b>'
        f'<div class="lbl" style="margin-top:.5rem">Data</div>{len(df)} records · {df.district.nunique()} districts · '
        f'{df.year.min()}-{df.year.max()}</div>', unsafe_allow_html=True)
    st.caption("Academic prototype. Not for real farming, insurance or policy decisions.")


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
            st.markdown("**What each input is**")
            ui.definition_list(roles)
            st.caption("Correlations are over all 642 records and mostly reflect differences between districts.")
    with right:
        with st.container(key="card_importance"):
            st.markdown("**How much the ANN relies on each input** (permutation importance)")
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
        st.markdown("**Your inputs vs this district's typical values**")
        st.dataframe(pd.DataFrame(rows, columns=["Input", "Your value", f"{pretty(values['district'])} median", "Difference"]),
                     hide_index=True, width="stretch")
        st.caption("Large differences mean the estimate is based on unusual conditions for this district.")


def page_predict():
    ui.page_header("🌱", "Predict Kharif rice yield", "Enter a district and the season's conditions to get an ANN estimate.",
                   "ANN · 39→64→32→1")
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(key="card_inputs"):
            ui.section("1 · Location", "Defaults below are this district's historical medians.")
            district = st.selectbox("District", districts, format_func=pretty,
                                    help="Maharashtra districts with Kharif rice records (1997-2019).")
            sub = df[df.district == district]
            ui.chips([f"{len(sub)} records", f"{int(sub.year.min())}-{int(sub.year.max())}",
                      f"typical area {sub.area_ha.median():,.0f} ha"], muted=True)
            ui.section("2 · Season conditions", "Rainfall, temperature and humidity are June–September values.")
            with st.form("predict_form", border=False):
                r1a, r1b = st.columns(2)
                year = r1a.number_input(LABELS["year"], 1997, 2035, int(df.year.max()), 1, key=f"year_{district}",
                                        help="2019 means crop year 2019-20 (Kharif sown June-July 2019).")
                area = r1b.number_input(LABELS["area_ha"], 0.0, 1_000_000.0, float(sub.area_ha.median()), 100.0,
                                        key=f"area_{district}", help="Area under Kharif rice, in hectares.")
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
                        st.metric(f"{pretty(district)} historical average", f"{base:.2f} t/ha",
                                  delta=f"{shown - base:+.2f} t/ha vs average",
                                  help="Mean yield in the training years 1997-2015.")
                        st.caption("Statistical estimate from a small academic model. See *Model performance* for its "
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


MAP_COLORS = ["#F1F8F2", "#CFE8D5", "#9FD1AB", "#6BB783", "#3E9A63"]


def district_map_tab():
    ui.section("District-wise average rice yield", "Mean reported Kharif rice yield per district, 1997-2019. Hover a district "
               "for its name and value.")
    gj = data.load_boundaries()
    if gj is None:
        ui.callout("warn", "Boundary data not found. ", "Run `python fetch_boundaries.py` to download the district outlines "
                   "(OpenStreetMap). The charts in the other tabs still work.")
        return
    stats = df.groupby("district")["yield_t_ha"].agg(["mean", "min", "max", "count"])
    yrs = df.groupby("district")["year"].agg(["min", "max"])
    feats, drawn = [], set()
    for f in gj["features"]:
        d = f["properties"]["district"]
        has = d in stats.index
        drawn.add(d)
        feats.append({"type": "Feature", "geometry": f["geometry"], "properties": {
            "district": d, "district_name": pretty(d),
            "mean_yield": round(float(stats.loc[d, "mean"]), 3) if has else None,
            "records": int(stats.loc[d, "count"]) if has else 0,
            "years": f"{int(yrs.loc[d, 'min'])}-{int(yrs.loc[d, 'max'])}" if has else "no records"}})
    no_shape = sorted(set(stats.index) - drawn)

    pick = st.selectbox("Highlight a district (optional)", ["All districts"] + districts,
                        format_func=lambda d: d if d == "All districts" else pretty(d), key="map_pick")
    vals = stats["mean"]
    stops = list(np.linspace(float(vals.min()), float(vals.max()), len(MAP_COLORS)))
    color = alt.Color("properties.mean_yield:Q", scale=alt.Scale(domain=stops, range=MAP_COLORS),
                      legend=alt.Legend(title="Mean yield (t/ha)", orient="right", direction="vertical", gradientLength=230,
                                        format=".1f", titleLimit=200))
    shapes = alt.Chart(alt.Data(values=feats)).mark_geoshape(stroke="white", strokeWidth=1).encode(
        color=color, tooltip=[alt.Tooltip("properties.district_name:N", title="District"),
                              alt.Tooltip("properties.mean_yield:Q", title="Mean yield (t/ha)", format=".2f"),
                              alt.Tooltip("properties.records:Q", title="Records"),
                              alt.Tooltip("properties.years:N", title="Years")])
    layers = [shapes]
    pts = data.load_district_points()
    if pts is not None:
        pts = pts[pts.district.isin(drawn)].assign(label=lambda d: d.district.map(pretty))
        layers.append(alt.Chart(pts).mark_text(fontSize=8.5, color="#14281D", fontWeight=600).encode(
            longitude="lon:Q", latitude="lat:Q", text="label:N"))
    if pick != "All districts":
        sel = [f for f in feats if f["properties"]["district"] == pick]
        layers.append(alt.Chart(alt.Data(values=sel)).mark_geoshape(fill=None, stroke=FOREST, strokeWidth=3.5))
    with st.container(key="card_map"):
        ui.show(alt.layer(*layers).project(type="mercator"), 560)
        st.caption("Darker green = higher average yield. Outlines: current OpenStreetMap district boundaries (ODbL), simplified.")

    if pick != "All districts":
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


# ============================================================================== TRENDS
def page_trends():
    ui.page_header("📈", "Data & trends", "Explore the 642 Kharif rice records used to train and test the model.",
                   f"{df.year.min()}–{df.year.max()}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df)}")
    c2.metric("Districts", f"{df.district.nunique()}")
    c3.metric("Years", f"{df.year.min()}-{df.year.max()}")
    c4.metric("Mean yield", f"{df.yield_t_ha.mean():.2f} t/ha")

    t1, t2, tm, t3, t4 = st.tabs(["Yield trends", "District comparison", "Yield map", "Weather vs yield",
                                  "Weather over time"])

    with t1:
        ui.section("Yield over time", "Yearly mean across districts, with the middle 50% of districts shaded.")
        pick = st.multiselect("Overlay districts (optional, up to 5)", districts, format_func=pretty, max_selections=5)
        g = df.groupby("year")["yield_t_ha"]
        yearly = pd.DataFrame({"year": g.mean().index, "mean": g.mean().values, "q25": g.quantile(.25).values,
                               "q75": g.quantile(.75).values})
        x = alt.X("year:Q", title="Agricultural year", axis=alt.Axis(format="d", tickMinStep=1),
                  scale=alt.Scale(zero=False, nice=False, padding=8))
        band = alt.Chart(yearly).mark_area(color=MINT, opacity=.65).encode(x=x, y="q25:Q", y2="q75:Q")
        lines = pd.concat([yearly[["year", "mean"]].rename(columns={"mean": "yield_t_ha"}).assign(series="All districts (mean)"),
                           df[df.district.isin(pick)][["year", "yield_t_ha", "district"]]
                           .assign(series=lambda d: d.district.map(pretty)).drop(columns="district")])
        palette = [FOREST, AMBER, BLUE, "#8E4B9E", "#C0504D", "#00897B"]
        names = ["All districts (mean)"] + [pretty(p) for p in pick]
        ln = alt.Chart(lines).mark_line(point=True, strokeWidth=2.5).encode(
            x=x, y=alt.Y("yield_t_ha:Q", title="Yield (t/ha)"),
            color=alt.Color("series:N", title="", scale=alt.Scale(domain=names, range=palette[:len(names)])),
            tooltip=["series", alt.Tooltip("year:Q", format="d"), alt.Tooltip("yield_t_ha:Q", format=".2f", title="Yield")])
        with st.container(key="card_trend"):
            ui.show(band + ln, 360)
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

    with t3:
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
        ccols = st.columns(3)
        for col, c in zip(ccols, cols):
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
                    st.markdown(f"**{info[c]}**  \nPearson r = **{r:+.2f}**")
                    ui.show(pts + fit, 260)
        if within:
            ui.callout("info", "Within-district view. ", "After removing district averages the relationships are much "
                       "weaker. Most of the apparent weather effect comes from differences between districts.")
        else:
            ui.callout("warn", "Be careful reading these correlations. ", "Wet, humid districts (e.g. in the Konkan) also "
                       "have higher yields for other reasons. Tick the box above to see year-to-year effects within districts.")

    with t4:
        ui.section("Weather over time", "Average across districts (NASA POWER, June-September).")
        yr = df.groupby("year")[["rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"]].mean().reset_index()
        xw = alt.X("year:Q", title="Agricultural year", axis=alt.Axis(format="d", tickMinStep=1),
                   scale=alt.Scale(zero=False, nice=False, padding=8))
        w1, w2 = st.columns(2)
        with w1:
            with st.container(key="card_w_rain"):
                st.markdown("**Rainfall, Jun-Sep (mm)**")
                ui.show(alt.Chart(yr).mark_bar(color=BLUE, opacity=.85).encode(
                    x=xw, y=alt.Y("rain_kharif_mm:Q", title="Mean rainfall (mm)"),
                    tooltip=[alt.Tooltip("year:Q", format="d"), alt.Tooltip("rain_kharif_mm:Q", format=".0f")]), 280)
        with w2:
            with st.container(key="card_w_temp"):
                st.markdown("**Temperature (°C) and humidity (%)**")
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


def page_performance():
    ann, base, lin = crow("ANN (all features)", "test"), crow("District-mean baseline", "test"), crow("Linear Regression", "test")
    under = ann.RMSE >= min(base.RMSE, lin.RMSE)
    ui.page_header("📊", "Model performance", "How the ANN compares with two simple reference models on unseen years.",
                   "ANN below baselines on test" if under else "ANN best on test")
    sp = split_info
    st.caption(f"Chronological split · train {sp['train']} ({sp['n_train']} rows) · validation {sp['val']} ({sp['n_val']}) · "
               f"test {sp['test']} ({sp['n_test']}). The saved ANN is the best of 5 random seeds by validation loss.")

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

    t1, t2, t3, t4 = st.tabs(["Comparison", "Actual vs predicted", "Training curves", "Seed variability"])

    with t1:
        ui.section("All models", "Green cells mark the best value in each column within a split.")
        tab = comp[["model", "split", "MAE", "MSE", "RMSE", "R2"]].copy()
        tab["model"] = pd.Categorical(tab.model, ORDER, ordered=True)
        tab = tab.sort_values(["split", "model"], ascending=[False, True]).reset_index(drop=True)
        tab["split"] = tab.split.map({"val": "Validation", "test": "Test"})
        tab["model"] = tab.model.astype(str)
        tab.columns = ["Model", "Split", "MAE", "MSE", "RMSE", "R²"]
        st.dataframe(best_style(tab), hide_index=True, width="stretch")

        ch1, ch2 = st.columns(2)
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
        a1, a2 = st.columns(2)
        pts = alt.Chart(p).mark_circle(size=70, opacity=.75, color=color).encode(
            x=alt.X("actual_yield_t_ha:Q", title="Actual yield (t/ha)", scale=alt.Scale(domain=[0, hi])),
            y=alt.Y("pred:Q", title="Predicted yield (t/ha)", scale=alt.Scale(domain=[0, hi])),
            tooltip=["district", alt.Tooltip("year:Q", format="d"), alt.Tooltip("actual_yield_t_ha:Q", format=".2f", title="Actual"),
                     alt.Tooltip("pred:Q", format=".2f", title="Predicted")])
        diag = alt.Chart(pd.DataFrame({"x": [0, hi], "y": [0, hi]})).mark_line(color=SLATE, strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
        with a1:
            with st.container(key="card_avp"):
                st.markdown("**Actual vs predicted**")
                ui.show(pts + diag, 340)
        res = alt.Chart(p).mark_circle(size=70, opacity=.75, color=color).encode(
            x=alt.X("pred:Q", title="Predicted yield (t/ha)"), y=alt.Y("resid:Q", title="Residual = actual − predicted (t/ha)"),
            tooltip=["district", alt.Tooltip("year:Q", format="d"), alt.Tooltip("resid:Q", format="+.2f", title="Residual")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=SLATE, strokeDash=[5, 4]).encode(y="y:Q")
        with a2:
            with st.container(key="card_resid"):
                st.markdown("**Residuals**")
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
        refs = pd.DataFrame({"model": ["District-mean baseline", "Linear Regression"],
                             "RMSE": [base.RMSE, lin.RMSE]})
        dots = alt.Chart(s).mark_circle(size=140, opacity=.9).encode(
            x=alt.X("seed:O", title="Random seed", axis=alt.Axis(labelAngle=0)), y=alt.Y("RMSE:Q", title="Test RMSE (t/ha)", scale=alt.Scale(zero=False)),
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


# ================================================================================ ABOUT
def page_about():
    ui.page_header("🌾", "About CropYield AI", "Project overview, data, model, technology and limitations.", "Academic prototype")
    t1, t2, t3, t4, t5 = st.tabs(["Overview", "Data", "Model & workflow", "Technology", "Limitations"])

    with t1:
        ui.section("What this project does")
        st.write("CropYield AI estimates **Kharif rice yield (tonnes/hectare)** for districts of Maharashtra from the "
                 "agricultural year, cultivated area and June–September rainfall, temperature and humidity, using an "
                 "Artificial Neural Network. It is a Soft Computing mini project.")
        ui.mini_cards([("🎯 Objective", "Predict district rice yield and test whether an ANN beats simple baselines."),
                       ("🧪 Method", "Chronological split, leakage-free preprocessing, district-mean baseline, Linear "
                                     "Regression and ANN with early stopping."),
                       ("📌 Outcome", "Working pipeline and dashboard. The ANN did not beat the baselines on the test years.")], 3)
        ui.section("Objectives")
        ui.steps(["Build a clean district-year dataset of Kharif rice yield for Maharashtra.",
                  "Add June–September weather for every district from a public source.",
                  "Train an ANN and evaluate it on unseen future years.",
                  "Compare against a district-mean baseline and Linear Regression; test whether weather helps.",
                  "Provide a simple web interface for prediction and exploration."])

    with t2:
        ui.section("Datasets and sources")
        ui.mini_cards([("🌾 Crop data", "District-level Kharif rice area, production and yield for Maharashtra, 1997-98 to "
                                        "2019-20 (India Agriculture Crop Production dataset supplied by the team). "
                                        "642 usable records after cleaning."),
                       ("🛰️ Weather", "NASA POWER daily MERRA-2 reanalysis: precipitation, 2 m temperature and relative "
                                      "humidity, aggregated to June–September per district-year."),
                       ("📍 Locations", "One representative point per district: the centre of the district boundary "
                                        "from OpenStreetMap Nominatim.")], 3)
        ui.section("Model inputs", "Observed range in the dataset.")
        tbl = pd.DataFrame({"Input": ["District", LABELS["year"], LABELS["area_ha"], LABELS["rain_kharif_mm"],
                                      LABELS["temp_kharif_c"], LABELS["rh_kharif_pct"]],
                            "Observed range": ["34 districts", f"{df.year.min()}-{df.year.max()}",
                                               f"{df.area_ha.min():,.0f} - {df.area_ha.max():,.0f}",
                                               f"{df.rain_kharif_mm.min():,.0f} - {df.rain_kharif_mm.max():,.0f}",
                                               f"{df.temp_kharif_c.min():.1f} - {df.temp_kharif_c.max():.1f}",
                                               f"{df.rh_kharif_pct.min():.1f} - {df.rh_kharif_pct.max():.1f}"]})
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.caption("Target: rice yield in t/ha. Production is deliberately not an input: it would leak the answer.")

    with t3:
        ui.section("ANN architecture")
        ui.chips([meta["ann"]["architecture"].split(";")[0], "Adam · MSE · batch 32", "4,673 parameters"])
        img = data.diagram_image("03_ann_architecture.png")
        if img:
            st.image(img, width="stretch")
        ui.section("Workflow")
        img = data.diagram_image("02_workflow.png")
        if img:
            st.image(img, width="stretch")
        st.caption(f"Preprocessing: numeric inputs standardised, district one-hot encoded, both fitted on training years only. "
                   f"Best epoch {meta['ann']['best_epoch']} of {meta['ann']['epochs_run']} run (early stopping).")
        with st.expander("System architecture diagram"):
            img = data.diagram_image("01_system_architecture.png")
            if img:
                st.image(img, width="stretch")

    with t4:
        ui.section("Technology stack")
        v = meta["versions"]
        ui.mini_cards([("🐍 Python", "pandas · NumPy for data handling"), ("🧠 TensorFlow / Keras", f"ANN model (v{v['tensorflow']})"),
                       ("📐 scikit-learn", f"Scaling, encoding, Linear Regression, metrics (v{v['sklearn']})"),
                       ("🛰️ NASA POWER · OSM", "Weather API and district coordinates (via requests)"),
                       ("📊 Altair · Matplotlib", "Dashboard charts and report figures"),
                       ("🖥️ Streamlit", "This web application")], 3)

    with t5:
        ui.section("Limitations")
        ui.mini_cards([
            ("⚠️ Small dataset", "Only 642 observations; test results rest on two years (51 rows)."),
            ("⚠️ Model quality", "The ANN did not beat simple baselines on the test set (see Model performance)."),
            ("🛰️ Shared weather", "Four district pairs share one NASA grid cell: Hingoli/Parbhani, Akola/Amravati, "
                                  "Nagpur/Wardha, Bhandara/Gondia."),
            ("📍 One point per district", "Each district's weather is a single geographic point, not a district average."),
            ("🔗 Not causal", "Correlations in historical data do not establish causation."),
            ("🕒 Not a forecast", "Inputs cover June–September and are known only after the season."),
            ("🌱 Missing drivers", "No soil, irrigation or fertiliser information is used."),
            ("📉 Extrapolation", "Years or weather outside the training range give less reliable estimates.")], 2)
        st.markdown("&nbsp;", unsafe_allow_html=True)
        ui.callout("info", "Disclaimer. ", "Academic prototype for a Soft Computing mini project. Do not use it for farming, "
                   "insurance, pricing or policy decisions.")


{"Predict yield": page_predict, "Data & trends": page_trends, "Model performance": page_performance,
 "About & limitations": page_about}[page]()
ui.footer()
