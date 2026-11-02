"""Chart builders shared by the Dashboard and Data & Trends pages. All inputs are the real project dataframes."""
import altair as alt
import numpy as np
import pandas as pd

from src import data
from src.ui import AMBER, BLUE, FOREST, GREEN, MINT

PALETTE = [FOREST, AMBER, BLUE, "#8E4B9E", "#C0504D", "#00897B"]


def pretty(d: str) -> str:
    return d.title()


def _year_axis():
    return alt.X("year:Q", title="Agricultural year", axis=alt.Axis(format="d", tickMinStep=1),
                 scale=alt.Scale(zero=False, nice=False, padding=8))


def yield_trend(df: pd.DataFrame, picks=()):
    """Yearly mean yield across districts, middle-50% band, optional district overlays."""
    g = df.groupby("year")["yield_t_ha"]
    yearly = pd.DataFrame({"year": g.mean().index, "mean": g.mean().values,
                           "q25": g.quantile(.25).values, "q75": g.quantile(.75).values})
    x = _year_axis()
    band = alt.Chart(yearly).mark_area(color=MINT, opacity=.7).encode(x=x, y="q25:Q", y2="q75:Q")
    lines = pd.concat([
        yearly[["year", "mean"]].rename(columns={"mean": "yield_t_ha"}).assign(series="All districts (mean)"),
        df[df.district.isin(picks)][["year", "yield_t_ha", "district"]]
        .assign(series=lambda d: d.district.map(pretty)).drop(columns="district")])
    names = ["All districts (mean)"] + [pretty(p) for p in picks]
    ln = alt.Chart(lines).mark_line(point=True, strokeWidth=2.5).encode(
        x=x, y=alt.Y("yield_t_ha:Q", title="Yield (t/ha)"),
        color=alt.Color("series:N", title="", scale=alt.Scale(domain=names, range=PALETTE[:len(names)])),
        tooltip=["series", alt.Tooltip("year:Q", format="d"), alt.Tooltip("yield_t_ha:Q", format=".2f", title="Yield (t/ha)")])
    return band + ln


def district_bars(df: pd.DataFrame):
    """Vertical bars of mean yield per district, sorted high to low; amber = fewer than 15 records."""
    s = df.groupby("district")["yield_t_ha"].agg(["mean", "count"]).reset_index()
    s["district"] = s["district"].map(pretty)
    s["history"] = np.where(s["count"] >= 15, "15+ records", "fewer than 15 records")
    return alt.Chart(s).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("district:N", sort=alt.EncodingSortField("mean", order="descending"), title="",
                axis=alt.Axis(labelAngle=-55, labelLimit=120)),
        y=alt.Y("mean:Q", title="Mean yield (t/ha)"),
        color=alt.Color("history:N", title="", scale=alt.Scale(domain=["15+ records", "fewer than 15 records"],
                                                               range=[GREEN, AMBER])),
        tooltip=[alt.Tooltip("district:N", title="District"), alt.Tooltip("mean:Q", format=".2f", title="Mean yield (t/ha)"),
                 alt.Tooltip("count:Q", title="Records")])


def rain_vs_yield(df: pd.DataFrame):
    """Scatter of June-September rainfall against yield with a straight-line fit; returns (chart, Pearson r)."""
    pts = alt.Chart(df.assign(district=df.district.map(pretty))).mark_circle(size=42, opacity=.55, color=GREEN).encode(
        x=alt.X("rain_kharif_mm:Q", title="Rainfall, Jun-Sep (mm)", scale=alt.Scale(zero=False)),
        y=alt.Y("yield_t_ha:Q", title="Yield (t/ha)"),
        tooltip=["district", alt.Tooltip("year:Q", format="d"), alt.Tooltip("rain_kharif_mm:Q", format=".0f", title="Rainfall (mm)"),
                 alt.Tooltip("yield_t_ha:Q", format=".2f", title="Yield (t/ha)")])
    fit = pts.transform_regression("rain_kharif_mm", "yield_t_ha").mark_line(color=AMBER, strokeWidth=3)
    return pts + fit, float(df["rain_kharif_mm"].corr(df["yield_t_ha"]))


def records_per_year(df: pd.DataFrame):
    r = df.groupby("year").size().reset_index(name="records")
    return alt.Chart(r).mark_bar(color=GREEN, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=_year_axis(), y=alt.Y("records:Q", title="Records"),
        tooltip=[alt.Tooltip("year:Q", format="d"), alt.Tooltip("records:Q", title="Records")])


def records_per_district(df: pd.DataFrame):
    r = df.groupby("district").size().reset_index(name="records")
    r["district"] = r["district"].map(pretty)
    return alt.Chart(r).mark_bar(color=GREEN, cornerRadiusEnd=3).encode(
        y=alt.Y("district:N", sort=alt.EncodingSortField("records", order="descending"), title=""),
        x=alt.X("records:Q", title="Records (years with data)"),
        tooltip=[alt.Tooltip("district:N", title="District"), alt.Tooltip("records:Q", title="Records")])


def yield_trend_smooth(df: pd.DataFrame):
    """Smooth mean-yield line with a soft green area underneath (dashboard version)."""
    g = df.groupby("year")["yield_t_ha"]
    yearly = pd.DataFrame({"year": g.mean().index, "mean": g.mean().values, "records": g.size().values})
    x = _year_axis()
    grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                        stops=[alt.GradientStop(color="rgba(46,139,87,0.02)", offset=0),
                               alt.GradientStop(color="rgba(46,139,87,0.32)", offset=1)])
    area = alt.Chart(yearly).mark_area(interpolate="monotone", color=grad).encode(x=x, y=alt.Y("mean:Q", title="Mean yield (t/ha)"))
    line = alt.Chart(yearly).mark_line(interpolate="monotone", color=GREEN, strokeWidth=3,
                                       point=alt.OverlayMarkDef(color=GREEN, fill="white", size=70, strokeWidth=2)).encode(
        x=x, y="mean:Q", tooltip=[alt.Tooltip("year:Q", format="d", title="Year"),
                                  alt.Tooltip("mean:Q", format=".2f", title="Mean yield (t/ha)"),
                                  alt.Tooltip("records:Q", title="Districts with data")])
    return area + line


MAP_COLORS = ["#F1F8F2", "#CFE8D5", "#9FD1AB", "#6BB783", "#3E9A63"]


def district_map(df: pd.DataFrame, pick: str = "All districts", labels: bool = True, legend: bool = True):
    """District choropleth of mean yield. Returns (chart, districts_without_boundary) or (None, []) if no boundary file."""
    gj = data.load_boundaries()
    if gj is None:
        return None, []
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
    vals = stats["mean"]
    stops = list(np.linspace(float(vals.min()), float(vals.max()), len(MAP_COLORS)))
    color = alt.Color("properties.mean_yield:Q", scale=alt.Scale(domain=stops, range=MAP_COLORS),
                      legend=alt.Legend(title="Mean yield (t/ha)", orient="right", direction="vertical", gradientLength=200,
                                        format=".1f", titleLimit=200) if legend else None)
    shapes = alt.Chart(alt.Data(values=feats)).mark_geoshape(stroke="white", strokeWidth=1).encode(
        color=color, tooltip=[alt.Tooltip("properties.district_name:N", title="District"),
                              alt.Tooltip("properties.mean_yield:Q", title="Mean yield (t/ha)", format=".2f"),
                              alt.Tooltip("properties.records:Q", title="Records"),
                              alt.Tooltip("properties.years:N", title="Years")])
    layers = [shapes]
    pts = data.load_district_points() if labels else None
    if pts is not None:
        pts = pts[pts.district.isin(drawn)].assign(label=lambda d: d.district.map(pretty))
        layers.append(alt.Chart(pts).mark_text(fontSize=8.5, color="#14281D", fontWeight=600).encode(
            longitude="lon:Q", latitude="lat:Q", text="label:N"))
    if pick != "All districts":
        sel = [f for f in feats if f["properties"]["district"] == pick]
        layers.append(alt.Chart(alt.Data(values=sel)).mark_geoshape(fill=None, stroke=FOREST, strokeWidth=3.5))
    return alt.layer(*layers).project(type="mercator"), no_shape
