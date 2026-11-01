"""EDA figures + summary for the merged dataset. Run after preprocess.py:  python eda.py
Descriptive only: no scaling, splitting or modelling happens here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config as C

T = C.TARGET
NUM = ["year", "area_ha", "rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct", T]


def save(name):
    plt.tight_layout()
    plt.savefig(C.EDA_DIR / name, dpi=130)
    plt.close()
    print("saved", name)


def main():
    C.EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.MERGED_CSV)

    sns.histplot(df[T], bins=40, kde=True)
    plt.xlabel("Rice yield (t/ha)"); plt.title("Yield distribution (Kharif rice, Maharashtra)")
    save("01_yield_distribution.png")

    g = df.groupby("year")[T]
    plt.plot(g.median().index, g.median(), marker="o", label="median across districts")
    plt.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), alpha=0.25, label="25-75%")
    plt.xlabel("Year (Kharif)"); plt.ylabel("Yield (t/ha)"); plt.legend(); plt.title("Year-wise yield")
    save("02_yield_trend.png")

    order = df.groupby("district")[T].median().sort_values().index
    plt.figure(figsize=(8, 0.3 * len(order) + 1.5))
    sns.boxplot(data=df, y="district", x=T, order=order)
    plt.xlabel("Yield (t/ha)"); plt.title("District-wise yield")
    save("03_district_yield_boxplot.png")

    counts = df.groupby("district").size().sort_values()
    plt.figure(figsize=(7, 0.3 * len(counts) + 1.5))
    counts.plot.barh(); plt.xlabel("Observations"); plt.title("Records per district")
    save("04_district_record_counts.png")

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, col, lab in zip(ax, ["area_ha", "rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"],
                           ["Area (ha)", "Jun-Sep rainfall (mm)", "Jun-Sep mean temp (C)", "Jun-Sep mean RH (%)"]):
        sns.histplot(df[col], bins=30, ax=a); a.set_xlabel(lab)
    save("05_feature_distributions.png")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col, lab in zip(ax, ["rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"],
                           ["Jun-Sep rainfall (mm)", "Jun-Sep mean temp (C)", "Jun-Sep mean RH (%)"]):
        sns.scatterplot(data=df, x=col, y=T, hue="year", palette="viridis", s=16, ax=a, legend=(col == "rh_kharif_pct"))
        a.set_xlabel(lab); a.set_ylabel("Yield (t/ha)")
    save("06_weather_vs_yield.png")

    corr = df[NUM].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Pearson correlation")
    save("07_correlation_heatmap.png")

    # within-district correlation (removes between-district level differences)
    dm = df.copy()
    for c in NUM:
        dm[c] = dm[c] - dm.groupby("district")[c].transform("mean")
    wcorr = dm[NUM].corr()[T].drop(T)

    out = ["# EDA summary\n", f"Rows: {len(df)}, districts: {df.district.nunique()}, years {df.year.min()}-{df.year.max()}\n",
           "## Descriptive statistics\n", df[NUM].describe().T.round(3).to_markdown() if hasattr(pd.DataFrame, "to_markdown") and _has_tabulate()
           else "```\n" + df[NUM].describe().T.round(3).to_string() + "\n```",
           "\n## Correlation with yield (all rows)\n```\n" + corr[T].drop(T).round(3).to_string() + "\n```",
           "\n## Correlation with yield after removing district means\n```\n" + wcorr.round(3).to_string() + "\n```\n"]
    (C.EDA_DIR / "eda_summary.md").write_text("\n".join(out))
    print(corr[T].drop(T).round(3).to_string())


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
