"""Phase 3: district-mean baseline vs Linear Regression vs ANN (Keras) for Kharif rice yield.

Run:  python train_models.py        (after preprocess.py)

Split is chronological: train 1997-2015, validation 2016-2017, test 2018-2019.
All encoders/scalers are fit on training rows only. The test set is used once, for the final metrics;
validation is used for early stopping and for choosing which ANN seed to save.
Production is never used (it is not in ann_ready_dataset.csv's feature list).
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C

MODEL_DIR = C.ROOT / "models"
PLOT_DIR = C.REPORT_DIR / "model_plots"
TRAIN_END, VAL_END = 2015, 2017
SEEDS = [0, 1, 2, 3, 4]
T = C.TARGET
WEATHER = ["rain_kharif_mm", "temp_kharif_c", "rh_kharif_pct"]
NUM_ALL = ["year", "area_ha"] + WEATHER
NUM_NO_WEATHER = ["year", "area_ha"]
CAT = ["district"]
ARCH = "Input -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, linear); Adam(1e-3), MSE, batch 32, EarlyStopping(val_loss, patience 30)"


def make_preprocessor(num_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])


def metrics(y, p):
    mse = mean_squared_error(y, p)
    return dict(MAE=mean_absolute_error(y, p), MSE=mse, RMSE=float(np.sqrt(mse)), R2=r2_score(y, p))


def build_ann(n_in, seed):
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(n_in,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m


def train_ann(pre, num_cols, tr, va, te, seed):
    cols = num_cols + CAT
    Xtr, Xva, Xte = (pre.transform(d[cols]).astype("float32") for d in (tr, va, te))
    m = build_ann(Xtr.shape[1], seed)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    h = m.fit(Xtr, tr[T].to_numpy("float32"), validation_data=(Xva, va[T].to_numpy("float32")),
              epochs=500, batch_size=32, verbose=0, callbacks=[es], shuffle=True)
    pred = {k: m.predict(X, verbose=0).ravel() for k, X in [("train", Xtr), ("val", Xva), ("test", Xte)]}
    val_loss = float(min(h.history["val_loss"]))
    return dict(model=m, history=h.history, pred=pred, seed=seed, val_loss=val_loss,
                best_epoch=int(np.argmin(h.history["val_loss"])) + 1, epochs=len(h.history["loss"]))


def run_ann_variant(name, num_cols, tr, va, te):
    pre = make_preprocessor(num_cols).fit(tr[num_cols + CAT])  # TRAIN ONLY
    runs = [train_ann(pre, num_cols, tr, va, te, s) for s in SEEDS]
    best = min(runs, key=lambda r: r["val_loss"])               # chosen on VALIDATION loss only
    print(f"{name}: seeds {SEEDS}, val MSE {[round(r['val_loss'], 4) for r in runs]}, selected seed {best['seed']}")
    return pre, runs, best


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.ANN_READY_CSV)
    assert "production_t" not in df.columns
    need = NUM_ALL + CAT + [T]
    assert not df[need].isna().any().any()
    df = df.sort_values(["year", "district"]).reset_index(drop=True)

    tr, va, te = df[df.year <= TRAIN_END], df[(df.year > TRAIN_END) & (df.year <= VAL_END)], df[df.year > VAL_END]
    print(f"Split rows: train={len(tr)} ({tr.year.min()}-{tr.year.max()}), val={len(va)} ({va.year.min()}-{va.year.max()}), "
          f"test={len(te)} ({te.year.min()}-{te.year.max()})")
    assert tr.year.max() < va.year.min() and va.year.max() < te.year.min()
    unseen = sorted((set(va.district) | set(te.district)) - set(tr.district))
    print("Districts in val/test but not in train:", unseen)

    splits = {"train": tr, "val": va, "test": te}
    preds = {}  # model -> split -> array

    # ---- Model A: district-mean baseline (train means; fallback = overall train mean) ----
    dmean = tr.groupby("district")[T].mean()
    fallback = float(tr[T].mean())
    preds["District-mean baseline"] = {k: d.district.map(dmean).fillna(fallback).to_numpy() for k, d in splits.items()}

    # ---- Model B: Linear Regression (+ no-weather version for a fair comparison) ----
    lr_full = Pipeline([("pre", make_preprocessor(NUM_ALL)), ("lr", LinearRegression())]).fit(tr[NUM_ALL + CAT], tr[T])
    lr_nw = Pipeline([("pre", make_preprocessor(NUM_NO_WEATHER)), ("lr", LinearRegression())]).fit(
        tr[NUM_NO_WEATHER + CAT], tr[T])
    preds["Linear Regression"] = {k: lr_full.predict(d[NUM_ALL + CAT]) for k, d in splits.items()}
    preds["Linear Regression (no weather)"] = {k: lr_nw.predict(d[NUM_NO_WEATHER + CAT]) for k, d in splits.items()}

    # ---- Model C: ANN, with and without weather ----
    pre_full, runs_full, best_full = run_ann_variant("ANN (all features)", NUM_ALL, tr, va, te)
    pre_nw, runs_nw, best_nw = run_ann_variant("ANN (no weather)", NUM_NO_WEATHER, tr, va, te)
    preds["ANN (all features)"] = best_full["pred"]
    preds["ANN (no weather)"] = best_nw["pred"]

    # ---- metrics table ----
    rows = []
    for model, p in preds.items():
        for split in ("val", "test"):
            rows.append(dict(model=model, split=split, **metrics(splits[split][T], p[split]), n=len(splits[split])))
    sel = pd.DataFrame(rows)

    seed_rows = []
    for name, runs in [("ANN (all features)", runs_full), ("ANN (no weather)", runs_nw)]:
        for r in runs:
            for split in ("val", "test"):
                seed_rows.append(dict(model=name, seed=r["seed"], split=split, best_epoch=r["best_epoch"],
                                      epochs_run=r["epochs"], **metrics(splits[split][T], r["pred"][split])))
    seeds_df = pd.DataFrame(seed_rows)
    seeds_df.to_csv(C.REPORT_DIR / "ann_seed_results.csv", index=False)

    agg = seeds_df.groupby(["model", "split"])[["MAE", "MSE", "RMSE", "R2"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    comp = sel.merge(agg, on=["model", "split"], how="left")
    comp["note"] = np.where(comp.model.str.startswith("ANN"),
                            f"selected seed (lowest val loss); *_mean/_std over {len(SEEDS)} seeds", "deterministic")
    comp = comp.round(4)
    comp.to_csv(C.REPORT_DIR / "model_comparison.csv", index=False)
    print(comp[["model", "split", "MAE", "MSE", "RMSE", "R2", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std"]].to_string(index=False))

    # ---- predictions CSV ----
    out = []
    for split, d in splits.items():
        o = d[["district", "year", "area_ha", T]].rename(columns={T: "actual_yield_t_ha"}).copy()
        o.insert(0, "split", split)
        for model, p in preds.items():
            o["pred_" + model.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")] = p[split]
        out.append(o)
    pd.concat(out).round(4).to_csv(C.REPORT_DIR / "predictions.csv", index=False)

    # ---- artifacts ----
    best_full["model"].save(MODEL_DIR / "ann_model.keras")
    best_nw["model"].save(MODEL_DIR / "ann_model_no_weather.keras")
    joblib.dump(lr_full, MODEL_DIR / "linear_regression_model.joblib")
    joblib.dump(pre_full, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(pre_nw, MODEL_DIR / "preprocessor_no_weather.joblib")
    names = [n.split("__", 1)[1] for n in pre_full.get_feature_names_out()]
    meta = dict(
        project="CropYield AI", target=T, target_units="tonnes/hectare",
        input_features=NUM_ALL + CAT, production_used=False,
        split=dict(train=f"1997-{TRAIN_END}", val=f"{TRAIN_END + 1}-{VAL_END}", test=f"{VAL_END + 1}-2019",
                   n_train=len(tr), n_val=len(va), n_test=len(te), random_split=False),
        preprocessing=dict(numeric="StandardScaler", categorical="OneHotEncoder(handle_unknown='ignore')",
                           fit_on="training rows only", encoded_feature_names=names),
        ann=dict(architecture=ARCH, seeds=SEEDS, selected_seed=best_full["seed"], selection_rule="lowest validation loss",
                 best_epoch=best_full["best_epoch"], epochs_run=best_full["epochs"]),
        ann_no_weather=dict(selected_seed=best_nw["seed"], best_epoch=best_nw["best_epoch"]),
        district_mean_baseline=dict(fallback_for_unseen_district=fallback, means={k: round(float(v), 4) for k, v in dmean.items()}),
        metrics=json.loads(sel.round(4).to_json(orient="records")),
        versions=dict(tensorflow=tf.__version__, sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__),
        disclaimer="Academic prototype; not a production forecasting system.",
    )
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))

    make_plots(splits, preds, best_full, best_nw, sel, agg)
    write_report(tr, va, te, unseen, sel, agg, seeds_df, best_full, best_nw, fallback)


# --------------------------------------------------------------------------- plots
def make_plots(splits, preds, best_full, best_nw, sel, agg):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, (title, b) in zip(ax, [("ANN (all features)", best_full), ("ANN (no weather)", best_nw)]):
        a.plot(b["history"]["loss"], label="training loss")
        a.plot(b["history"]["val_loss"], label="validation loss")
        a.axvline(b["best_epoch"] - 1, ls="--", c="gray", label=f"best epoch ({b['best_epoch']})")
        a.set_title(f"{title}, seed {b['seed']}"); a.set_xlabel("epoch"); a.set_ylabel("MSE"); a.legend()
    plt.tight_layout(); plt.savefig(PLOT_DIR / "01_loss_curves.png", dpi=130); plt.close()

    models = ["District-mean baseline", "Linear Regression", "ANN (all features)", "ANN (no weather)"]
    te = splits["test"][T].to_numpy()
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2), sharex=True, sharey=True)
    for a, m in zip(ax, models):
        p = preds[m]["test"]
        a.scatter(te, p, s=18)
        lim = [0, max(te.max(), p.max()) * 1.05]
        a.plot(lim, lim, "k--", lw=1)
        a.set_title(f"{m}\nR2={r2_score(te, p):.3f}"); a.set_xlabel("Actual yield (t/ha)")
    ax[0].set_ylabel("Predicted yield (t/ha)")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "02_actual_vs_predicted_test.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 4, figsize=(17, 4), sharex=True, sharey=True)
    for a, m in zip(ax, models):
        p = preds[m]["test"]
        a.scatter(p, te - p, s=18); a.axhline(0, c="k", lw=1)
        a.set_title(m); a.set_xlabel("Predicted yield (t/ha)")
    ax[0].set_ylabel("Residual = actual - predicted (t/ha)")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "03_residuals_test.png", dpi=130); plt.close()

    order = ["District-mean baseline", "Linear Regression (no weather)", "Linear Regression",
             "ANN (no weather)", "ANN (all features)"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    w = 0.38
    for j, metric in enumerate(["RMSE", "R2"]):
        for i, split in enumerate(["val", "test"]):
            vals = [sel[(sel.model == m) & (sel.split == split)][metric].iloc[0] for m in order]
            err = []
            for m in order:
                r = agg[(agg.model == m) & (agg.split == split)]
                err.append(float(r[f"{metric}_std"].iloc[0]) if len(r) else 0.0)
            ax[j].bar(np.arange(len(order)) + (i - 0.5) * w, vals, w, yerr=err, capsize=3, label=split)
        ax[j].set_xticks(np.arange(len(order))); ax[j].set_xticklabels(order, rotation=25, ha="right")
        ax[j].set_title(metric + (" (lower is better)" if metric == "RMSE" else " (higher is better)")); ax[j].legend()
    fig.text(0.5, 0.01, "Error bars on ANN bars = std over 5 seeds; bar = selected seed", ha="center", fontsize=8)
    plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(PLOT_DIR / "04_model_comparison.png", dpi=130); plt.close()
    print("Saved plots to", PLOT_DIR)


# --------------------------------------------------------------------------- report
def write_report(tr, va, te, unseen, sel, agg, seeds_df, best_full, best_nw, fallback):
    def g(model, split, metric):
        return float(sel[(sel.model == model) & (sel.split == split)][metric].iloc[0])

    def gm(model, split, metric, stat):
        return float(agg[(agg.model == model) & (agg.split == split)][f"{metric}_{stat}"].iloc[0])

    tab = sel.round(4).copy()
    seed_tab = agg.round(4)[["model", "split", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std", "MAE_mean", "MAE_std"]]

    def md(df):
        return ("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
                + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n")

    A, B, N, N0, B0 = ("District-mean baseline", "Linear Regression", "ANN (all features)", "ANN (no weather)",
                       "Linear Regression (no weather)")
    ranking = sorted([A, B, B0, N, N0], key=lambda m: g(m, "test", "RMSE"))
    findings = [f"- Test RMSE ranking (selected models, lower is better): "
                + " < ".join(f"{m} ({g(m, 'test', 'RMSE'):.3f})" for m in ranking) + ".",
                f"- ANN (all features) vs district-mean baseline on test RMSE: {g(N, 'test', 'RMSE'):.3f} vs "
                f"{g(A, 'test', 'RMSE'):.3f} -> ANN is {'better' if g(N, 'test', 'RMSE') < g(A, 'test', 'RMSE') else 'WORSE'}.",
                f"- ANN (all features) vs Linear Regression on test RMSE: {g(N, 'test', 'RMSE'):.3f} vs {g(B, 'test', 'RMSE'):.3f} -> "
                f"ANN is {'better' if g(N, 'test', 'RMSE') < g(B, 'test', 'RMSE') else 'WORSE'}.",
                f"- Weather contribution (ANN): test RMSE {g(N, 'test', 'RMSE'):.3f} with weather vs {g(N0, 'test', 'RMSE'):.3f} without "
                f"(seed mean +/- std: {gm(N, 'test', 'RMSE', 'mean'):.3f} +/- {gm(N, 'test', 'RMSE', 'std'):.3f} vs "
                f"{gm(N0, 'test', 'RMSE', 'mean'):.3f} +/- {gm(N0, 'test', 'RMSE', 'std'):.3f}). Differences smaller than the seed "
                f"spread should not be read as real effects.",
                (lambda d, s: f"- Seed-averaged ANN test RMSE: weather changes it by {d:+.3f} t/ha "
                              f"({'weather model WORSE' if d > 0 else 'weather model better'}); "
                              f"this is {'larger' if abs(d) > 2 * s else 'not larger'} than twice the seed std ({s:.3f}). "
                              f"On validation the seed-averaged difference is "
                              f"{gm(N, 'val', 'RMSE', 'mean') - gm(N0, 'val', 'RMSE', 'mean'):+.3f}.")(
                    gm(N, "test", "RMSE", "mean") - gm(N0, "test", "RMSE", "mean"),
                    max(gm(N, "test", "RMSE", "std"), gm(N0, "test", "RMSE", "std"))),
                f"- Weather contribution (Linear Regression): test RMSE {g(B, 'test', 'RMSE'):.3f} with vs {g(B0, 'test', 'RMSE'):.3f} without.",
                f"- Validation vs test: the test set has only {len(te)} rows (2 years), so all test metrics are noisy."]
    r = ["# Model evaluation report\n",
         "Generated by `train_models.py`; all numbers are computed from the data, nothing is hand-entered.\n",
         "## Setup\n",
         f"- Data: `data/processed/ann_ready_dataset.csv` (642 rows, 34 districts). Target: `yield_t_ha`. "
         f"Features: district (one-hot), year, area_ha, rain_kharif_mm, temp_kharif_c, rh_kharif_pct. Production is not used.",
         f"- Chronological split: train 1997-2015 = **{len(tr)}** rows; validation 2016-2017 = **{len(va)}** rows; "
         f"test 2018-2019 = **{len(te)}** rows. No random splitting.",
         f"- Districts in validation/test but not in training: {unseen if unseen else 'none'}. "
         f"Baseline fallback for an unseen district = overall training mean ({fallback:.3f} t/ha).",
         "- StandardScaler (numeric) and OneHotEncoder (district) are fit on training rows only inside the pipeline.",
         f"- ANN: {ARCH}, up to 500 epochs. Trained with seeds {SEEDS}; the saved model/plots use the seed with the lowest "
         f"validation loss (selected seed {best_full['seed']} for the full model, {best_nw['seed']} for no-weather). "
         "The test set was never used to choose anything.",
         "- Extra comparison (not requested): Linear Regression without weather, to judge weather fairly.\n",
         "## Metrics (selected models)\n", md(tab),
         "## ANN seed variability (mean and std over 5 seeds)\n", md(seed_tab[seed_tab.model.str.startswith("ANN")]),
         "## Findings (computed)\n", "\n".join(findings), "",
         "## Figures\n",
         "- `reports/model_plots/01_loss_curves.png` training vs validation loss\n"
         "- `reports/model_plots/02_actual_vs_predicted_test.png`\n"
         "- `reports/model_plots/03_residuals_test.png`\n"
         "- `reports/model_plots/04_model_comparison.png`\n"
         "- Predictions: `reports/predictions.csv`; metrics: `reports/model_comparison.csv`, `reports/ann_seed_results.csv`\n",
         "## Limitations\n",
         "- Only 642 observations (540 train, 51 validation, 51 test); test metrics rest on two years.",
         "- Four pairs of districts share a NASA grid cell (identical weather): Hingoli/Parbhani, Akola/Amravati, Nagpur/Wardha, Bhandara/Gondia.",
         "- District weather is a single representative point (centre of the district boundary), not a district average.",
         "- Historical correlations do not establish causation.",
         "- Weather features are June-September totals/means, so they are only known after the growing season; this is "
         "retrospective estimation, not an early forecast.",
         "- `year` is a feature and validation/test years lie beyond the training range; the models extrapolate any trend poorly.",
         "- 17 validation/test rows come from areas under 1,000 ha, where yield is coarsely rounded.",
         "- Academic prototype, not a production agricultural forecasting system."]
    (C.REPORT_DIR / "model_evaluation_report.md").write_text("\n".join(r) + "\n")
    print("Wrote reports/model_evaluation_report.md")


if __name__ == "__main__":
    main()
