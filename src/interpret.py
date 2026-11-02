"""Permutation feature importance for the saved ANN (no retraining).

For each input the values are shuffled across the evaluation rows, the saved preprocessor + ANN predict again, and
the increase in RMSE is recorded. A large increase means the ANN's predictions on those rows depend on that input;
it says nothing about cause and effect. The district column is shuffled as ONE raw column, before one-hot encoding.
"""
import numpy as np
import pandas as pd

TARGET = "yield_t_ha"
DISPLAY = {"district": "District", "year": "Agricultural year", "area_ha": "Cultivated area",
           "rain_kharif_mm": "Rainfall (Jun-Sep)", "temp_kharif_c": "Temperature", "rh_kharif_pct": "Humidity"}


def evaluation_rows(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """The held-out TEST years from the saved split (e.g. '2018-2019')."""
    lo, hi = (int(v) for v in meta["split"]["test"].split("-"))
    return df[(df.year >= lo) & (df.year <= hi)].reset_index(drop=True)


def _rmse(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(((pred - y) ** 2).mean(axis=-1))


def permutation_importance(predictor, eval_df: pd.DataFrame, n_repeats: int = 30, seed: int = 0) -> pd.DataFrame:
    """Return one row per input: mean and std (over repeats) of the RMSE increase, in t/ha."""
    feats = list(predictor.features)
    X = eval_df[feats].reset_index(drop=True)
    y = eval_df[TARGET].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    frames = [X]                                            # index 0 = unshuffled baseline
    for f in feats:
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(X[f].to_numpy())
            frames.append(Xp)
    batch = pd.concat(frames, ignore_index=True)            # one predict call for everything
    out = predictor.model.predict(predictor.pre.transform(batch[feats]).astype("float32"), verbose=0).ravel()
    out = out.reshape(len(frames), len(X))

    base = float(_rmse(out[0], y))
    rows = []
    for i, f in enumerate(feats):
        inc = _rmse(out[1 + i * n_repeats: 1 + (i + 1) * n_repeats], y) - base
        rows.append(dict(feature=f, label=DISPLAY.get(f, f), importance=float(inc.mean()), std=float(inc.std())))
    res = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    res.attrs.update(baseline_rmse=base, n_rows=len(X), n_repeats=n_repeats)
    return res
