# CropYield AI - ANN-Based Rice Yield Prediction for Maharashtra

Soft Computing mini project. Predicts Kharif rice yield (tonnes/hectare) for Maharashtra districts from the year,
cultivated area and June-September rainfall, temperature and humidity, using an Artificial Neural Network.

> **Honest summary:** on the 2018-2019 test years the ANN did *not* beat a simple district-mean baseline or Linear
> Regression (test RMSE 0.502 vs 0.486 / 0.489 t/ha). See `FINAL_PROJECT_REPORT.md`.

## 1. Install (Python 3.11-3.13)
Run these in the project folder (its name is `Crop ` with a trailing space, so keep the quotes):
```bash
cd "Desktop/Crop "
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.13, TensorFlow 2.21.0, Streamlit 1.66.1, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4.

## 2. Run the web app
```bash
streamlit run app.py
```
Open http://localhost:8501. The app uses the already-trained model in `models/`; it does not retrain anything.

## 3. Run the tests (optional, ~10 s)
```bash
python -m pytest -q          # expected: 40 passed
```

## 4. Rebuild everything from scratch (optional)
Needs internet for steps 1-2. Inputs in `data/` (`India Agriculture Crop Production.csv`, Pune reference file) are never modified.
```bash
python fetch_coordinates.py   # district coordinates (OpenStreetMap Nominatim)  -> data/raw/district_coordinates.csv
python fetch_weather.py       # daily NASA POWER weather (cached per district)  -> data/raw/weather_daily_all_districts.csv
python preprocess.py          # clean + merge -> data/processed/*.csv, reports/data_quality_report.md, preprocessing_log.md
python eda.py                 # EDA plots -> reports/eda/
python train_models.py        # baseline, Linear Regression, ANN -> models/, reports/ (retrains the ANN!)
python fetch_boundaries.py    # district outlines for the yield map (OpenStreetMap) -> data/raw/maharashtra_district_boundaries.geojson
python make_diagrams.py       # diagrams -> reports/diagrams/
```
`train_models.py` retrains and overwrites the saved models; skip it unless you want to reproduce training.

## New in Phase 6
| Feature | Where | How to use |
|---|---|---|
| District yield map | **Data & trends → Yield map** | Hover a district for name, mean yield, records and years; use *Highlight a district* to outline one. |
| Understanding the Prediction | **Predict yield** (bottom of page) | Plain-language input descriptions and a permutation-importance chart computed with the saved ANN on the 2018-19 test rows. Importance shows predictive reliance, not causation. |
| Prediction report (CSV) | **Predict yield → Download Prediction Report** | Appears after a successful prediction; one-row CSV with your inputs, the ANN estimate, estimated production and the district average. |

## Project layout
| Path | Purpose |
|---|---|
| `app.py`, `src/` | Streamlit app (Predict, Data & trends, Model performance, About); `predictor.py` model loader and validation, `interpret.py` permutation importance, `report.py` CSV report, `data.py` loaders, `ui.py` + `style.css` look and feel |
| `models/` | `ann_model.keras`, `preprocessor.joblib`, `linear_regression_model.joblib`, `model_metadata.json` (+ no-weather ANN) |
| `data/` | original CSVs, `raw/` downloads, `processed/` datasets (`ann_ready_dataset.csv` = final) |
| `reports/` | data-quality, preprocessing and model-evaluation reports, metrics CSVs, `eda/`, `model_plots/`, `diagrams/` |
| `tests/` | automated checks (data, split, model loading, validation, app pages) |
| `config.py`, `fetch_*.py`, `preprocess.py`, `eda.py`, `train_models.py`, `make_diagrams.py` | pipeline scripts |
| `FINAL_PROJECT_REPORT.md` | full project report |
| `SCREENSHOT_CHECKLIST.md` | screenshots to capture for submission |
| `README_phase2.md` | notes on the data-preparation phase |

## Limitations
642 observations; four district pairs share a NASA weather grid cell; one weather point per district; weather is only
known after the season; no soil/irrigation data; correlation is not causation. Academic prototype, not for real decisions.
