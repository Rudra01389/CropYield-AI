# Screenshot checklist (capture these manually)

Start the app first:
```bash
cd "Desktop/Crop "          # note: the folder name ends with a space
source venv/bin/activate     # or whichever environment has the requirements installed
streamlit run app.py         # opens http://localhost:8501
```
Use a full-width browser window; save as PNG/JPG named as below. Do not edit or crop out numbers.

| # | Screenshot | How to get it | Suggested file name |
|---|---|---|---|
| L1 | Login screen | Open the app (signed out): illustration, branding, "Welcome, Farmer!", the demo-farmer preview, the **Continue as Demo Farmer** button and the demo notice. | `L1_login.png` |
| 0 | Dashboard (landing page, after login) | Click **Continue as Demo Farmer**; capture "Welcome, Ramesh Patil!", the farmer-profile card, KPI cards, model-accuracy gauge, trend, map and tables (scroll for the rest). | `00_dashboard.png` |
| 1 | Prediction page (before predicting) | Sidebar → **Predict Yield**. Show the green header, sidebar and input form. | `01_predict_home.png` |
| 2 | Prediction result for a district | Pick **Pune** (or any district), keep defaults, click **Predict Yield**. Capture the three metric cards. | `02_prediction_result.png` |
| 3 | Second district for comparison | Pick **Kolhapur** or **Solapur**, click **Predict Yield**. | `03_prediction_second_district.png` |
| 4 | Input validation | Set *Cultivated area* to 0 and click **Predict Yield** (shows the error), or set rainfall to 6000 (shows the extrapolation warning). | `04_validation_message.png` |
| 5 | Data & trends page | Sidebar → **Data & Trends**. Capture the **Yield trends** tab (optionally overlay 2–3 districts), then the **District comparison** and **Weather vs yield** tabs. | `05_data_trends_*.png` |
| 6 | Model performance comparison | Sidebar → **Model Performance** → **Comparison** tab; capture the metric cards, the amber "ANN underperforms" banner, metrics table and RMSE/R² charts. | `06_model_performance.png` |
| 7 | Actual vs predicted graph | Same page → **Actual vs predicted** tab; choose *Test (2018-19)* and *ANN (all features)*; capture both the scatter and residual charts. | `07_actual_vs_predicted.png` |
| 8 | ANN training vs validation loss | Same page → **Training curves** tab (or open `reports/model_plots/01_loss_curves.png`). Optional: **Seed variability** tab. | `08_loss_curve.png` |
| 10 | Project folder structure | In VS Code Explorer (or Finder), expand the project root, `data/`, `models/`, `reports/`, `src/`, `tests/`. | `10_folder_structure.png` |
| 11 | Tests passing | In the terminal run `python -m pytest -q` and capture the `62 passed` line. | `11_tests_passed.png` |
| 12 | Architecture diagrams | Open the three PNGs in `reports/diagrams/`. | `12_diagrams.png` |
| 13 | District yield map | Sidebar → **Data & Trends → Yield map**; optionally highlight a district and hover to show the tooltip. | `13_yield_map.png` |
| 14 | Understanding the Prediction | After a prediction, scroll to **Understanding the Prediction** (input descriptions + importance chart + caveat). | `14_understanding.png` |
| 15 | Downloaded report | Click **Download Prediction Report**; screenshot the button and the CSV opened in Excel/Numbers. | `15_report_download.png` |

Tips: take screenshots of the real running app only; the first prediction can take a few seconds while TensorFlow loads.
