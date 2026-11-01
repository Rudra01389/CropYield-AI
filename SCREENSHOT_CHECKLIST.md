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
| 1 | Home / prediction page (before predicting) | Sidebar → **Predict yield**. Show the green header, sidebar and input form. | `01_predict_home.png` |
| 2 | Prediction result for a district | Pick **Pune** (or any district), keep defaults, click **Predict Yield**. Capture the three metric cards. | `02_prediction_result.png` |
| 3 | Second district for comparison | Pick **Kolhapur** or **Solapur**, click **Predict Yield**. | `03_prediction_second_district.png` |
| 4 | Input validation | Set *Cultivated area* to 0 and click **Predict Yield** (shows the error), or set rainfall to 6000 (shows the extrapolation warning). | `04_validation_message.png` |
| 5 | Data & trends page | Sidebar → **Data & trends**; optionally overlay 2–3 districts; include the yield trend and the district bar chart (scroll for both). | `05_data_trends.png` |
| 6 | Model performance comparison | Sidebar → **Model performance**; capture the metric cards, warning banner, metrics table and RMSE bar chart. | `06_model_performance.png` |
| 7 | Actual vs predicted graph | Same page, scroll to **Actual vs predicted yield (ANN)**; show the *Test* subset. | `07_actual_vs_predicted.png` |
| 8 | ANN training vs validation loss | Same page, expand **Training vs validation loss** (or open `reports/model_plots/01_loss_curves.png`). | `08_loss_curve.png` |
| 9 | About & limitations page | Sidebar → **About & limitations** (architecture, data sources, disclaimer). | `09_about.png` |
| 10 | Project folder structure | In VS Code Explorer (or Finder), expand the project root, `data/`, `models/`, `reports/`, `src/`, `tests/`. | `10_folder_structure.png` |
| 11 | Tests passing | In the terminal run `python -m pytest -q` and capture the `28 passed` line. | `11_tests_passed.png` |
| 12 | Architecture diagrams | Open the three PNGs in `reports/diagrams/`. | `12_diagrams.png` |

Tips: take screenshots of the real running app only; the first prediction can take a few seconds while TensorFlow loads.
