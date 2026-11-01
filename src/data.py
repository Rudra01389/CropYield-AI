"""Read-only access to the Phase 2/3 data and evaluation outputs used by the app."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "ann_ready_dataset.csv")


def load_metadata() -> dict:
    return json.loads((ROOT / "models" / "model_metadata.json").read_text())


def load_comparison() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "model_comparison.csv")


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "predictions.csv")


def load_seed_results() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "ann_seed_results.csv")


def diagram_image(name: str):
    p = ROOT / "reports" / "diagrams" / name
    return str(p) if p.exists() else None


def loss_curve_image():
    p = ROOT / "reports" / "model_plots" / "01_loss_curves.png"
    return str(p) if p.exists() else None
