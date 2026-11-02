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


def _signed_area(ring):
    """Planar signed area of a lon/lat ring: > 0 means counter-clockwise."""
    return 0.5 * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]))


def _rewind_for_d3(geom):
    """Vega/d3 draw spherical polygons: exterior rings must be CLOCKWISE (opposite of the GeoJSON RFC 7946
    convention used by OpenStreetMap), otherwise the polygon is treated as 'everything except the district'.
    Only the order of the same vertices is changed; no coordinate is altered."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    fixed = []
    for poly in polys:
        rings = []
        for i, ring in enumerate(poly):
            ccw = _signed_area([tuple(pt[:2]) for pt in ring]) > 0
            rings.append(ring[::-1] if (ccw if i == 0 else not ccw) else ring)   # exterior CW, holes CCW
        fixed.append(rings)
    return {"type": geom["type"], "coordinates": fixed if geom["type"] == "MultiPolygon" else fixed[0]}


def load_boundaries():
    """District polygons (OpenStreetMap) as a GeoJSON dict ready for Vega/Altair, or None if not downloaded."""
    p = ROOT / "data" / "raw" / "maharashtra_district_boundaries.geojson"
    try:
        gj = json.loads(p.read_text())
        for f in gj["features"]:
            f["geometry"] = _rewind_for_d3(f["geometry"])
        return gj
    except (OSError, ValueError, KeyError):
        return None


def load_district_points():
    """Representative point per district (centre of the OSM boundary) for map labels, or None."""
    p = ROOT / "data" / "raw" / "district_coordinates.csv"
    try:
        return pd.read_csv(p)[["district", "lat", "lon"]]
    except (OSError, KeyError):
        return None


def load_seed_results() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "ann_seed_results.csv")


def diagram_image(name: str):
    p = ROOT / "reports" / "diagrams" / name
    return str(p) if p.exists() else None


def loss_curve_image():
    p = ROOT / "reports" / "model_plots" / "01_loss_curves.png"
    return str(p) if p.exists() else None
