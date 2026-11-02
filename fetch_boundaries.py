"""Download district boundary polygons for the Maharashtra yield map.

Source: OpenStreetMap (via Nominatim /lookup), licence ODbL. The OSM relation id of every district was already
recorded by fetch_coordinates.py in data/raw/district_coordinates.csv, so no district is searched by name again.
Polygons are simplified by the service (polygon_threshold) to keep the file small - good for a map, not for surveying.

Run:  python fetch_boundaries.py   -> data/raw/maharashtra_district_boundaries.geojson
Districts whose polygon cannot be fetched or validated are listed and left out; nothing is drawn or guessed for them.
"""
import json
import time

import pandas as pd
import requests

import config as C

URL = "https://nominatim.openstreetmap.org/lookup"
HEADERS = {"User-Agent": "CropYieldAI-student-project/1.0 (fattesing.r@somaiya.edu)"}
OUT = C.RAW_DIR / "maharashtra_district_boundaries.geojson"
MH_LAT, MH_LON = (15.6, 22.1), (72.6, 80.9)   # rough Maharashtra bounding box (sanity check only)


def inside_maharashtra(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    pts = [pt for poly in polys for ring in poly for pt in ring]
    return all(MH_LON[0] <= x <= MH_LON[1] and MH_LAT[0] <= y <= MH_LAT[1] for x, y in pts)


def main():
    if not C.COORD_CSV.exists():
        raise SystemExit("Run fetch_coordinates.py first (it records the OSM relation ids).")
    coords = pd.read_csv(C.COORD_CSV)
    by_id = {int(r.osm_id): r.district for r in coords.itertuples() if r.osm_type == "relation"}
    ids = list(by_id)
    feats, seen = [], set()
    for i in range(0, len(ids), 20):                         # the API accepts up to 50 ids per call
        chunk = ids[i:i + 20]
        r = requests.get(URL, headers=HEADERS, timeout=60, params={
            "osm_ids": ",".join(f"R{x}" for x in chunk), "format": "geojson",
            "polygon_geojson": 1, "polygon_threshold": 0.005})
        r.raise_for_status()
        for f in r.json()["features"]:
            oid = int(f["properties"]["osm_id"])
            g = f.get("geometry") or {}
            if oid in by_id and g.get("type") in ("Polygon", "MultiPolygon") and inside_maharashtra(g):
                seen.add(oid)
                feats.append({"type": "Feature", "geometry": g, "properties": {
                    "district": by_id[oid], "osm_id": oid, "osm_name": f["properties"].get("name"),
                    "source": "OpenStreetMap contributors (ODbL)"}})
        time.sleep(1.2)                                       # Nominatim usage policy
    missing = sorted(by_id[i] for i in set(ids) - seen)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"Saved {OUT.name}: {len(feats)} district polygons, {OUT.stat().st_size / 1024:.0f} KB")
    print("Without a valid polygon:", missing or "none")


if __name__ == "__main__":
    main()
