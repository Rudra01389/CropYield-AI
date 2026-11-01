"""Get a representative (lat, lon) for each Maharashtra district in the crop file.

Source: OpenStreetMap Nominatim (https://nominatim.openstreetmap.org), ODbL licence.
For an administrative district it returns the centre point of the district boundary relation,
i.e. a representative point of the district, NOT the headquarters town.
Each answer is validated (type = administrative, inside Maharashtra's bounding box).
Districts that fail validation are listed and left out - nothing is guessed.

Run:  python fetch_coordinates.py   -> data/raw/district_coordinates.csv  (cached)
"""
import time

import pandas as pd
import requests

import config as C

URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "CropYieldAI-student-project/1.0 (fattesing.r@somaiya.edu)"}
MH_LAT, MH_LON = (15.6, 22.1), (72.6, 80.9)  # rough Maharashtra bounding box

# Names used by OSM today when they differ from the crop file (official renamings).
OSM_NAME = {"AURANGABAD": "Chhatrapati Sambhajinagar", "OSMANABAD": "Dharashiv",
            "AHMEDNAGAR": "Ahilyanagar", "GONDIA": "Gondiya"}


def query(name):
    r = requests.get(URL, headers=HEADERS, timeout=30, params={
        "format": "jsonv2", "limit": 5, "countrycodes": "in", "state": "Maharashtra", "county": name})
    r.raise_for_status()
    # prefer true district boundaries; sub-districts (tehsils) are rejected
    hits = sorted(r.json(), key=lambda h: h.get("addresstype") != "state_district")
    for hit in hits:
        lat, lon = float(hit["lat"]), float(hit["lon"])
        if (hit.get("category") == "boundary" and hit.get("addresstype") in ("state_district", "county")
                and "subdistrict" not in hit["display_name"].lower().split(",")[0]
                and MH_LAT[0] <= lat <= MH_LAT[1]
                and MH_LON[0] <= lon <= MH_LON[1] and "Maharashtra" in hit["display_name"]):
            return hit
    return None


def main(districts):
    out = C.COORD_CSV
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["district"])
    have = set(done["district"])
    rows = done.to_dict("records")
    failed = []
    for d in districts:
        if d in have:
            continue
        tries = [d.title()] + ([OSM_NAME[d]] if d in OSM_NAME else [])
        hit = None
        for t in tries:
            hit = query(t)
            time.sleep(1.2)  # Nominatim usage policy: max 1 request/second
            if hit:
                break
        if hit:
            rows.append({"district": d, "lat": float(hit["lat"]), "lon": float(hit["lon"]),
                         "osm_name": hit["name"], "osm_type": hit["osm_type"], "osm_id": hit["osm_id"],
                         "source": "OpenStreetMap Nominatim (boundary centre)"})
            print(f"OK   {d:12s} -> {hit['lat'][:7]}, {hit['lon'][:7]} ({hit['name']})")
        else:
            failed.append(d)
            print(f"FAIL {d}")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}  ({len(rows)} districts). Failed: {failed}")
    return failed


if __name__ == "__main__":
    import preprocess
    main(sorted(preprocess.load_crop()["district"].unique()))
