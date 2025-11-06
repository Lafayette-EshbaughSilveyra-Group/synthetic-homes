import os
import json
import math
import csv
from typing import Dict, Any, List, Optional, Tuple


def walk_home_dirs(root: str) -> List[str]:
    homes = []
    for dirpath, dirnames, filenames in os.walk(root):
        if ("cleaned.geojson" in filenames) or ("expanded.idf" in filenames):
            homes.append(dirpath)
    return sorted(homes)


def load_geojson(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_from_geojson(gj: Dict[str, Any]) -> Dict[str, Any]:
    feat = (gj.get("features") or [None])[0] or {}
    props = feat.get("properties", {})
    rec = {}
    rec["source"] = "geojson"
    rec["name"] = str(props.get("name", props.get("Name", "unknown"))).replace(" ", "_")
    rec["R_wall"] = props.get("wall_r_value")
    rec["R_roof"] = props.get("roof_r_value")
    rec["U_window"] = props.get("window_u_value")
    rec["ACH"] = props.get("air_change_rate")
    rec["COP_heat"] = props.get("hvac_heating_cop")
    rec["COP_cool"] = props.get("hvac_cooling_cop")
    area_sqft = props.get("Total Square Feet Living Area")
    rec["area_m2"] = float(area_sqft) * 0.092903 if area_sqft is not None else None
    height_ft = props.get("height_ft")
    rec["height_m"] = float(height_ft) * 0.3048 if height_ft is not None else None
    rec["volume_m3"] = (rec["area_m2"] * rec["height_m"]) if (
                rec["area_m2"] is not None and rec["height_m"] is not None) else None
    return rec


def parse_idf_objects(idf_text: str) -> List[Tuple[str, Dict[str, str]]]:
    objects = []
    buf = []
    for raw in idf_text.splitlines():
        line = raw
        if "!-" in line:
            line = line.split("!-")[0]
        if "!" in line:
            line = line.split("!")[0]
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        if line.endswith(";"):
            block = " ".join(buf)
            buf = []
            parts = [p.strip(" ;") for p in block.split(",")]
            if not parts:
                continue
            obj_type = parts[0].upper()
            fields_list = parts[1:]
            fields = {}
            for i, val in enumerate(fields_list, start=1):
                fields[f"Field_{i}"] = val
            objects.append((obj_type, fields))
    return objects


def find_objects(objects, obj_type_prefix):
    prefix = obj_type_prefix.upper()
    return [(t, f) for (t, f) in objects if t.startswith(prefix)]


def float_or_none(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def polygon_area_xy(coords: List[Tuple[float, float]]) -> float:
    if len(coords) < 3:
        return 0.0
    s = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def extract_from_idf(idf_path: str) -> Dict[str, Any]:
    rec = {"source": "idf", "name": os.path.basename(os.path.dirname(idf_path))}
    try:
        with open(idf_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return rec
    objs = parse_idf_objects(text)

    # WINDOW MATERIAL: simple glazing -> U
    wins = find_objects(objs, "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM")
    if wins:
        # Field_2 corresponds to UFactor in standard order
        rec["U_window"] = float_or_none(wins[0][1].get("Field_2"))

    # ZONE INFILTRATION: DESIGN FLOW RATE -> ACH
    infl = find_objects(objs, "ZONEINFILTRATION:DESIGNFLOWRATE")
    if infl:
        # Typical order: Name, ZoneName, ScheduleName, Method, Design_Flow_Rate, Flow/Area, Flow/Person, ACH, ...
        # ACH is usually Field_8
        rec["ACH"] = float_or_none(infl[0][1].get("Field_8"))

    # HVACTEMPLATE:ZONE:PTAC -> cooling COP and gas heating efficiency (proxy for COP_heat)
    ptac = find_objects(objs, "HVACTEMPLATE:ZONE:PTAC")
    if ptac:
        # From your generator: Cooling_Coil_Gross_Rated_Cooling_COP and Gas_Heating_Coil_Efficiency
        # In positional terms these are commonly Field_6 and Field_9, but layouts vary; we try a few positions.
        for fld in ("Field_6", "Field_7", "Field_8", "Field_9", "Field_10"):
            val = float_or_none(ptac[0][1].get(fld))
            if val is not None:
                rec.setdefault("COP_cool", val)
                break
        # crude scan for another numeric as heating efficiency
        for fld in ("Field_8", "Field_9", "Field_10", "Field_11", "Field_12"):
            val = float_or_none(ptac[0][1].get(fld))
            if val is not None and ("COP_cool" not in rec or val != rec["COP_cool"]):
                rec.setdefault("COP_heat", val)

    # MATERIAL conductivity back-calc to R (IP) given known thickness in generator
    # Wall Material thickness in generator = 0.2 m, R_ip = 0.2 / (0.1761 * conductivity)
    mats = find_objects(objs, "MATERIAL")
    wall_k = None
    roof_k = None
    for _, fld in mats:
        name = fld.get("Field_1", "").lower()
        k = float_or_none(fld.get("Field_4"))
        if not k:
            continue
        if "wall material" in name and wall_k is None:
            wall_k = k
        if "roof material" in name and roof_k is None:
            roof_k = k
    if wall_k:
        rec["R_wall"] = 0.2 / (0.1761 * wall_k)
    if roof_k:
        rec["R_roof"] = 0.15 / (0.1761 * roof_k)

    # Derive height and floor area from BUILDINGSURFACE:DETAILED if possible
    surfs = find_objects(objs, "BUILDINGSURFACE:DETAILED")
    max_z = 0.0
    floor_coords = None
    for _, fld in surfs:
        # Vertices typically start at Field_9 as triples (X,Y,Z)
        coords = []
        i = 9
        while True:
            x = fld.get(f"Field_{i}");
            y = fld.get(f"Field_{i + 1}");
            z = fld.get(f"Field_{i + 2}")
            if x is None or y is None or z is None:
                break
            x = float_or_none(x);
            y = float_or_none(y);
            z = float_or_none(z)
            if x is None or y is None or z is None:
                break
            coords.append((x, y, z))
            max_z = max(max_z, z)
            i += 3
        # Detect floor by all z == 0
        if coords and all(abs(p[2]) < 1e-6 for p in coords):
            floor_coords = [(p[0], p[1]) for p in coords]
    if max_z > 0:
        rec["height_m"] = max_z
    if floor_coords:
        rec["area_m2"] = polygon_area_xy(floor_coords)
    if rec.get("area_m2") is not None and rec.get("height_m") is not None:
        rec["volume_m3"] = rec["area_m2"] * rec["height_m"]

    return rec


def merge_records(pref: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(fallback)
    out.update({k: v for k, v in pref.items() if v is not None})
    return out


def extract_dataset(root: str) -> List[Dict[str, Any]]:
    rows = []
    for home_dir in walk_home_dirs(root):
        gj_path = os.path.join(home_dir, "cleaned.geojson")
        idf_path = os.path.join(home_dir, "expanded.idf")
        gj = load_geojson(gj_path) if os.path.exists(gj_path) else None
        rec_gj = extract_from_geojson(gj) if gj else {}
        rec_idf = extract_from_idf(idf_path) if os.path.exists(idf_path) else {}
        # Prefer geojson values, fallback to idf-derived where missing
        rec = merge_records(rec_gj, rec_idf)
        rec["home_path"] = home_dir
        rows.append(rec)
    return rows


def save_csv_json(rows: List[Dict[str, Any]], out_csv: str, out_json: str):
    if not rows:
        print("No rows to save.")
        return
    # Determine columns
    cols = ["name", "home_path", "source", "R_wall", "R_roof", "U_window", "ACH", "COP_heat", "COP_cool", "area_m2",
            "height_m", "volume_m3"]
    # Ensure all keys exist
    for r in rows:
        for c in cols:
            r.setdefault(c, None)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(rows)} rows to:\n - {out_csv}\n - {out_json}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extract building parameter dataset from cleaned.geojson / expanded.idf")
    p.add_argument("--root", default="dataset", help="Root folder containing home subfolders")
    p.add_argument("--out_csv", default="building_params.csv", help="Output CSV path")
    p.add_argument("--out_json", default="building_params.json", help="Output JSON path")
    args = p.parse_args()
    rows = extract_dataset(args.root)
    save_csv_json(rows, args.out_csv, args.out_json)
