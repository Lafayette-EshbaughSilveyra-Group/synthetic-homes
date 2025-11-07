import os
import csv
import json
from typing import Dict, Any, List, Tuple, Optional

R_SI_TO_R_IP = 1.0 / 0.1761  # R_SI * (1/0.1761) = R_IP


def walk_idfs(root: str) -> List[str]:
    idfs = []
    for dp, dn, fn in os.walk(root):
        for name in fn:
            if name.lower().endswith(".idf"):
                idfs.append(os.path.join(dp, name))
    return sorted(idfs)


def float_or_none(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_idf_objects(text: str) -> List[Tuple[str, List[str]]]:
    """
    Very lightweight IDF parser returning a list of (OBJECTTYPE, [fields...])
    Strips comments starting with ! and !-; concatenates lines until ; is found.
    Field list includes everything after the object type.
    """
    objects: List[Tuple[str, List[str]]] = []
    buf: List[str] = []
    for raw in text.splitlines():
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
            fields = parts[1:]
            objects.append((obj_type, fields))
    return objects


def build_lookup(objects: List[Tuple[str, List[str]]]) -> Dict[str, List[List[str]]]:
    d: Dict[str, List[List[str]]] = {}
    for t, fields in objects:
        d.setdefault(t, []).append(fields)
    return d


def collect_material_ks(lookup: Dict[str, List[List[str]]]) -> Dict[str, Dict[str, float]]:
    """
    Return a dict name-> {'thickness': m, 'k': W/m-K} for MATERIAL only.
    Field positions (EnergyPlus standard for MATERIAL):
      1: Name, 2: Roughness, 3: Thickness (m), 4: Conductivity (W/m-K), 5: Density, 6: Specific Heat, ...
    """
    out = {}
    for fields in lookup.get("MATERIAL", []):
        if len(fields) >= 4:
            name = fields[0].strip().lower()
            thickness = float_or_none(fields[2])
            k = float_or_none(fields[3])
            if name and thickness is not None and k is not None and k > 0 and thickness > 0:
                out[name] = {"thickness": thickness, "k": k}
    return out


def construction_layers(lookup: Dict[str, List[List[str]]]) -> Dict[str, List[str]]:
    """
    Map construction name -> list of layer names (strings). For MATERIAL only layers we'll handle.
    Field positions (EnergyPlus standard for CONSTRUCTION):
      1: Name, 2..: layer names
    """
    cons = {}
    for fields in lookup.get("CONSTRUCTION", []):
        if len(fields) >= 2:
            name = fields[0].strip().lower()
            layers = [f.strip().lower() for f in fields[1:] if f.strip()]
            cons[name] = layers
    return cons


def r_ip_for_construction(cons_name: str, cons_layers: Dict[str, List[str]], matdb: Dict[str, Dict[str, float]]) -> \
Optional[float]:
    """
    Compute nominal R (IP) for a construction by summing thickness/k across MATERIAL layers.
    Ignores air films, resistances from non-'MATERIAL' types, etc. Best-effort.
    """
    if not cons_name:
        return None
    layers = cons_layers.get(cons_name.lower())
    if not layers:
        return None
    r_si = 0.0
    any_layer = False
    for lname in layers:
        m = matdb.get(lname)
        if m:
            any_layer = True
            r_si += (m["thickness"] / m["k"])
    if not any_layer or r_si <= 0:
        return None
    return r_si * R_SI_TO_R_IP


def polygon_area_xy(coords: List[Tuple[float, float]]) -> float:
    if len(coords) < 3:
        return 0.0
    s = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def extract_from_idf_file(idf_path: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "file": os.path.basename(idf_path),
        "path": idf_path,
        "R_wall": None,
        "R_roof": None,
        "U_window": None,
        "ACH": None,
        "COP_heat": None,
        "COP_cool": None,
        "area_m2": None,
        "height_m": None,
        "volume_m3": None,
    }
    try:
        with open(idf_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        rec["error"] = str(e)
        return rec

    objects = parse_idf_objects(text)
    lookup = build_lookup(objects)

    # Materials and constructions
    matdb = collect_material_ks(lookup)
    cons = construction_layers(lookup)

    # Surfaces: capture which constructions are used by Walls and Roofs,
    # also compute floor area (z ~ 0) and max height (max z among vertices).
    wall_cons_names = set()
    roof_cons_names = set()
    max_z = 0.0
    floor_area = None

    for fields in lookup.get("BUILDINGSURFACE:DETAILED", []):
        # Expected order: Name, SurfaceType, ConstructionName, ZoneName, OutsideBoundaryCondition, SunExposure, WindExposure, NumberOfVertices, then XYZ triplets
        surface_type = fields[1].strip().lower() if len(fields) > 1 else ""
        cons_name = fields[2].strip().lower() if len(fields) > 2 else ""
        if surface_type == "wall":
            if cons_name:
                wall_cons_names.add(cons_name)
        elif surface_type == "roof":
            if cons_name:
                roof_cons_names.add(cons_name)

        # vertices start at field index 8 (0-based), i.e., position 9 in 1-based docs
        verts = []
        i = 8
        while i + 2 < len(fields):
            x = float_or_none(fields[i]);
            y = float_or_none(fields[i + 1]);
            z = float_or_none(fields[i + 2])
            if x is None or y is None or z is None:
                break
            verts.append((x, y, z))
            if z is not None:
                max_z = max(max_z, z)
            i += 3
        # detect floor by all z==0 (or very close)
        if verts and all(abs(v[2]) < 1e-6 for v in verts):
            floor_xy = [(v[0], v[1]) for v in verts]
            area = polygon_area_xy(floor_xy)
            if area > 0:
                # if multiple floors surfaces are seen, choose the largest one
                floor_area = max(area, floor_area or 0.0)

    # Compute R for walls/roofs using first matching construction (best-effort)
    r_wall_vals = []
    for c in wall_cons_names:
        r_ip = r_ip_for_construction(c, cons, matdb)
        if r_ip is not None:
            r_wall_vals.append(r_ip)
    r_roof_vals = []
    for c in roof_cons_names:
        r_ip = r_ip_for_construction(c, cons, matdb)
        if r_ip is not None:
            r_roof_vals.append(r_ip)

    rec["R_wall"] = min(r_wall_vals) if r_wall_vals else None  # conservative: use min R among walls
    rec["R_roof"] = min(r_roof_vals) if r_roof_vals else None  # conservative: use min R among roofs
    rec["area_m2"] = floor_area
    rec["height_m"] = max_z if max_z > 0 else None
    rec["volume_m3"] = (rec["area_m2"] * rec["height_m"]) if (rec["area_m2"] and rec["height_m"]) else None

    # Window U (simple glazing system)
    sg = lookup.get("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM", [])
    if sg and len(sg[0]) >= 3:
        # Field order: Name, UFactor, SHGC, VT
        rec["U_window"] = float_or_none(sg[0][1])

    # ACH from ZoneInfiltration:DesignFlowRate when using AirChanges/Hour method
    for fields in lookup.get("ZONEINFILTRATION:DESIGNFLOWRATE", []):
        # Find the token 'AirChanges/Hour' (case-insensitive); take first numeric field after it as ACH.
        ach = None
        for idx, val in enumerate(fields):
            if isinstance(val, str) and val.strip().lower() == "airchanges/hour":
                # scan forward for first numeric
                for j in range(idx + 1, len(fields)):
                    v = float_or_none(fields[j])
                    if v is not None:
                        ach = v
                        break
                break
        if ach is not None:
            rec["ACH"] = ach
            break

    # Cooling COP (very schema-dependent). Try common objects/fields.
    # 1) HVACTEMPLATE:ZONE:PTAC — field near 'Cooling_Coil_Gross_Rated_Cooling_COP'
    for fields in lookup.get("HVACTEMPLATE:ZONE:PTAC", []):
        # crude scan for a numeric that could be COP; keep the largest in a plausible range (2..8)
        cands = [float_or_none(v) for v in fields if float_or_none(v) is not None]
        cands = [v for v in cands if 1.5 <= v <= 10.0]
        if cands:
            rec["COP_cool"] = max(cands)
            break
    # 2) COIL:COOLING:DX:SINGLESPEED — rated COP is usually field 10 or near 'Gross_Rated_Cooling_COP'
    if rec.get("COP_cool") is None:
        for fields in lookup.get("COIL:COOLING:DX:SINGLESPEED", []):
            # try to pick a plausible COP among numeric fields
            cands = [float_or_none(v) for v in fields if float_or_none(v) is not None]
            cands = [v for v in cands if 1.5 <= v <= 10.0]
            if cands:
                rec["COP_cool"] = max(cands)
                break

    # Heating "COP" — many real IDFs use gas/electric with efficiency rather than COP; we record a proxy if found.
    # Try common heating coil/equipment objects for an efficiency in (0.7..1.0) or COP in (1.5..6).
    for key in [
        "COIL:HEATING:ELECTRIC", "COIL:HEATING:DX:SINGLESPEED",
        "COIL:HEATING:GAS", "FURNACE:HEATONLY", "AIRLOOPHVAC:UNITARYHEATCOOL"
    ]:
        if rec.get("COP_heat") is not None:
            break
        for fields in lookup.get(key, []):
            cands = [float_or_none(v) for v in fields if float_or_none(v) is not None]
            # heuristics: prefer numbers in 0.7..1.0 (efficiency) or 1.5..6 (COP)
            effs = [v for v in cands if 0.6 <= v <= 1.0]
            cops = [v for v in cands if 1.5 <= v <= 6.0]
            if cops:
                rec["COP_heat"] = max(cops)
                break
            if effs:
                rec["COP_heat"] = max(effs)
                break

    return rec


def extract_directory(root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idf_files = walk_idfs(root)
    for p in idf_files:
        rows.append(extract_from_idf_file(p))
    return rows


def save_outputs(rows: List[Dict[str, Any]], out_csv: str, out_json: str):
    cols = ["file", "path", "R_wall", "R_roof", "U_window", "ACH", "COP_heat", "COP_cool", "area_m2", "height_m",
            "volume_m3"]
    # fill keys
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
    print(f"Saved {len(rows)} rows:\n - {out_csv}\n - {out_json}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extract input-side variables from all IDFs in a directory (recursively).")
    p.add_argument("--root", default="dataset_real", help="Root directory containing .idf files")
    p.add_argument("--out_csv", default="real_building_params.csv", help="Output CSV path")
    p.add_argument("--out_json", default="real_building_params.json", help="Output JSON path")
    args = p.parse_args()
    rows = extract_directory(args.root)
    save_outputs(rows, args.out_csv, args.out_json)
