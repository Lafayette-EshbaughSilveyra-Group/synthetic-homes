import pandas as pd
import numpy as np
import re
import math

INPUT = "options_saturations.csv"
OUTPUT = "options_saturations_summary.csv"

df = pd.read_csv(INPUT)

# --- Handle wide/combined option tables ---
# If CSV has only combined `option` and a saturation_* column, convert to expected columns.
if "Parameter" not in df.columns:
    def _norm_col_local(c: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(c).strip().lower())
    norm_cols = {_norm_col_local(c): c for c in df.columns}

    # Option column
    for k in ["option", "combinedoption", "options"]:
        if k in norm_cols:
            df.rename(columns={norm_cols[k]: "Option"}, inplace=True)
            break

    # Saturation / weight column
    sat_col = None
    for nk, orig in norm_cols.items():
        if nk.startswith("saturation") or nk.startswith("weight") or nk.startswith("share") or nk.startswith("applicability"):
            sat_col = orig
            break
    if sat_col is None:
        raise ValueError(f"Could not find a saturation/weight column in {list(df.columns)}")
    df.rename(columns={sat_col: "Saturation"}, inplace=True)

    # Dummy Parameter for compatibility
    df["Parameter"] = "(combined)"

# Normalize column names / common variants
# We treat the CSV as a long table of (parameter/characteristic, option/level, saturation/weight).

def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", c.strip().lower())

colmap = {_norm_col(c): c for c in df.columns}

def rename_if_exists(norm_name: str, target_name: str):
    if norm_name in colmap:
        src = colmap[norm_name]
        if src != target_name:
            df.rename(columns={src: target_name}, inplace=True)

# Parameter / characteristic
for k in ["parameter", "characteristic", "measure", "upgrade", "configparam"]:
    rename_if_exists(k, "Parameter")

# Option / level / value
for k in ["option", "level", "value", "choice", "setting"]:
    rename_if_exists(k, "Option")

# Saturation / weight / share / applicability
# If multiple exist, prefer an explicit saturation, else weight/share.
for k in ["saturation", "sat", "share", "fraction", "applicability", "weight", "weights"]:
    if "Saturation" not in df.columns:
        rename_if_exists(k, "Saturation")

# Final sanity checks
required = ["Parameter", "Option", "Saturation"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns {missing}. Found columns: {list(df.columns)}")

# Wide/combined option block: if "Parameter" not in columns, try to detect and melt
if "Parameter" not in df.columns:
    # normalize column names for detection
    def _norm_col_local(c: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(c).strip().lower())
    norm_cols = {_norm_col_local(c): c for c in df.columns}

# Clean up values
df["Parameter"] = df["Parameter"].astype(str).str.strip()
df["Option"] = df["Option"].astype(str).str.strip()

# Coerce saturation/weights to numeric. Accept percentages like '12%'.
def _to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x.endswith("%"):
            try:
                return float(x[:-1]) / 100.0
            except ValueError:
                return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

df["Saturation"] = df["Saturation"].apply(_to_float)


def weighted_quantile(values, weights, quantiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return [np.nan for _ in quantiles]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum_w = np.cumsum(weights)
    total = cum_w[-1]
    out = []
    for q in quantiles:
        target = q * total
        idx = np.searchsorted(cum_w, target, side="left")
        idx = min(idx, len(values) - 1)
        out.append(values[idx])
    return out


def parse_num(text):
    if not isinstance(text, str):
        return np.nan
    m = re.search(r"([0-9]*\.?[0-9]+)", text)
    return float(m.group(1)) if m else np.nan


rows = []

# 1) HVAC Cooling Efficiency -> COP_cool from SEER
if (df["Parameter"] == "HVAC Cooling Efficiency").any():
    sub = df[df["Parameter"] == "HVAC Cooling Efficiency"].copy()
    sub["Saturation"] = pd.to_numeric(sub["Saturation"], errors="coerce")
    sub["seer"] = sub["Option"].apply(parse_num)
    # keep only plausible SEERs
    sub.loc[~sub["seer"].between(5, 40), "seer"] = np.nan
    sub["cop_cool"] = sub["seer"] / 3.412
    qs = weighted_quantile(sub["cop_cool"], sub["Saturation"], [0.1, 0.5, 0.9])
    rows.append({
        "parameter": "HVAC Cooling Efficiency",
        "metric": "COP_cool",
        "q10": qs[0],
        "q50": qs[1],
        "q90": qs[2],
    })

# 2) HVAC Heating Efficiency -> COP_heat-ish from AFUE / efficiencies
if (df["Parameter"] == "HVAC Heating Efficiency").any():
    sub = df[df["Parameter"] == "HVAC Heating Efficiency"].copy()
    sub["Saturation"] = pd.to_numeric(sub["Saturation"], errors="coerce")
    sub["val"] = sub["Option"].apply(parse_num)

    # Heuristic:
    # - If 1 < val <= 100: treat as AFUE % and /100
    # - If 0.2 <= val <= 1.2: treat as already an efficiency fraction
    eff = []
    w = []
    for v, s in zip(sub["val"], sub["Saturation"]):
        if math.isnan(v) or s <= 0:
            continue
        if 1 < v <= 100:
            eff.append(v / 100.0)
            w.append(s)
        elif 0.2 <= v <= 1.2:
            eff.append(v)
            w.append(s)
        # ignore HSPF / other encodings here for simplicity;
        # you can extend this if you see them in your file

    qs = weighted_quantile(np.array(eff), np.array(w), [0.1, 0.5, 0.9]) if eff else [np.nan] * 3
    rows.append({
        "parameter": "HVAC Heating Efficiency",
        "metric": "COP_heat_like",
        "q10": qs[0],
        "q50": qs[1],
        "q90": qs[2],
    })

# 3) Wall Insulation -> R_wall_SI (if encoded as R-## in IP)
wall_mask = df["Parameter"].astype(str).str.contains("Wall", case=False) & \
            df["Parameter"].astype(str).str.contains("Insulation", case=False)
if wall_mask.any():
    sub = df[wall_mask].copy()
    sub["Saturation"] = pd.to_numeric(sub["Saturation"], errors="coerce")
    sub["r_val"] = sub["Option"].apply(parse_num)
    # If values are already SI (typical range 0.5–10), keep them.
    # If they look like IP R-values (> 10), convert to SI.
    def r_to_si(r):
        if math.isnan(r):
            return np.nan
        if r > 10.0:
            return r * 0.1761  # IP -> SI
        return r              # assume already SI
    sub["r_si"] = sub["r_val"].apply(r_to_si)
    qs = weighted_quantile(sub["r_si"], sub["Saturation"], [0.1, 0.5, 0.9])
    rows.append({
        "parameter": "Wall Insulation",
        "metric": "R_wall_SI",
        "q10": qs[0],
        "q50": qs[1],
        "q90": qs[2],
    })

# 4) Roof Insulation -> R_roof_SI
roof_mask = df["Parameter"].astype(str).str.contains("Roof", case=False) & \
            df["Parameter"].astype(str).str.contains("Insulation", case=False)
if roof_mask.any():
    sub = df[roof_mask].copy()
    sub["Saturation"] = pd.to_numeric(sub["Saturation"], errors="coerce")
    sub["r_val"] = sub["Option"].apply(parse_num)
    def r_to_si_roof(r):
        if math.isnan(r):
            return np.nan
        if r > 10.0:
            return r * 0.1761
        return r
    sub["r_si"] = sub["r_val"].apply(r_to_si_roof)
    qs = weighted_quantile(sub["r_si"], sub["Saturation"], [0.1, 0.5, 0.9])
    rows.append({
        "parameter": "Roof Insulation",
        "metric": "R_roof_SI",
        "q10": qs[0],
        "q50": qs[1],
        "q90": qs[2],
    })

# 5) Window U-Factor -> assume already SI or in normal range
if (df["Parameter"] == "Window U-Factor").any():
    sub = df[df["Parameter"] == "Window U-Factor"].copy()
    sub["Saturation"] = pd.to_numeric(sub["Saturation"], errors="coerce")
    sub["u"] = sub["Option"].apply(parse_num)


    # Heuristic: if clearly IP (<= 1.2), convert to SI; else assume SI
    def to_si(u):
        if math.isnan(u):
            return np.nan
        if u <= 1.2:
            return u * 5.678  # Btu/h-ft2-F -> W/m2-K
        return u


    sub["u_si"] = sub["u"].apply(to_si)
    qs = weighted_quantile(sub["u_si"], sub["Saturation"], [0.1, 0.5, 0.9])
    rows.append({
        "parameter": "Window U-Factor",
        "metric": "U_window_SI",
        "q10": qs[0],
        "q50": qs[1],
        "q90": qs[2],
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT, index=False)
print(summary)
