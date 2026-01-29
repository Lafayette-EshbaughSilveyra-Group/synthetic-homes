import pandas as pd
import numpy as np
import re
import math

INPUT = "options_saturations.csv"
OUTPUT = "options_saturations_summary.csv"

df = pd.read_csv(INPUT)

# Expect columns: option, saturation_4A (or similar)
# Rename to standard names
opt_col = None
sat_col = None
for c in df.columns:
    lc = c.lower()
    if opt_col is None and lc.startswith("option"):
        opt_col = c
    if sat_col is None and (lc.startswith("saturation") or lc.startswith("weight")):
        sat_col = c

if opt_col is None or sat_col is None:
    raise ValueError(f"Could not find option/saturation columns in {df.columns}")

df = df.rename(columns={opt_col: "Option", sat_col: "Saturation"})

# Ensure numeric saturation
df["Saturation"] = pd.to_numeric(df["Saturation"], errors="coerce").fillna(0)

rows = []

# Helpers
num_re = re.compile(r"([0-9]*\.?[0-9]+)")

def extract_segment(opt: str, prefix: str):
    """Return the full combined segment starting with prefix up to '__' or end."""
    if not isinstance(opt, str):
        return ""
    m = re.search(prefix + r".*?(?=__|$)", opt, flags=re.IGNORECASE)
    return m.group(0) if m else ""


def extract_num_from_segment(seg: str):
    """Extract first number from a segment, else NaN."""
    if not seg:
        return np.nan
    norm = re.sub(r"(?P<int>\d+)p(?P<frac>\d+)", r"\g<int>.\g<frac>", seg)
    nm = num_re.search(norm)
    return float(nm.group(1)) if nm else np.nan


def extract_r(opt, prefix):
    """Extract numeric R-value from a combined option string."""
    seg = extract_segment(opt, prefix)
    val = extract_num_from_segment(seg)
    if math.isnan(val):
        return np.nan
    # convert IP R to SI if large
    if val > 10:
        return val * 0.1761
    return val


def extract_cop(opt, prefixes):
    """Extract COP from any of the given prefixes. Ignores '*_unknown'."""
    if not isinstance(prefixes, (list, tuple)):
        prefixes = [prefixes]
    for p in prefixes:
        seg = extract_segment(opt, p)
        if not seg:
            continue
        if "unknown" in seg.lower():
            continue
        val = extract_num_from_segment(seg)
        if not math.isnan(val):
            return val
    return np.nan


def weighted_quantiles(values, weights, qs=(0.1, 0.5, 0.9)):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return [np.nan for _ in qs]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cw = np.cumsum(weights)
    tot = cw[-1]
    out = []
    for q in qs:
        t = q * tot
        loc = np.searchsorted(cw, t, side="left")
        loc = min(loc, len(values) - 1)
        out.append(values[loc])
    return out

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_std(values, weights):
    """Population-style weighted std (consistent with np.average weighting)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & ~np.isnan(weights) & ~np.isnan(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    avg = np.average(values[mask], weights=weights[mask])
    var = np.average((values[mask] - avg) ** 2, weights=weights[mask])
    return float(np.sqrt(var))

# 1) HVAC Cooling COP (from combined options like 'COPc_3.2' or 'COPcool_3.2')
cool_vals = []
cool_w = []
for opt, w in zip(df["Option"], df["Saturation"]):
    v = extract_cop(opt, ["copc_", "copcool_", "coolcop_", "seer_"])
    if not math.isnan(v) and w > 0:
        # If SEER was accidentally captured, convert to COP (seer/3.412).
        if "seer" in opt.lower() and v > 5:
            v = v / 3.412
        cool_vals.append(v)
        cool_w.append(w)

if cool_vals:
    qs = weighted_quantiles(cool_vals, cool_w, qs=(0.10, 0.25, 0.50, 0.75, 0.90))
    rows.append(dict(
        parameter="HVAC Cooling Efficiency",
        metric="COP_cool",
        q10=qs[0],
        q25=qs[1],
        q50=qs[2],
        q75=qs[3],
        q90=qs[4],
        mean=weighted_mean(cool_vals, cool_w),
        std=weighted_std(cool_vals, cool_w),
    ))

# 2) HVAC Heating COP (from combined options like 'COPh_3.0' or 'COPheat_3.0')
heat_vals = []
heat_w = []
for opt, w in zip(df["Option"], df["Saturation"]):
    v = extract_cop(opt, ["coph_", "copheat_", "heatcop_", "hspf_"])
    if not math.isnan(v) and w > 0:
        # If HSPF was captured, rough COP-like conversion (hspf/3.412).
        if "hspf" in opt.lower() and v > 3:
            v = v / 3.412
        # If AFUE percent captured (unlikely), convert to fraction and leave as-is.
        if "afue" in opt.lower() and v > 1:
            v = v / 100.0
        heat_vals.append(v)
        heat_w.append(w)

if heat_vals:
    qs = weighted_quantiles(heat_vals, heat_w, qs=(0.10, 0.25, 0.50, 0.75, 0.90))
    rows.append(dict(
        parameter="HVAC Heating Efficiency",
        metric="COP_heat_like",
        q10=qs[0],
        q25=qs[1],
        q50=qs[2],
        q75=qs[3],
        q90=qs[4],
        mean=weighted_mean(heat_vals, heat_w),
        std=weighted_std(heat_vals, heat_w),
    ))

# 1) Wall insulation
wall_vals = []
wall_w = []
for opt, w in zip(df["Option"], df["Saturation"]):
    if "rwall" in opt.lower():
        v = extract_r(opt, "rwall_")
        if not math.isnan(v) and w > 0:
            wall_vals.append(v)
            wall_w.append(w)

if wall_vals:
    qs = weighted_quantiles(wall_vals, wall_w, qs=(0.10, 0.25, 0.50, 0.75, 0.90))
    rows.append(dict(
        parameter="Wall Insulation",
        metric="R_wall_SI",
        q10=qs[0],
        q25=qs[1],
        q50=qs[2],
        q75=qs[3],
        q90=qs[4],
        mean=weighted_mean(wall_vals, wall_w),
        std=weighted_std(wall_vals, wall_w),
    ))

# 2) Roof insulation
roof_vals = []
roof_w = []
for opt, w in zip(df["Option"], df["Saturation"]):
    if "rroof" in opt.lower():
        v = extract_r(opt, "rroof_")
        if not math.isnan(v) and w > 0:
            roof_vals.append(v)
            roof_w.append(w)

if roof_vals:
    qs = weighted_quantiles(roof_vals, roof_w, qs=(0.10, 0.25, 0.50, 0.75, 0.90))
    rows.append(dict(
        parameter="Roof Insulation",
        metric="R_roof_SI",
        q10=qs[0],
        q25=qs[1],
        q50=qs[2],
        q75=qs[3],
        q90=qs[4],
        mean=weighted_mean(roof_vals, roof_w),
        std=weighted_std(roof_vals, roof_w),
    ))

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT, index=False)
print(summary)
