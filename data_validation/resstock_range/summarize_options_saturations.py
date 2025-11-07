import pandas as pd
import numpy as np
import re
import math

INPUT = "options_saturations.csv"
OUTPUT = "options_saturations_summary.csv"

df = pd.read_csv(INPUT)


def weighted_quantile(values, weights, quantiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & (weights > 0)
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
if (df["Parameter"] == "Wall Insulation").any():
    sub = df[df["Parameter"] == "Wall Insulation"].copy()
    sub["r_ip"] = sub["Option"].apply(parse_num)
    # basic sanity filter
    sub.loc[~sub["r_ip"].between(1, 60), "r_ip"] = np.nan
    sub["r_si"] = sub["r_ip"] * 0.1761
    qs = weighted_quantile(sub["r_si"], sub["Saturation"], [0.1, 0.5, 0.9])
    rows.append({
        "parameter": "Wall Insulation",
        "metric": "R_wall_SI",
        "q10": qs[0],
        "q50": qs[1],
        "q90": qs[2],
    })

# 4) Roof Insulation -> R_roof_SI
if (df["Parameter"] == "Roof Insulation").any():
    sub = df[df["Parameter"] == "Roof Insulation"].copy()
    sub["r_ip"] = sub["Option"].apply(parse_num)
    sub.loc[~sub["r_ip"].between(1, 100), "r_ip"] = np.nan
    sub["r_si"] = sub["r_ip"] * 0.1761
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

# 6) Infiltration (if present and looks like ACHnat)
if (df["Parameter"] == "Infiltration").any():
    sub = df[df["Parameter"] == "Infiltration"].copy()
    sub["val"] = sub["Option"].apply(parse_num)
    # crude: keep only plausible ACHnat range
    sub.loc[~sub["val"].between(0.1, 3.0), "val"] = np.nan
    qs = weighted_quantile(sub["val"], sub["Saturation"], [0.1, 0.5, 0.9])
    rows.append({
        "parameter": "Infiltration",
        "metric": "ACH_nat",
        "q10": qs[0],
        "q50": qs[1],
        "q90": qs[2],
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT, index=False)
print(summary)
