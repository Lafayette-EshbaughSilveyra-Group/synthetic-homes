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

def extract_r(opt, prefix):
    """Extract numeric R-value from a combined option string.
    We first grab the whole segment starting with prefix up to the next '__' delimiter,
    then look for the first number inside (e.g., 'R-30', 'R38').
    """
    if not isinstance(opt, str):
        return np.nan
    # Capture the full segment like 'Rwall_Wood_Stud,_R-11' or 'Rroof_Finished,_R-30'
    m = re.search(prefix + r".*?(?=__|$)", opt, flags=re.IGNORECASE)
    if not m:
        return np.nan
    seg = m.group(0)
    nm = num_re.search(seg)
    if not nm:
        return np.nan
    val = float(nm.group(1))
    # convert IP R to SI if large
    if val > 10:
        return val * 0.1761
    return val

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
    vs = np.array(wall_vals)
    ws = np.array(wall_w)
    qs = []
    for q in [0.1, 0.5, 0.9]:
        idx = np.argsort(vs)
        vs2 = vs[idx]
        ws2 = ws[idx]
        cw = np.cumsum(ws2)
        tot = cw[-1]
        t = q * tot
        loc = np.searchsorted(cw, t)
        loc = min(loc, len(vs2)-1)
        qs.append(vs2[loc])
    rows.append(dict(parameter="Wall Insulation", metric="R_wall_SI", q10=qs[0], q50=qs[1], q90=qs[2]))

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
    vs = np.array(roof_vals)
    ws = np.array(roof_w)
    qs = []
    for q in [0.1, 0.5, 0.9]:
        idx = np.argsort(vs)
        vs2 = vs[idx]
        ws2 = ws[idx]
        cw = np.cumsum(ws2)
        tot = cw[-1]
        t = q * tot
        loc = np.searchsorted(cw, t)
        loc = min(loc, len(vs2)-1)
        qs.append(vs2[loc])
    rows.append(dict(parameter="Roof Insulation", metric="R_roof_SI", q10=qs[0], q50=qs[1], q90=qs[2]))

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT, index=False)
print(summary)
