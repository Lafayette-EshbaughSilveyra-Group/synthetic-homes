import os
import numpy as np
import pandas as pd

# Paths
GEN_CSV = "building_params.csv"
RANGE_CSV = os.path.join("resstock", "options_saturations_summary.csv")
OUT_CSV = "realism_stats_summary.csv"

# (name, gen_col, summary_metric)
VARIABLES = [
    ("R_wall",   "R_wall",   "R_wall_SI"),
    ("R_roof",   "R_roof",   "R_roof_SI"),
    ("COP_cool", "COP_cool", "COP_cool"),
    ("COP_heat", "COP_heat", "COP_heat_like"),
]


def load_data():
    if not os.path.exists(GEN_CSV):
        raise FileNotFoundError(f"Generated dataset not found: {GEN_CSV}")
    if not os.path.exists(RANGE_CSV):
        raise FileNotFoundError(f"ResStock summary not found: {RANGE_CSV}")
    gen = pd.read_csv(GEN_CSV)
    ranges = pd.read_csv(RANGE_CSV)
    return gen, ranges


def get_resstock_iqr(summary_df: pd.DataFrame, metric: str):
    """
    Returns (q25, q75, iqr) if available, else (nan, nan, nan).
    """
    sub = summary_df[summary_df["metric"] == metric]
    if sub.empty:
        return np.nan, np.nan, np.nan

    row = sub.iloc[0]
    q25 = float(row.get("q25", np.nan))
    q75 = float(row.get("q75", np.nan))

    if np.isnan(q25) or np.isnan(q75):
        return q25, q75, np.nan

    return q25, q75, q75 - q25


def main():
    gen, ranges = load_data()
    rows = []

    for name, gen_col, metric in VARIABLES:
        if gen_col not in gen.columns:
            print(f"[skip] {name}: column '{gen_col}' not found in generated data.")
            continue

        # --- synthetic homes stats ---
        vals = (
            gen[gen_col]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
        )

        if vals.size == 0:
            print(f"[skip] {name}: no valid synthetic values.")
            continue

        syn_mean = float(np.mean(vals))
        syn_median = float(np.median(vals))

        # --- ResStock IQR ---
        q25, q75, iqr = get_resstock_iqr(ranges, metric)

        rows.append({
            "variable": name,
            "synthetic_mean": syn_mean,
            "synthetic_median": syn_median,
            "resstock_q25": q25,
            "resstock_q75": q75,
            "resstock_IQR": iqr,
        })

        print(
            f"[ok] {name}: "
            f"synthetic mean={syn_mean:.3f}, median={syn_median:.3f}, "
            f"ResStock IQR={iqr:.3f}"
        )

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"[ok] Wrote summary CSV: {OUT_CSV}")
    else:
        print("[warn] No variables processed; no CSV written.")


if __name__ == "__main__":
    main()