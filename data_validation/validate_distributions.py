
import argparse
import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import scipy's ks_2samp; if unavailable, provide a fallback
try:
    from scipy.stats import ks_2samp as _ks_2samp
    def ks_2samp(x, y):
        return _ks_2samp(x, y, alternative='two-sided', method='auto')
except Exception:
    # Asymptotic two-sample KS p-value approximation
    # Reference: Numerical Recipes / Massey (1951)
    def ks_2samp(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        x.sort(); y.sort()
        n1, n2 = len(x), len(y)
        if n1 == 0 or n2 == 0:
            return type('KSResult', (), {'statistic': np.nan, 'pvalue': np.nan})
        data_all = np.concatenate([x, y])
        cdf1 = np.searchsorted(x, data_all, side='right') / n1
        cdf2 = np.searchsorted(y, data_all, side='right') / n2
        d = np.max(np.abs(cdf1 - cdf2))
        # Effective n and asymptotic p-value
        ne = n1 * n2 / (n1 + n2)
        # Kolmogorov distribution approximation
        # p ≈ 2 * sum_{j=1..∞} (-1)^{j-1} exp(-2 j^2 (d * sqrt(ne))^2)
        t = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
        # compute a few terms
        p = 0.0
        for j in range(1, 101):
            p += (-1)**(j-1) * math.exp(-2.0 * (j*j) * (t*t))
        p = max(0.0, min(1.0, 2.0 * p))
        return type('KSResult', (), {'statistic': d, 'pvalue': p})

PREFERRED_VARS = [
    "R_wall", "R_roof", "U_window", "ACH",
    "COP_heat", "COP_cool", "area_m2", "volume_m3"
]

def ecdf(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    x = np.sort(arr)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

def clean_series(s: pd.Series) -> np.ndarray:
    return s.replace([np.inf, -np.inf], np.nan).dropna().astype(float).values

def quantiles(a: np.ndarray, qs=(0.1, 0.5, 0.9)):
    if a.size == 0:
        return [np.nan for _ in qs]
    return [float(np.quantile(a, q)) for q in qs]

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Validate distributional similarity between real and generated datasets (ECDF + KS).")
    ap.add_argument("--real_csv", required=True, help="CSV for real dataset (rows=homes, cols=variables)")
    ap.add_argument("--gen_csv", required=True, help="CSV for generated dataset")
    ap.add_argument("--out_dir", default="validation_outputs", help="Directory to save plots and summary")
    ap.add_argument("--vars", nargs="*", default=None, help="Variables to compare (default: intersection of preferred vars)")
    ap.add_argument("--min_n", type=int, default=10, help="Minimum non-NaN count per group to run test/plot")
    ap.add_argument("--include_log_area", action="store_true", help="If set, also compare log(area_m2) when available")
    args = ap.parse_args()

    ensure_outdir(args.out_dir)

    real = pd.read_csv(args.real_csv)
    gen  = pd.read_csv(args.gen_csv)

    # Determine variables
    if args.vars:
        vars_to_compare = args.vars
    else:
        cols_real = set(real.columns)
        cols_gen = set(gen.columns)
        inter = [v for v in PREFERRED_VARS if v in cols_real and v in cols_gen]
        vars_to_compare = inter

    # Optionally add log-area
    if args.include_log_area and "area_m2" in real.columns and "area_m2" in gen.columns:
        real["log_area_m2"] = np.log(real["area_m2"])
        gen["log_area_m2"]  = np.log(gen["area_m2"])
        vars_to_compare = vars_to_compare + ["log_area_m2"]

    if not vars_to_compare:
        raise SystemExit("No overlapping variables to compare. Specify --vars or check your CSV columns.")

    summary_rows = []

    for var in vars_to_compare:
        r = clean_series(real[var]) if var in real.columns else np.array([])
        g = clean_series(gen[var])  if var in gen.columns  else np.array([])

        n_r, n_g = r.size, g.size
        if n_r < args.min_n or n_g < args.min_n:
            print(f"[skip] {var}: insufficient data (real={n_r}, gen={n_g}, min_n={args.min_n})")
            summary_rows.append({
                "variable": var, "n_real": n_r, "n_gen": n_g,
                "ks_D": np.nan, "ks_p": np.nan,
                "real_q10": np.nan, "real_q50": np.nan, "real_q90": np.nan,
                "gen_q10": np.nan, "gen_q50": np.nan, "gen_q90": np.nan,
                "note": "insufficient data"
            })
            continue

        # KS test
        ks = ks_2samp(r, g)
        D = float(ks.statistic) if not np.isnan(ks.statistic) else np.nan
        p = float(ks.pvalue) if not np.isnan(ks.pvalue) else np.nan

        # Quantiles
        rq10, rq50, rq90 = quantiles(r)
        gq10, gq50, gq90 = quantiles(g)

        # ECDF plot
        xr, yr = ecdf(r)
        xg, yg = ecdf(g)

        plt.figure(figsize=(6, 4.5))
        plt.step(xr, yr, where="post", linewidth=2, label="Real")
        plt.step(xg, yg, where="post", linewidth=2, label="Generated")
        title = f"ECDF: {var} (KS D={D:.3f}, p={p:.4f})"
        plt.title(title)
        plt.xlabel(var)
        plt.ylabel("ECDF")
        plt.grid(alpha=0.3)
        plt.legend()
        out_png = os.path.join(args.out_dir, f"ecdf_{var}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

        print(f"[ok] {var}: KS D={D:.3f}, p={p:.4f} (n_real={n_r}, n_gen={n_g})  -> {out_png}")

        summary_rows.append({
            "variable": var, "n_real": n_r, "n_gen": n_g,
            "ks_D": D, "ks_p": p,
            "real_q10": rq10, "real_q50": rq50, "real_q90": rq90,
            "gen_q10": gq10, "gen_q50": gq50, "gen_q90": gq90,
            "note": ""
        })

    # Save summary CSV and JSON
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, "ks_summary.csv")
    summary_json = os.path.join(args.out_dir, "ks_summary.json")
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary:\n - {summary_csv}\n - {summary_json}")
    print("Done.")

if __name__ == "__main__":
    main()
