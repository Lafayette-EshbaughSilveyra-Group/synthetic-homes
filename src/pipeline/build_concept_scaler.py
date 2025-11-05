from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from experiments.constants import HVAC_TEXT_SAMPLES, INSULATION_TEXT_SAMPLES

# ---- concept enum ----
HVAC = "hvac"
INSUL = "insulation"

# ---- default sim vars (override with CLI flags if your column names differ) ----
DEFAULTS = {
    HVAC: {"sim_var": "Electricity:HVAC [J](Hourly)", "sim_stat": "mean",
           "out": "energyplus_data/hvac_scaler_params.json"},
    INSUL: {"sim_var": "Heating Coil Heating Energy [J](Hourly)", "sim_stat": "mean",
            "out": "energyplus_data/insulation_scaler_params.json"},
}


# ---- dataclasses ----
@dataclass
class ModStats:
    mean: float
    std: float


@dataclass
class ScalerParams:
    text: ModStats
    sim: ModStats


def _resolve_sim_block(entry: dict, sim_var: str) -> dict | None:
    """
    Resolve a simulation variable block from a summary entry with tolerant matching.
    Tries, in order:
      1) Exact key match
      2) Exact match after stripping trailing/leading whitespace
      3) Case/whitespace-insensitive match
      4) Suffix match (entry key endswith desired key), useful when a home-specific prefix is present
    Returns the matched block (dict) or None.
    """
    # 1) exact
    if sim_var in entry:
        return entry[sim_var]

    # Build normalized lookup maps
    wanted = sim_var.strip()
    by_stripped = {k.strip(): (k, v) for k, v in entry.items()}
    by_lower = {k.strip().lower(): (k, v) for k, v in entry.items()}

    # 2) stripped exact
    if wanted in by_stripped:
        _, v = by_stripped[wanted]
        return v

    # 3) case-insensitive exact
    if wanted.lower() in by_lower:
        _, v = by_lower[wanted.lower()]
        return v

    # 4) suffix match on stripped keys (handles prefixes like "FF_0001 ..." or trailing spaces)
    candidates = [(k, v) for k, v in entry.items() if k.strip().endswith(wanted)]
    if len(candidates) == 1:
        return candidates[0][1]
    if len(candidates) > 1:
        # Prefer the shortest key (least-prefix) as a heuristic
        k_best, v_best = sorted(candidates, key=lambda kv: len(kv[0]))[0]
        return v_best

    return None


# ---- utils ----
def _fit_mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    m = float(xs.mean())
    sd = float(xs.std(ddof=1) if len(xs) > 1 else 1.0)
    return m, max(sd, 1e-8)


def _nearest_index(value: float, bins: list[float]) -> int:
    return int(np.argmin([abs(value - b) for b in bins]))  # 0..4


# ---- text composition per concept ----
def _compose_hvac_text(meta_row: dict) -> tuple[str, int]:
    # Levels defined by your factorial:
    heat_bins = [0.7, 0.8, 0.9, 0.95, 1.0]
    cool_bins = [1.0, 2.0, 3.0, 3.5, 4.0]
    h_idx = _nearest_index(meta_row["hvac_heating_cop"], heat_bins)
    c_idx = _nearest_index(meta_row["hvac_cooling_cop"], cool_bins)
    hvac_idx = int(round((h_idx + c_idx) / 2))  # 0..4
    text = HVAC_TEXT_SAMPLES[hvac_idx]
    return text, hvac_idx


def _compose_insulation_text(meta_row: dict) -> tuple[str, int]:
    wall_bins = [4.0, 7.0, 13.0, 20.0, 30.0]
    roof_bins = [10.0, 20.0, 30.0, 40.0, 50.0]
    w_idx = _nearest_index(meta_row["wall_r_value"], wall_bins)
    r_idx = _nearest_index(meta_row["roof_r_value"], roof_bins)
    ins_idx = int(round((w_idx + r_idx) / 2))  # 0..4
    text = INSULATION_TEXT_SAMPLES[ins_idx]
    return text, ins_idx


def _ordinal_score_from_index(idx: int) -> float:
    # 0..4 -> 1..5
    return float(idx + 1)


# ---- main builder ----
def build_concept_scaler(concept: str,
                         sim_var: str | None = None,
                         sim_stat: str | None = None,
                         out: str | None = None) -> int:
    """
    Fit and save per-concept scaler parameters for text/simulation fusion.

    This function reads the factorial simulation summaries and metadata produced by the
    full-factorial EnergyPlus generator (i.e., `energyplus_data/summary_stats.json` and
    `energyplus_data/factorial_meta.json`) and computes z-score parameters that align
    the text-side signal and the simulation-side signal on the same scale, per concept
    (currently: "hvac" or "insulation").

    Workflow
    --------
    1) Map each synthetic home to a concept-specific ordinal text index (0..4) from the
       factorial metadata (e.g., HVAC levels derived from heating/cooling COP bins;
       insulation levels from wall/roof R-value bins). Convert that index to a numeric
       text score in [1..5].
    2) Extract a single simulation scalar per home from `summary_stats.json` using
       `sim_var` (column key) and `sim_stat` (e.g., "mean").
    3) Compute mean and std for the text scores and for the sim scalars; these become
       the scaler parameters.
    4) Write the result as JSON to `out`, e.g.:
           energyplus_data/hvac_scaler_params.json
           energyplus_data/insulation_scaler_params.json

    Parameters
    ----------
    concept : {"hvac", "insulation"}
        Which concept to build a scaler for. Determines how metadata is mapped to
        a text level and which defaults to use.
    sim_var : str, optional
        Key in `summary_stats.json` for the simulation variable to use (defaults are
        set in DEFAULTS per concept).
    sim_stat : {"mean","min","max","std"}, optional
        Which statistic from the selected simulation variable to use.
    out : str, optional
        Output path for the scaler JSON. Defaults are set in DEFAULTS.

    Returns
    -------
    int
        0 on success. Also prints small diagnostics (variance contribution and
        correlations) to help verify parity between text and simulation signals.

    Output format
    -------------
    {
      "text": {"mean": <float>, "std": <float>},
      "sim":  {"mean": <float>, "std": <float>}
    }

    Notes
    -----
    - The produced scaler is consumed at label time by `fuse_equal_from_raw(...)` which
      standardizes text and sim inputs and fuses them 50/50.
    - Ensure that the `sim_var` you choose here matches the *raw* metric you will pass
      at runtime when labeling (e.g., HVAC electricity mean in J/hour).
    """
    concept = concept.lower()
    assert concept in (HVAC, INSUL), "concept must be 'hvac' or 'insulation'"

    # defaults if not provided
    sim_var = sim_var or DEFAULTS[concept]["sim_var"]
    sim_stat = sim_stat or DEFAULTS[concept]["sim_stat"]
    out = out or DEFAULTS[concept]["out"]

    root = Path(".")
    summary_path = root / "energyplus_data" / "summary_stats.json"
    meta_path = root / "energyplus_data" / "factorial_meta.json"

    summary = json.load(open(summary_path))
    meta = json.load(open(meta_path))

    homes = sorted(summary.keys())
    T_raw, S_raw = [], []

    for h in homes:
        m = meta[h]

        if concept == HVAC:
            text, idx = _compose_hvac_text(m)
        else:
            text, idx = _compose_insulation_text(m)

        # numeric text score (label-free ordinal)
        t_score = _ordinal_score_from_index(idx)

        # simulation scalar
        s_block = _resolve_sim_block(summary[h], sim_var)
        if s_block is None:
            keys_preview = list(summary[h].keys())[:8]
            raise KeyError(f"{sim_var} not in summary for {h}. Tried exact/stripped/case-insensitive/suffix matching. Example keys: {keys_preview}")
        if sim_stat not in s_block:
            raise KeyError(f"Statistic '{sim_stat}' not found for '{sim_var}'. Available: {list(s_block.keys())}")
        s_score = float(s_block[sim_stat])

        T_raw.append(t_score)
        S_raw.append(s_score)

    # fit z-score params per modality
    muT, sdT = _fit_mean_std(T_raw)
    muS, sdS = _fit_mean_std(S_raw)
    params = ScalerParams(text=ModStats(muT, sdT), sim=ModStats(muS, sdS))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(asdict(params), f, indent=2)

    # diagnostics (label-free)
    zT = (np.array(T_raw) - muT) / sdT
    zS = (np.array(S_raw) - muS) / sdS
    fused = 0.5 * zT + 0.5 * zS
    print(
        f"[{concept}] var(0.5*zT)={np.var(0.5 * zT):.3f}  "
        f"var(0.5*zS)={np.var(0.5 * zS):.3f}  "
        f"corr(fused,zT)={np.corrcoef(fused, zT)[0, 1]:.3f}  "
        f"corr(fused,zS)={np.corrcoef(fused, zS)[0, 1]:.3f}"
    )
    print(f"✅ saved {out}")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, choices=[HVAC, INSUL])
    ap.add_argument("--sim-var", help="Exact key from summary_stats.json", default=None)
    ap.add_argument("--sim-stat", choices=["mean", "min", "max", "std"], default=None)
    ap.add_argument("--out", help="Output scaler JSON path", default=None)
    a = ap.parse_args()
    raise SystemExit(build_concept_scaler(a.concept, a.sim_var, a.sim_stat, a.out))
