from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from src.experiments.constants import HVAC_TEXT_SAMPLES, INSULATION_TEXT_SAMPLES

# ---- concept enum ----
HVAC = "hvac"
INSUL = "insulation"

# ---- default sim vars (override with CLI flags if your column names differ) ----
DEFAULTS = {
    HVAC:  {"sim_var": "Electricity:HVAC [J](Hourly)", "sim_stat": "mean", "out": "energyplus_data/hvac_scaler_params.json"},
    INSUL: {"sim_var": "Heating Coil Heating Energy [J](Hourly)", "sim_stat": "mean", "out": "energyplus_data/insulation_scaler_params.json"},
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

# ---- utils ----
def _fit_mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    m  = float(xs.mean())
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
    concept = concept.lower()
    assert concept in (HVAC, INSUL), "concept must be 'hvac' or 'insulation'"

    # defaults if not provided
    sim_var  = sim_var  or DEFAULTS[concept]["sim_var"]
    sim_stat = sim_stat or DEFAULTS[concept]["sim_stat"]
    out      = out      or DEFAULTS[concept]["out"]

    root = Path(".")
    summary_path = root / "energyplus_data" / "summary_stats.json"
    meta_path    = root / "energyplus_data" / "factorial_meta.json"

    summary = json.load(open(summary_path))
    meta    = json.load(open(meta_path))

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
        s_block = summary[h].get(sim_var)
        if s_block is None:
            # help debug by surfacing available keys
            keys_preview = list(summary[h].keys())[:8]
            raise KeyError(f"{sim_var} not in summary for {h}. Example keys: {keys_preview}")
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
    fused = 0.5*zT + 0.5*zS
    print(
        f"[{concept}] var(0.5*zT)={np.var(0.5*zT):.3f}  "
        f"var(0.5*zS)={np.var(0.5*zS):.3f}  "
        f"corr(fused,zT)={np.corrcoef(fused,zT)[0,1]:.3f}  "
        f"corr(fused,zS)={np.corrcoef(fused,zS)[0,1]:.3f}"
    )
    print(f"✅ saved {out}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, choices=[HVAC, INSUL])
    ap.add_argument("--sim-var", help="Exact key from summary_stats.json", default=None)
    ap.add_argument("--sim-stat", choices=["mean","min","max","std"], default=None)
    ap.add_argument("--out", help="Output scaler JSON path", default=None)
    a = ap.parse_args()
    raise SystemExit(build_concept_scaler(a.concept, a.sim_var, a.sim_stat, a.out))