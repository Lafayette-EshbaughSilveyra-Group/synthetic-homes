import pandas as pd
import numpy as np
import re
import math
from pathlib import Path

METADATA_PATH = Path("baseline_metadata_only.csv")
OUTPUT_PATH = Path("options_saturations.csv")


# ==== Load metadata, auto-detect delimiter ====

def load_metadata(path: Path) -> pd.DataFrame:
    # Try TSV first, then CSV if that fails.
    try:
        df = pd.read_csv(path, sep=",")
        if df.shape[1] == 1:
            raise ValueError("Looks like it's not TSV.")
        return df
    except Exception:
        df = pd.read_csv(path, sep="\t")
        return df


md = load_metadata(METADATA_PATH)

# ==== Identify key columns ====
CZ_COL = "in.ashrae_iecc_climate_zone_2004_2_a_split"
WT_COL = "weight"

for col in (CZ_COL, WT_COL):
    if col not in md.columns:
        raise ValueError(f"Expected column '{col}' not found in metadata.")

print(f"Loaded {len(md)} rows from metadata.")
print("Example climate zone values:")
print(md[CZ_COL].dropna().astype(str).str.upper().value_counts().head())

# ====  Filter to climate zone 4A ====

md_4a = md[md[CZ_COL].astype(str).str.upper().str.contains("4A")].copy()
if md_4a.empty:
    raise ValueError("No rows found for climate zone 4A. Check CZ_COL or values.")

print(f"Rows in climate zone 4A: {len(md_4a)}")


# ==== Define how to map each row -> your option label ====

def assign_option(row: pd.Series) -> str:
    """
    Define options based on a composite of roof insulation (with ceiling fallback),
    wall insulation, cooling COP, and heating COP, all derived from metadata fields.
    This matches the dimensions varied in the generated IDFs.

    - Roof insulation: Uses 'in.insulation_roof' (primary), falls back to
      'in.insulation_ceiling'. If both missing, "Rroof_unknown". Numeric values
      are rounded to nearest int, non-numeric are cleaned and labeled.
    - Wall insulation: Uses 'in.insulation_wall'. Same numeric/categorical logic,
      "Rwall_unknown" if missing.
    - Cooling COP: Uses 'in.hvac_cooling_efficiency'. If value >= 8, treated as
      SEER and converted to COP; else as COP. Missing → "COPc_unknown". Label is
      e.g., "COPc_3p5".
    - Heating COP: Uses 'in.hvac_heating_efficiency' and
      'in.hvac_heating_type_and_fuel'. For heat pump, value > 10 is HSPF,
      converted to COP; else as COP. For electric, COP=1.0. For fuel-fired,
      if value > 2, treat as AFUE percent; else as fraction. Missing → "COPh_unknown".
      Label is e.g., "COPh_0p92".
    - Returns composite string: "{roof_label}__{wall_label}__{cop_cool_label}__{cop_heat_label}"
    """

    # Helper for insulation label
    def insul_label(val, prefix, unknown_label):
        if pd.isna(val):
            return unknown_label
        try:
            num_val = float(val)
            label = str(int(round(num_val)))
        except (ValueError, TypeError):
            label = str(val).strip().replace(" ", "_")
        return f"{prefix}{label}"

    # Roof insulation (with ceiling fallback)
    roof_ins = row.get("in.insulation_roof")
    ceiling_ins = row.get("in.insulation_ceiling")
    val_roof = roof_ins if pd.notna(roof_ins) else ceiling_ins
    roof_label = insul_label(val_roof, "Rroof_", "Rroof_unknown")

    # Wall insulation
    wall_ins = row.get("in.insulation_wall")
    wall_label = insul_label(wall_ins, "Rwall_", "Rwall_unknown")

    # --- HVAC parsing helpers for ResStock-style strings ---

    def parse_cooling_cop(text):
        """
        Cooling efficiency strings example patterns:
          'AC, SEER 13'
          'Room AC, EER 10.7'
          'ASHP, SEER 13, 7.7 HSPF'
          'Ducted Heat Pump' (no number)
        Extract SEER/EER/COP if present and convert to COP.
        """
        if pd.isna(text):
            return np.nan
        s = str(text)

        m = re.search(r"SEER\s*([0-9]*\.?[0-9]+)", s, flags=re.I)
        if m:
            seer = float(m.group(1))
            return seer / 3.412

        m = re.search(r"EER\s*([0-9]*\.?[0-9]+)", s, flags=re.I)
        if m:
            eer = float(m.group(1))
            return eer / 3.412

        m = re.search(r"COP\s*([0-9]*\.?[0-9]+)", s, flags=re.I)
        if m:
            return float(m.group(1))

        return np.nan


    def parse_heating_cop(text, heat_type_str):
        """
        Heating efficiency strings example patterns:
          'Fuel Furnace, 80% AFUE'
          'Fuel Furnace, 92.5% AFUE'
          'ASHP, SEER 13, 7.7 HSPF'
        Extract HSPF/AFUE/COP if present and convert to COP-like number.
        """
        ht = (heat_type_str or "").lower()

        # Electric resistance-type systems
        if "electric" in ht and "ashp" not in ht and "heat pump" not in ht:
            return 1.0

        if pd.isna(text):
            return np.nan
        s = str(text)

        # Heat pumps: HSPF -> COP
        m = re.search(r"HSPF\s*([0-9]*\.?[0-9]+)", s, flags=re.I)
        if m:
            hspf = float(m.group(1))
            return hspf / 3.412

        # Furnaces/boilers: AFUE percent -> fraction (COP-like)
        m = re.search(r"([0-9]*\.?[0-9]+)\s*%?\s*AFUE", s, flags=re.I)
        if m:
            afue = float(m.group(1))
            if afue > 1.5:
                afue = afue / 100.0
            return afue

        # Direct COP if ever present
        m = re.search(r"COP\s*([0-9]*\.?[0-9]+)", s, flags=re.I)
        if m:
            return float(m.group(1))

        return np.nan


    # Cooling COP (robust parse from strings)
    cool_eff = row.get("in.hvac_cooling_efficiency")
    cop_cool = parse_cooling_cop(cool_eff)

    # fallback: sometimes SEER appears only in heating string for ASHP
    if math.isnan(cop_cool):
        cop_cool = parse_cooling_cop(row.get("in.hvac_heating_efficiency"))

    if math.isnan(cop_cool):
        cop_cool_label = "COPc_unknown"
    else:
        cop_cool = round(cop_cool, 1)
        cop_str = str(cop_cool).replace('.', 'p')
        cop_cool_label = f"COPc_{cop_str}"

    # Heating COP (robust parse from strings + heat type)
    heat_eff = row.get("in.hvac_heating_efficiency")
    heat_type = str(row.get("in.hvac_heating_type_and_fuel", "")).lower()
    cop_heat = parse_heating_cop(heat_eff, heat_type)

    if math.isnan(cop_heat):
        cop_heat_label = "COPh_unknown"
    else:
        cop_heat = round(cop_heat, 2)
        cop_str = str(cop_heat).replace('.', 'p')
        cop_heat_label = f"COPh_{cop_str}"

    return f"{roof_label}__{wall_label}__{cop_cool_label}__{cop_heat_label}"


# Apply the option mapping
md_4a["option"] = md_4a.apply(assign_option, axis=1)

# ==== Compute WEIGHTED saturations for 4A ====

total_weight_4a = md_4a[WT_COL].sum()

sats_4a = (
    md_4a
    .groupby("option", as_index=False)[WT_COL]
    .sum()
    .assign(saturation_4A=lambda df: df[WT_COL] / total_weight_4a)
    .drop(columns=[WT_COL])
    .sort_values("saturation_4A", ascending=False)
)

print("First few 4A options + saturations:")
print(sats_4a.head(10))

sats_4a.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved 4A saturations to: {OUTPUT_PATH}")
