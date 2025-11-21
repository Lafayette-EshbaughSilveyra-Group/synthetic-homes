import pandas as pd
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

    # Cooling COP
    cool_eff = row.get("in.hvac_cooling_efficiency")
    if pd.isna(cool_eff):
        cop_cool_label = "COPc_unknown"
    else:
        try:
            val = float(cool_eff)
            if val >= 8:
                cop = val / 3.412  # SEER to COP
            else:
                cop = val
            cop = round(cop, 1)
            cop_str = str(cop).replace('.', 'p')
            cop_cool_label = f"COPc_{cop_str}"
        except Exception:
            cop_cool_label = "COPc_unknown"

    # Heating COP
    heat_eff = row.get("in.hvac_heating_efficiency")
    heat_type = str(row.get("in.hvac_heating_type_and_fuel", "")).lower()
    if pd.isna(heat_eff):
        cop_heat_label = "COPh_unknown"
    else:
        try:
            val = float(heat_eff)
            if "heat pump" in heat_type:
                if val > 10:
                    cop = val / 3.412  # HSPF to COP
                else:
                    cop = val
            elif "electric" in heat_type:
                cop = 1.0
            else:
                # fuel-fired
                if val > 2:
                    eff_frac = val / 100.0
                else:
                    eff_frac = val
                cop = eff_frac
            cop = round(cop, 2)
            cop_str = str(cop).replace('.', 'p')
            cop_heat_label = f"COPh_{cop_str}"
        except Exception:
            cop_heat_label = "COPh_unknown"

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
