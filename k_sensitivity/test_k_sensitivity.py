

#!/usr/bin/env python3
"""
Calibration-resolution sensitivity experiment.

This script evaluates how sensitive final supervision scores are to the
resolution of the synthetic ordinal calibration grid. It does not modify the
source dataset. It reads calibration artifacts and target examples, recomputes
calibration statistics for k={2,3,5}, relabels the same target examples under
each calibration resolution, and writes CSV/JSON outputs to the working
directory.

Expected project layout, relative to this file:
    ../dataset/                         target examples; read-only
    energyplus_data/factorial_meta.json calibration metadata
    energyplus_data/summary_stats.json  calibration simulation summaries

The script is intentionally defensive about JSON schemas because the dataset
has evolved over time. It attempts to extract existing raw text scores and raw
simulation features from per-example JSON files. If it cannot find target rows
in ../dataset, it falls back to using the calibration grid itself as the target
set, which still tests the effect of changing calibration resolution.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DATASET_DIR = Path("../dataset")
OUTPUT_DIR = Path("k_sensitivity_results")

# Existing five-rung ordinal calibration grid. Smaller k values are derived by
# subselecting these same rungs, so no new simulations are required.
LEVEL_SETS: Dict[str, Sequence[int]] = {
    "k2": (1, 5),
    "k3": (1, 3, 5),
    "k5": (1, 2, 3, 4, 5),
}

PARAM_LEVELS: Dict[str, Sequence[float]] = {
    "wall_r_value": (4.0, 7.0, 13.0, 20.0, 30.0),
    "roof_r_value": (10.0, 20.0, 30.0, 40.0, 50.0),
    "hvac_heating_cop": (0.7, 0.8, 0.9, 0.95, 1.0),
    "hvac_cooling_cop": (1.0, 2.0, 3.0, 3.5, 4.0),
}

HVAC_SIM_FEATURE_CANDIDATES = (
    "Electricity:HVAC [J](Hourly)",
    "hvac_electricity",
    "hvac",
    "hvac_energy",
)

INSULATION_SIM_FEATURE_CANDIDATES = (
    "Heating Coil Heating Energy [J](Hourly)",
    "heating_coil",
    "heating_coil_energy",
    "insulation",
)


@dataclass(frozen=True)
class RawExample:
    """Raw target example before calibration."""

    example_id: str
    text_hvac: float
    text_insulation: float
    sim_hvac: float
    sim_insulation: float


@dataclass(frozen=True)
class Scaler:
    """Calibration statistics for one concept."""

    text_mean: float
    text_std: float
    sim_mean: float
    sim_std: float


@dataclass(frozen=True)
class LabeledExample:
    """Example after calibration and fusion."""

    example_id: str
    k_name: str
    hvac_label: float
    insulation_label: float
    overall_label: float


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def find_project_file(relative_path: str) -> Path:
    """Find a project artifact from likely working-directory locations."""
    candidates = [
        Path(relative_path),
        Path("..") / relative_path,
        Path("../energyplus_data") / Path(relative_path).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {relative_path!r}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def flatten_records(value: Any) -> List[Dict[str, Any]]:
    """Convert common JSON shapes into a list of dictionaries."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]

    if isinstance(value, dict):
        for key in ("records", "examples", "data", "results", "homes"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]

        # Common shape: {example_id: { ... }}
        if all(isinstance(item, dict) for item in value.values()):
            records = []
            for key, item in value.items():
                record = dict(item)
                record.setdefault("example_id", key)
                record.setdefault("id", key)
                records.append(record)
            return records

    return []


def get_identifier(record: Dict[str, Any]) -> Optional[str]:
    for key in ("example_id", "id", "home_id", "name", "folder", "address_folder"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def get_nested(record: Dict[str, Any], path: Sequence[str]) -> Optional[Any]:
    current: Any = record
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None



def nearest_level(parameter_name: str, value: Any) -> Optional[int]:
    numeric_value = as_float(value)
    if numeric_value is None:
        return None

    levels = PARAM_LEVELS[parameter_name]
    index = min(range(len(levels)), key=lambda i: abs(levels[i] - numeric_value))
    return index + 1


def extract_calibration_levels(record: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Map raw factorial parameter values to one-based ordinal levels 1..5."""
    levels = {
        "wall": nearest_level("wall_r_value", record.get("wall_r_value")),
        "roof": nearest_level("roof_r_value", record.get("roof_r_value")),
        "heating": nearest_level("hvac_heating_cop", record.get("hvac_heating_cop")),
        "cooling": nearest_level("hvac_cooling_cop", record.get("hvac_cooling_cop")),
    }
    if any(value is None for value in levels.values()):
        return None
    return {key: int(value) for key, value in levels.items() if value is not None}

def resolve_sim_block(record: Dict[str, Any], sim_var: str) -> Optional[Any]:
    if sim_var in record:
        return record[sim_var]

    wanted = sim_var.strip()

    by_stripped = {str(k).strip(): v for k, v in record.items()}
    if wanted in by_stripped:
        return by_stripped[wanted]

    by_lower = {str(k).strip().lower(): v for k, v in record.items()}
    if wanted.lower() in by_lower:
        return by_lower[wanted.lower()]

    matches = [
        (str(k), v)
        for k, v in record.items()
        if str(k).strip().endswith(wanted)
    ]
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        return sorted(matches, key=lambda item: len(item[0]))[0][1]

    return None

def extract_stat_value(record: Dict[str, Any], feature_candidates: Sequence[str]) -> Optional[float]:
    stat_keys = ("mean", "average", "avg", "value", "min", "max", "std")

    for candidate in feature_candidates:
        block = resolve_sim_block(record, candidate)
        if isinstance(block, dict):
            for stat_key in stat_keys:
                numeric_value = as_float(block.get(stat_key))
                if numeric_value is not None:
                    return numeric_value
        else:
            numeric_value = as_float(block)
            if numeric_value is not None:
                return numeric_value

    for container_key in ("features", "summary", "summary_stats", "stats", "outputs"):
        container = record.get(container_key)
        if not isinstance(container, dict):
            continue

        for candidate in feature_candidates:
            block = resolve_sim_block(container, candidate)
            if isinstance(block, dict):
                for stat_key in stat_keys:
                    numeric_value = as_float(block.get(stat_key))
                    if numeric_value is not None:
                        return numeric_value
            else:
                numeric_value = as_float(block)
                if numeric_value is not None:
                    return numeric_value

    return None


def load_calibration_records() -> List[RawExample]:
    """Load the 5^4 factorial calibration grid as raw examples."""
    meta_path = find_project_file("energyplus_data/factorial_meta.json")
    stats_path = find_project_file("energyplus_data/summary_stats.json")

    meta_records = flatten_records(read_json(meta_path))
    stat_records = flatten_records(read_json(stats_path))

    stats_by_id: Dict[str, Dict[str, Any]] = {}
    for record in stat_records:
        identifier = get_identifier(record)
        if identifier is not None:
            stats_by_id[identifier] = record

    calibration_rows: List[RawExample] = []
    skipped = 0
    for meta_record in meta_records:
        identifier = get_identifier(meta_record)
        if identifier is None:
            skipped += 1
            continue

        levels = extract_calibration_levels(meta_record)
        stats_record = stats_by_id.get(identifier, {})
        merged = {**stats_record, **meta_record}

        sim_hvac = extract_stat_value(merged, HVAC_SIM_FEATURE_CANDIDATES)
        sim_insulation = extract_stat_value(merged, INSULATION_SIM_FEATURE_CANDIDATES)

        if levels is None or sim_hvac is None or sim_insulation is None:
            skipped += 1
            continue

        # Ordinal text calibration scores are derived from the same template rungs
        # described in the appendix, not from model-generated text labels.
        text_hvac = statistics.mean((levels["heating"], levels["cooling"]))
        text_insulation = statistics.mean((levels["wall"], levels["roof"]))

        calibration_rows.append(
            RawExample(
                example_id=identifier,
                text_hvac=float(text_hvac),
                text_insulation=float(text_insulation),
                sim_hvac=float(sim_hvac),
                sim_insulation=float(sim_insulation),
            )
        )

    if not calibration_rows:
        raise RuntimeError(
            "No calibration rows could be loaded. Check factorial_meta.json and summary_stats.json schemas."
        )

    print(f"Loaded {len(calibration_rows)} calibration rows; skipped {skipped} rows.")
    return calibration_rows


def include_calibration_row(example: RawExample, allowed_levels: Sequence[int]) -> bool:
    """Infer inclusion from the ordinal text scores.

    The calibration text scores are averages of two integer levels. To avoid losing
    mixed pairs such as (1, 5) whose average is 3, this function is only used when
    calibration rows are loaded without explicit levels. In the current script we
    refilter using metadata below, so this function is kept only as a safe fallback.
    """
    allowed = set(allowed_levels)
    return int(round(example.text_hvac)) in allowed and int(round(example.text_insulation)) in allowed


def load_calibration_rows_by_k() -> Dict[str, List[RawExample]]:
    """Load calibration rows and filter each k using explicit factorial metadata."""
    meta_path = find_project_file("energyplus_data/factorial_meta.json")
    stats_path = find_project_file("energyplus_data/summary_stats.json")

    meta_records = flatten_records(read_json(meta_path))
    stat_records = flatten_records(read_json(stats_path))
    stats_by_id = {get_identifier(record): record for record in stat_records if get_identifier(record) is not None}

    rows_by_k: Dict[str, List[RawExample]] = {name: [] for name in LEVEL_SETS}
    skipped = 0

    for meta_record in meta_records:
        identifier = get_identifier(meta_record)
        levels = extract_calibration_levels(meta_record)
        if identifier is None or levels is None:
            skipped += 1
            continue

        stats_record = stats_by_id.get(identifier, {})
        merged = {**stats_record, **meta_record}
        sim_hvac = extract_stat_value(merged, HVAC_SIM_FEATURE_CANDIDATES)
        sim_insulation = extract_stat_value(merged, INSULATION_SIM_FEATURE_CANDIDATES)
        if sim_hvac is None or sim_insulation is None:
            skipped += 1
            continue

        row = RawExample(
            example_id=identifier,
            text_hvac=float(statistics.mean((levels["heating"], levels["cooling"]))),
            text_insulation=float(statistics.mean((levels["wall"], levels["roof"]))),
            sim_hvac=float(sim_hvac),
            sim_insulation=float(sim_insulation),
        )

        for k_name, allowed_levels in LEVEL_SETS.items():
            allowed = set(allowed_levels)
            if all(level in allowed for level in levels.values()):
                rows_by_k[k_name].append(row)

    for k_name, rows in rows_by_k.items():
        if not rows:
            print(
                f"No calibration rows found for {k_name}. This usually means raw factorial "
                "parameter values were not mapped to ordinal levels correctly. Check that "
                "PARAM_LEVELS matches factorial_meta.json."
            )
            raise RuntimeError(f"No calibration rows found for {k_name}.")
        example_ids_preview = ", ".join(row.example_id for row in rows[:3])
        print(f"{k_name}: example preview: {example_ids_preview}")
        expected = len(LEVEL_SETS[k_name]) ** 4
        print(f"{k_name}: loaded {len(rows)} calibration rows; expected approximately {expected}.")

    if skipped:
        print(f"Skipped {skipped} calibration rows while filtering by k.")


    return rows_by_k


def safe_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1.0
    std = statistics.pstdev(values)
    if math.isclose(std, 0.0, abs_tol=1e-12):
        return 1.0
    return std


def build_scaler(rows: Sequence[RawExample], concept: str) -> Scaler:
    if concept == "hvac":
        text_values = [row.text_hvac for row in rows]
        sim_values = [row.sim_hvac for row in rows]
    elif concept == "insulation":
        text_values = [row.text_insulation for row in rows]
        sim_values = [row.sim_insulation for row in rows]
    else:
        raise ValueError(f"Unknown concept: {concept}")

    return Scaler(
        text_mean=statistics.mean(text_values),
        text_std=safe_std(text_values),
        sim_mean=statistics.mean(sim_values),
        sim_std=safe_std(sim_values),
    )


def sigmoid(value: float) -> float:
    value = max(-3.0, min(3.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def fuse(text_value: float, sim_value: float, scaler: Scaler) -> float:
    text_z = (text_value - scaler.text_mean) / scaler.text_std
    sim_z = (sim_value - scaler.sim_mean) / scaler.sim_std
    return sigmoid(0.5 * text_z + 0.5 * sim_z)


def extract_text_score(record: Dict[str, Any], concept: str) -> Optional[float]:
    candidate_paths = [
        ("text_scores", concept),
        ("gpt_scores", concept),
        ("llm_scores", concept),
        ("raw_scores", f"text_{concept}"),
        ("raw", f"text_{concept}"),
        ("scores", concept),
        (concept, "text_score"),
        (concept, "score"),
    ]
    candidate_keys = (
        f"text_{concept}",
        f"{concept}_text",
        f"{concept}_text_score",
        f"mean_{concept}",
    )

    for key in candidate_keys:
        value = as_float(record.get(key))
        if value is not None:
            return value

    for path in candidate_paths:
        value = as_float(get_nested(record, path))
        if value is not None:
            return value

    return None


def extract_sim_score(record: Dict[str, Any], concept: str) -> Optional[float]:
    if concept == "hvac":
        candidates = HVAC_SIM_FEATURE_CANDIDATES
        candidate_keys = ("sim_hvac", "hvac_sim", "hvac_electricity", "mean_hvac_sim")
    else:
        candidates = INSULATION_SIM_FEATURE_CANDIDATES
        candidate_keys = ("sim_insulation", "insulation_sim", "heating_coil", "mean_insulation_sim")

    for key in candidate_keys:
        value = as_float(record.get(key))
        if value is not None:
            return value

    value = extract_stat_value(record, candidates)
    if value is not None:
        return value

    candidate_paths = [
        ("raw_scores", f"sim_{concept}"),
        ("raw", f"sim_{concept}"),
        ("simulation_scores", concept),
        ("simulation", concept),
    ]
    for path in candidate_paths:
        value = as_float(get_nested(record, path))
        if value is not None:
            return value

    return None


def load_target_examples_from_dataset(dataset_dir: Path) -> List[RawExample]:
    """Extract raw scoring inputs from ../dataset without modifying it."""
    if not dataset_dir.exists():
        print(f"Dataset directory {dataset_dir} does not exist.")
        return []

    examples: List[RawExample] = []
    seen: set[str] = set()

    for path in sorted(dataset_dir.rglob("*.json")):
        try:
            raw_json = read_json(path)
        except (json.JSONDecodeError, OSError):
            continue

        records = flatten_records(raw_json)
        if not records and isinstance(raw_json, dict):
            records = [raw_json]

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue

            identifier = get_identifier(record) or f"{path.parent.name}:{path.name}:{index}"
            text_hvac = extract_text_score(record, "hvac")
            text_insulation = extract_text_score(record, "insulation")
            sim_hvac = extract_sim_score(record, "hvac")
            sim_insulation = extract_sim_score(record, "insulation")

            if None in (text_hvac, text_insulation, sim_hvac, sim_insulation):
                continue

            if identifier in seen:
                continue
            seen.add(identifier)

            examples.append(
                RawExample(
                    example_id=identifier,
                    text_hvac=float(text_hvac),
                    text_insulation=float(text_insulation),
                    sim_hvac=float(sim_hvac),
                    sim_insulation=float(sim_insulation),
                )
            )

    print(f"Loaded {len(examples)} target examples from {dataset_dir}.")
    return examples


def label_examples(
    examples: Sequence[RawExample],
    k_name: str,
    hvac_scaler: Scaler,
    insulation_scaler: Scaler,
) -> List[LabeledExample]:
    labeled: List[LabeledExample] = []
    for example in examples:
        hvac_label = fuse(example.text_hvac, example.sim_hvac, hvac_scaler)
        insulation_label = fuse(example.text_insulation, example.sim_insulation, insulation_scaler)
        overall_label = statistics.mean((hvac_label, insulation_label))
        labeled.append(
            LabeledExample(
                example_id=example.example_id,
                k_name=k_name,
                hvac_label=hvac_label,
                insulation_label=insulation_label,
                overall_label=overall_label,
            )
        )
    return labeled


def rankdata(values: Sequence[float]) -> List[float]:
    """Average ranks for ties, 1-indexed."""
    sorted_pairs = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        average_rank = (i + 1 + j + 1) / 2.0
        for idx in range(i, j + 1):
            ranks[sorted_pairs[idx][0]] = average_rank
        i = j + 1
    return ranks


def pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return float("nan")
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_denominator = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denominator = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    denominator = x_denominator * y_denominator
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return float("nan")
    return numerator / denominator


def spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    return pearson(rankdata(x_values), rankdata(y_values))


def top_fraction_overlap(
    candidate_scores: Dict[str, float],
    reference_scores: Dict[str, float],
    fraction: float = 0.10,
) -> float:
    common_ids = sorted(set(candidate_scores) & set(reference_scores))
    if not common_ids:
        return float("nan")
    top_n = max(1, math.ceil(len(common_ids) * fraction))

    candidate_top = {
        example_id
        for example_id, _ in sorted(
            ((example_id, candidate_scores[example_id]) for example_id in common_ids),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_n]
    }
    reference_top = {
        example_id
        for example_id, _ in sorted(
            ((example_id, reference_scores[example_id]) for example_id in common_ids),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_n]
    }
    return len(candidate_top & reference_top) / top_n


def summarize_against_reference(
    labels_by_k: Dict[str, List[LabeledExample]], reference_k: str = "k5"
) -> List[Dict[str, Any]]:
    reference = {row.example_id: row.overall_label for row in labels_by_k[reference_k]}
    summaries: List[Dict[str, Any]] = []

    for k_name, rows in labels_by_k.items():
        candidate = {row.example_id: row.overall_label for row in rows}
        common_ids = sorted(set(candidate) & set(reference))
        candidate_values = [candidate[example_id] for example_id in common_ids]
        reference_values = [reference[example_id] for example_id in common_ids]

        if common_ids:
            mean_abs_diff = statistics.mean(
                abs(candidate[example_id] - reference[example_id]) for example_id in common_ids
            )
            max_abs_diff = max(abs(candidate[example_id] - reference[example_id]) for example_id in common_ids)
        else:
            mean_abs_diff = float("nan")
            max_abs_diff = float("nan")

        summaries.append(
            {
                "calibration_resolution": k_name,
                "num_examples": len(common_ids),
                "spearman_vs_k5": spearman(candidate_values, reference_values),
                "mean_abs_diff_vs_k5": mean_abs_diff,
                "max_abs_diff_vs_k5": max_abs_diff,
                "top_10_percent_overlap_vs_k5": top_fraction_overlap(candidate, reference, fraction=0.10),
                "top_20_percent_overlap_vs_k5": top_fraction_overlap(candidate, reference, fraction=0.20),
            }
        )

    return sorted(summaries, key=lambda item: item["calibration_resolution"])


def write_labeled_csv(path: Path, labels_by_k: Dict[str, List[LabeledExample]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("example_id", "calibration_resolution", "hvac_label", "insulation_label", "overall_label"),
        )
        writer.writeheader()
        for k_name in sorted(labels_by_k):
            for row in labels_by_k[k_name]:
                writer.writerow(
                    {
                        "example_id": row.example_id,
                        "calibration_resolution": row.k_name,
                        "hvac_label": row.hvac_label,
                        "insulation_label": row.insulation_label,
                        "overall_label": row.overall_label,
                    }
                )


def write_summary_csv(path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "calibration_resolution",
                "num_examples",
                "spearman_vs_k5",
                "mean_abs_diff_vs_k5",
                "max_abs_diff_vs_k5",
                "top_10_percent_overlap_vs_k5",
                "top_20_percent_overlap_vs_k5",
            ),
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_by_k = load_calibration_rows_by_k()
    scalers: Dict[str, Dict[str, Scaler]] = {}
    for k_name, rows in rows_by_k.items():
        scalers[k_name] = {
            "hvac": build_scaler(rows, "hvac"),
            "insulation": build_scaler(rows, "insulation"),
        }

    target_examples = load_target_examples_from_dataset(DATASET_DIR)
    target_source = "dataset"
    if not target_examples:
        print(
            "No reusable raw target examples were found in ../dataset. "
            "Falling back to the full calibration grid as the target set."
        )
        target_examples = load_calibration_records()
        target_source = "calibration_grid_fallback"

    labels_by_k: Dict[str, List[LabeledExample]] = {}
    for k_name in sorted(LEVEL_SETS):
        labels_by_k[k_name] = label_examples(
            target_examples,
            k_name,
            scalers[k_name]["hvac"],
            scalers[k_name]["insulation"],
        )

    summary_rows = summarize_against_reference(labels_by_k, reference_k="k5")

    write_summary_csv(OUTPUT_DIR / "k_sensitivity_summary.csv", summary_rows)
    write_labeled_csv(OUTPUT_DIR / "k_sensitivity_labels.csv", labels_by_k)
    write_json(
        OUTPUT_DIR / "k_sensitivity_summary.json",
        {
            "target_source": target_source,
            "num_target_examples": len(target_examples),
            "level_sets": {key: list(value) for key, value in LEVEL_SETS.items()},
            "scalers": {
                k_name: {
                    concept: scaler.__dict__ for concept, scaler in concept_scalers.items()
                }
                for k_name, concept_scalers in scalers.items()
            },
            "summary": summary_rows,
        },
    )

    print("\nCalibration-resolution sensitivity summary:")
    for row in summary_rows:
        print(
            f"{row['calibration_resolution']}: "
            f"n={row['num_examples']}, "
            f"spearman_vs_k5={row['spearman_vs_k5']:.4f}, "
            f"mean_abs_diff={row['mean_abs_diff_vs_k5']:.4f}, "
            f"top10_overlap={row['top_10_percent_overlap_vs_k5']:.4f}"
        )

    print(f"\nWrote results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()