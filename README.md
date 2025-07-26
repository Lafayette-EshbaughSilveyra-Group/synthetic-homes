# Urban Energy Data Pipeline

## Overview
This project builds a synthetic dataset for urban energy analysis using publicly available assessor data, AI models (including OpenAI GPT and LLaVA), and simulation tools. The pipeline:

1. **Scrapes** property data and images from streets listed in `STREETS`.
2. **Generates** GeoJSON building footprints and inspection reports via OpenAI's API.
3. **Converts** GeoJSON to IDF format for EnergyPlus simulation.
4. **Runs** EnergyPlus simulations to produce energy performance outputs.
5. **Labels** the simulation outputs and inspection reports using OpenAI's API or heuristics.

Additionally, **occlusion experiments** analyze model sensitivity to localized image changes for robustness analysis.

---

## Pipeline Flow

```mermaid
flowchart TB
    A[Department of Assessment] -- tabular data, images --> B["GPT-4o"]
    B -- GeoJSON data --> C["GeoJSON to IDF"]
    C -- IDF EnergyPlus input --> D["EnergyPlus"]
    D -- "results.json" --> E["GPT-4o"] & F["X (model inputs)"]
    B -- home inspection report --> E & F
    E -- labels --> G["Y (ground truth)"]
```
_Figure 1_: Flow of data through the dataset generation pipeline.

---

## Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup environment variables
Copy `.env.example` to `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=your-api-key-here
```

### Run with interactive script (recommended)
```bash
chmod +x run.sh    # Only once after cloning
./run.sh           # Choose from pipeline, experiments, occlusion
```

Or, run manually:
```bash
python -m src.main --mode pipeline
```

## Requirements
- Python 3.x
- EnergyPlus (with `expandobjects` in `PATH`)
- ChromeDriver for Selenium
- OpenAI API key

---

## Outputs

| $X$ (Input Data) | $Y$ (Ground Truth) |
|------------------|--------------------|
| `dataset/*/results.json` — EnergyPlus simulation results | `dataset/*/label.json` — Data labels $\in \mathbb{R}^2$ |
| `dataset/*/cleaned.geojson["features"][0]["inspection_note"]` — synthetically generated inspection note | |
| `results/final_dataset.jsonl` — all inputs and outputs | `results/final_dataset_summary.csv` — summary statistics |

> **Note**: All outputs are compiled into `final_dataset.jsonl` and `final_dataset_summary.csv` inside the `results/` directory for the entire dataset.
