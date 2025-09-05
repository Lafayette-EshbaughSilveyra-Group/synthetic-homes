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

## Publication

TODO: Update once the preprint is out

---

## Pipeline Flow

```mermaid
graph TD
  A["County Scraping"] --Home Image--> B["LLaVA 1"]
  A --Home Floor Plan--> C["LLaVA 2"]
  B --Home Image Description--> D["GPT-4-mini"]
  C --Home Floor Plan Description--> D
  A --Home Data--> D
  D --GeoJSON--> E[EnergyPlus]
  D --Home Inspection Notes--> F["GPT-4-mini"]
  E --Simulation Results--> G["Heuristic Labeler (Eq. 1)"]
  G --> H["Weighted Sum (Eq. 2)"]
  F --> H
  H --> I["Results"] 
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


## Experiments

### Occlusion & Reverse Occlusion

Occlusion and reverse occlusion tests are used to evaluate how models process images. Specifically, occlusion measures _necessity_—how important a region of an image is to produce the output—by masking each region of the input and comparing the model's output on this masked image to the output from the unmodified image. This produces a heatmap.

Reverse occlusion (sometimes referred to as inclusion) measures _sufficiency_—if a feature or subset of the input alone leads the model to make the same prediction, then that feature is sufficient for the model’s decision. This testing is performed by masking everything except a given portion of the image and comparing to a baseline, like the above.

### Ablation

In ablation testing, we try to determine importance of each modality that is passed into the model. This is important as the embedders for each type of modality are unique and they need to be balanced with respect to each other. As such, we:

1. Test only text input sensitivity (constant simulation input)
2. Test only simulation input sensitivity (constant text input)
3. Test both inputs together, in a bad/bad, good/bad, bad/good, good/good arrangement.

More details can be found in `experiments.md`.