# Urban Energy Data Pipeline

## Details

1. Scrapes the streets in STREETS, fetching images and tabular data,
2. Passes that information into the OpenAI API to generate GeoJSON and an inspection report,
3. Converts the GeoJSON to an IDF format for use in EnergyPlus,
4. Runs EnergyPlus on the IDF,
5. Classifies the EnergyPlus results and inspection report, giving the ground truth using OpenAI's API

OUTPUTS:
`dataset/{entry_folder}/results.json`           ⌉

                                                | INPUTS (i.e., X)

`dataset/{entry_folder}/inspection_report.txt`  ⌋

`dataset/{entry_folder}/label.json`             ] GROUND TRUTH (i.e., Y)

> **Note**: All of these are compiled for the entire dataset into `final_dataset.jsonl`

```mermaid
flowchart TB
    A[Department of Assessment] -- tablular data, images --> B["GPT-4o"]
    B -- GeoJSON data --> C["GeoJSON to IDF"]
    C -- IDF EnergyPlus input --> D["EnergyPlus"]
    D -- "results.json" --> E["GPT-4o"] & F["X (model inputs)"]
    B -- home inspection report --> E & F
    E -- labels --> G["Y (ground truth)"]
```
_Figure 1_: The flow of data through the dataset generation pipeline
