# Data Validation

We show realism by comparing our generated homes to the range of data from climate zone 4A in the ResStock dataset. A description of this process follows.

## Data Validation Sequence

1. Acquire `baseline_metadata_only.csv` from ResStock release 2024.1, and place it in the `resstock` folder.
2. Execute `resstock/make_4a_saturations.py` to extract the metadata from climate zone 4A. This produces `resstock/options_saturations.csv`.
3. Run `resstock/summarize_options_saturations.py` to summarize the extracted data. This produces `resstock/options_saturations_summary.csv`.
4. Run `extract_distributions_dataset.py --root ../dataset` to pull the distributions from the synthetic homes. This produces `building_params.csv` and `building_params.json`.
5. Run `plot_realism_ranges.py`. This plots the entire range of the synthetic homes as a probability function (EDCF) for each varying value in question. It also runs KS tests. All output is in `realism_figures`.