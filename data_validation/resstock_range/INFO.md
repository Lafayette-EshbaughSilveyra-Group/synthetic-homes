# ResStock Range Computation

To determine if the generated data is realistic, we find ranges for the values of variables in our generated files from the ResStock data. We use the `baseline_metadata_only.csv` file from ResStock's 2024.1 release and compute an `options_saturations.csv` file for climate zone 4A. This is then compared to the distribution of our generated homes.