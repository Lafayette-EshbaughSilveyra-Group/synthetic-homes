# Experimental Run | July 4, 2025 | ID `070425-02`

## Notes

- Shares identical notes with `070425-01`

## Observations

Remarkably, the variation across the board in the second experiment is low, even after implementing heuristics-based labeling. I noticed that the new heuristics were likely not being applied, as the final results were around 0.25—suppose GPT gives 0.50 for the neutral text. 0.5 + 0 / 2 = 0.25, our result.

Looking at the outputs of the E+ simulation generation script, one home reported using only 51 kWh per year, which is incredibly unrealistic. 

## Reflection

I do not believe these results correspond to the potential of the heuristics-based labeling system. Instead, I believe the simulations themselves are flawed. Consider the 51 kWh / yr home. It is expected to return 0 from the labeler at current (it is outside the range made by the min and max values). In fact, the `idf` lacks an infiltration simulation, meaning air loss and/or temperature change is not modeled at present. All in all, we need to get infiltration into the IDF. 

## Changes Made

- Added Infiltration object to the generated IDFs.
- Adjusted the max/min values for the labeler to fit the range we have.
- Changed weather data to KMSP (Minneapolis / St. Paul) for a colder winter.