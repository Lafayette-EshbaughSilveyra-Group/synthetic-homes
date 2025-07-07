## Experimental Run | July 3, 2025 | ID `070325-01`

## Notes

GPT continues to demonstrate ineffectiveness in understanding the E+ results. Quick and dirty estimate: SD among all of experiment 2 (E+) is about 0.001 (for each vector element). These results speak in tandem with `070225-01` and `070225-02` and suggest a deeper need for an unsupervised model to do the labeling based on the data as opposed to GPT. GPT remains useful in the realm of textual processing—it is invaluable there because that is what it is trained to do.

## Updates & Changes

This suggests that the completely seperated GPT calls were not enough—we'll need to bring in some other model. A desirable model is:

- easy to setup & use
- perhaps made with real-estate data in mind
- unsupervised

More work will need to be done to locate a feasible solution.