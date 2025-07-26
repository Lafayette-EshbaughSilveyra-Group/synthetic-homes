# Experimental Run | July 05, 2025 | ID `070525-01`

## Notes

- Implemented weighted sum for labeler (E+ @ 0.7; text @ 0.3)

## Observations

All in all, I observed real positive change from the previous experiments. Notably:

- In Experiment 2, the trend is now much more visible.
- In Experiment 3, I observed real positive change: the `good_text_poor_simulation` is greater than `good_text_good_simulation`. This trend is the same for `poor_text_good_simulation`; it is significantly smaller than `poor_text_poor_simulation`. Ideally, a good simulation and poor text should produce roughly the same label as a poor simulation and good text. I observe a significant difference between those two variables (since the labeler still prefers text), so some work still can be done to improve it. Specifically, I suspect that tweaking the weights in the weighted sum should alleviate this issue.

## Tweaks

A future experimental run will test using different weights. [ID: todo]