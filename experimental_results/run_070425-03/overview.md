# Experimental Run | July 4, 2025 | ID `070425-03`

## Observations

I observed much more variation in the heuristic labeling, but text still seems to weigh more.

## Reflection

This makes me wonder: would GPT work with these much more realistic values? It's easy to force the E+ labels to matter more (weigh them manually, something like 0.7 E+, 0.3 text). This gives two investigations:

1. GPT with these new `idf` files [ID: `070525-02` ]
   - Does it seem to work well?
   - Would weighing it help?
2. Simply weigh these scores (E+ : 0.7; text : 0.3) [ID: `070525-01`]