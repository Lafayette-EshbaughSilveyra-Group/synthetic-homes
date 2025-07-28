# Experimental Design

## Objective
We hope to evaluate the performance of the pipeline's labeling module. Specifically:

1. Directional Responsiveness—Do we see change in the expected direction?
2. Consistency—How stable are the outputs across multiple runs of the same input?

## Procedure
1. Run the labeling module 5 times to account for GPT output stochasticity. Record these results.
2. Compute 
   - mean label value
   - standard deviation
3. Analyze trends across inputs
   - Do inputs with more sever problems produce higher recommendation scores? (**obj. 1**)
   - Are outputs stable? (**obj. 2**)

## Experiments

### 1. Text Variation Responsiveness
**Objective**: Determine whether inspection report severity affects recommendations

#### Inputs
- synthetic inspection reports describing worsening conditions
- a constant E+ output

#### Hypotheses

- mean recommendation displays a proportional relationship with the severity of the conditions (as conditions get worse, recommendations increase).
- a stable output (a relatively small standard deviation)

### 2. E+ Variation Responsiveness
**Objective**: Determine whether E+ output differences affect recommendations

#### Inputs

- some neutral inspection report text
- E+ outputs: simulations with increasing inefficiency

#### Hypotheses

- mean recommendation displays a proportional relationship with the severity of the conditions (as conditions get worse, recommendations increase).
- a stable output (a relatively small standard deviation)

### 3. Combined Inputs Evaluation
**Objective**: observe pipeline behavior with realistic combined inputs

#### Inputs

- paired input text and E+ outputs
  - good text, good sim
  - good text, bad sim
  - bad text, good sim
  - bad text, bad sim

#### Hypotheses

- Results match (i.e., good / good => low recommendation; bad / bad => high recommendation)