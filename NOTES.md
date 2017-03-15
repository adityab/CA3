# Asymmetric, power law STDP rule

## Stimulating each neuron

1. p = 0.01, a_* = 0.0
  - Weight distribution becomes somewhat bimodal with a nice spread
2. p = 0.01, a_* = 1.0
  - Weight distribution is a mixture of many very sharp modes
  
# Asymmetric, multiplicative STDP rule

## Stimulating each neuron

1. p = 0.01, a_* = 0.0
  - Weight distribution becomes somewhat bimodal with a nice spread
2. p = 0.01, a_* = 1.0
  - Weight distribution is a mixture of many very sharp modes
  
# Asymmetric, additive STDP rule

## Stimulating each neuron

1. p = 0.01, a_* = 0.0
  - Weight distribution becomes strongly bimodal (two extreme spikes), is unstable process
2. p = 0.01, a_* = 1.0
  - Weight distribution becomes strongly bimodal (two extreme spikes), is unstable process
  
# Duration of activity propagation

Motifs make activity last longer after stopping stimulation

Stimulate 100 neurons out of 4000
1. Poisson stop: 100ms
  - Activity only seen for a few neurons during a part of the first second
2. Poisson stop: 200ms
  - Activity goes on until 3 seconds, stabilizing around 1300 neurons, did not run longer

