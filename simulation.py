import time
import numpy as np
import nest

from graph import *

'''
Simulation params
'''
simtime = 1200.0

'''
Network params
'''
N = 1000         # Number of neurons
p = 0.01        # ER-network connection probability
a_rec = 0.0     # Motif stats
a_conv = 0.0
a_div = 0.0
cc_chain = 0.0

# Define synaptic model
J = 0.1
delay = 1.5

nest.ResetKernel()

############## NETWORK ################

# Define custom synapse model
nest.CopyModel('static_synapse','excitatory', {'weight': J, 'delay': delay})

# Create population
population = nest.Create('iaf_psc_alpha', N)

# Create graph
print(col(BOLD, '* Generating SONET graph...'))
W = Graph(N, p, a_rec, a_conv, a_div, cc_chain)
print(col(BOLD, '* Done.'))

# Connect population ids according to generated graph
print(col(BOLD, '* Connecting neurons...'))

for node, neighbors in enumerate(W.adjacency_list()):
  if len(neighbors):
    idx = population[0]
    nest.Connect(
      [node + idx],
      [neighbor + idx for neighbor in neighbors],
      syn_spec='excitatory')

print(col(BOLD, '* Done.'))

############## DEVICES ################

# Run simulation
nest.Simulate(simtime)
