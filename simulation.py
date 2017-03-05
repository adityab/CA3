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
net_params = {
  'n': 100,  # Number of neurons
  'p': 0.01   # ER-network connection probability
}

# Define synaptic model
J = 0.1
delay = 1.5


nest.ResetKernel()

############## NETWORK ################

# Define custom synapse model
nest.CopyModel('static_synapse','excitatory', {'weight': J, 'delay': delay})

# Create population
population = nest.Create('iaf_psc_alpha', net_params['n'])

# Create graph
W = Graph(net_params['n'], net_params['p'])

# Connect population ids according to generated graph
for node, neighbors in enumerate(W.adjacency_list()):
  if len(neighbors):
    idx = population[0]
    nest.Connect(
      [node + idx],
      [neighbor + idx for neighbor in neighbors],
      syn_spec='excitatory')

############## DEVICES ################

# Run simulation
nest.Simulate(simtime)
