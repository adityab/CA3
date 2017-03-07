import time
import numpy as np
import matplotlib.pyplot as plt

import nest

from graph import *

'''
Simulation params
'''
T = 10
T_eql = 10

'''
Network params
'''
N = 3000          # Number of neurons
p = 0.01          # ER-network connection probability
a_rec = 5.0       # Motif stats p^2 * (1 + a_)
a_conv = 5.0
a_div = 5.0
cc_chain = 0.0

CE = 1000          # Number of poisson neurons projecting in ("cortical" inputs)

'''
Synaptic params
'''
mu = 1.0
alpha = 1.085
lamb = 0.02
JE = .1          # Excitatory synaptic weights
delay = 1.0

tau_w = 2.0

'''
Neuron params
'''
tau_m = 20.0      # Membrane time constant
V_reset = 10.0
V_th = 20.0      # Depolarization threshold
C_m = 1.0

'''
Signal params
'''
eta = 2.0                         # Rate scaling factor
nu_th = V_th / (JE * CE * tau_m)  # Rate required to drive neuron to threshold asymptotically
nu_ex = eta * nu_th               # Poisson excitator rate
p_rate = 1000.0 * nu_ex * CE      # Population rate of the excitators

nest.ResetKernel()
#nest.SetKernelStatus({'print_time': True})

############## NETWORK ################

# Define custom synapse model
nest.SetDefaults("stdp_synapse",{"tau_plus": tau_w,
                                 "mu_plus":  mu,
                                 "mu_minus": mu,
                                 "alpha":    alpha,
                                 "lambda":   lamb,
                                 "Wmax":     2.0 * JE})

# Create population
population = nest.Create('iaf_psc_alpha', N, {"tau_minus": tau_w, "tau_m": tau_m, "V_th": V_th, "C_m": C_m, \
  "V_reset": V_reset
  })
print(nest.GetStatus([population[0]], ["V_reset", "V_th", "C_m", "tau_m", "t_ref", "tau_minus"]))

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
      syn_spec={
        'delay': 1.0,
        'model': 'stdp_synapse',
        'weight': {
          'distribution': 'uniform',
          'low': 0.5 * JE,
          'high': 1.5 * JE
      }})

print(col(BOLD, '* Done.'))

############## DEVICES ################

# One poisson generator with the full input population noise rate
noise = nest.Create('poisson_generator', 1, {'rate': p_rate})

sd = nest.Create("spike_detector")

subset = population[:100]
nest.Connect(noise, subset, syn_spec={'weight': 1.0})
nest.Connect(population, sd)

########### SIMULATE ###############

connections = nest.GetConnections(population, synapse_model='stdp_synapse')
hists = []

# Run simulation
for t in range(T):
  print(col(BOLD, '* Simulating second ' + str(t+1) ))
  nest.Simulate(1000.0)

  current_weights = np.array(nest.GetStatus(connections, 'weight'))
  #hist = np.histogram(current_weights, normed=True, bins=100)[0]
  #hist = hist.tolist()
  #hists.append(hist)
  #print(np.mean(current_weights), ',', np.std(current_weights))
  #print(current_weights[:10])
  dat = nest.GetStatus(sd, keys="events")[0]
  print(col(YELLOW, '%d Neurons spiked %d times') % (len(set(dat['senders'])), len(dat['times']) ))
  nest.SetStatus(sd, [{"n_events": 0}])

#hists = np.transpose(np.array(hists))
#plt.matshow(hists, cmap=plt.cm.gray_r)
#plt.show()
