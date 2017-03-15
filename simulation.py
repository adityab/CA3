import time
import numpy as np
import matplotlib.pyplot as plt

import nest

from graph import *

'''
Simulation params
'''
T = 20
T_poisson = .2
synapse_type='stdp_synapse'

LOG_ACTIVITY=True
PLOT_WEIGHTS=True
WEIGHT_EVOL=False

'''
Network params
'''
N = 5000         # Number of neurons
p = 0.01          # ER-network connection probability
a_rec = 5.0       # Motif stats p^2 * (1 + a_)
a_conv = 5.0
a_div = 5.0
cc_chain = 1.0

CE = 1000         # Number of poisson neurons projecting in ("cortical" inputs)
N_in = 100        # TODO: Experiments are somewhat sensitive to this number

'''
Synaptic params, from Guetig et al 
'''
mu = 0.4
alpha = 1.05
lamb = 0.005
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
nest.SetKernelStatus({'print_time': True})
nest.SetKernelStatus({"local_num_threads": 4})

############## NETWORK ################

# Define custom synapse model
nest.SetDefaults("stdp_synapse",{"tau_plus": tau_w,
                                 "mu_plus":  mu,
                                 "mu_minus": 1.0,
                                 "alpha":    alpha,
                                 "lambda":   lamb,
                                 "Wmax":     2.0 * JE})

# Create population
population = nest.Create('iaf_psc_alpha', N, {
  "tau_minus": tau_w,
  "tau_m": tau_m,
  "V_th": V_th,
  "C_m": C_m,
  "V_reset": V_reset
  })
#print(nest.GetStatus([population[0]], ["V_reset", "V_th", "C_m", "tau_m", "t_ref", "tau_minus"]))

# Create graph
print(col(BOLD, '* Generating SONET graph...'))

W = Graph(N, p, a_rec, a_conv, a_div, cc_chain)

print(col(BOLD, '* Done.'))

# Connect population ids according to generated graph
print(col(BOLD, '* Connecting neurons...'))

for node, neighbors in enumerate(W.adjacency_list):
  if len(neighbors):
    idx = population[0]
    nest.Connect(
      [node + idx],
      [neighbor + idx for neighbor in neighbors],
      syn_spec={
        'delay': 1.0,
        'model': synapse_type,
        'weight': {
          'distribution': 'uniform',
          'low': 0.5 * JE,
          'high': 1.5 * JE
      }})

print(col(BOLD, '* Done.'))

############## DEVICES ################

# One poisson generator with the full input population noise rate, for N_in input neurons
noise = nest.Create('poisson_generator', 1, {'rate': p_rate, 'stop': T_poisson * 1000})
nest.Connect(noise, population[:N_in], syn_spec={'weight': 1.0})

if LOG_ACTIVITY:
  # Spike detector for the entire network
  sd = nest.Create("spike_detector")
  nest.Connect(population, sd)

########### SIMULATE ###############

connections = nest.GetConnections(population, synapse_model='stdp_synapse')

hists = []
# Run simulation
for t in range(T):
  print(col(BOLD, '* Simulating second 0.1 * ' + str((t+1)) ))
  nest.Simulate(100.0)
  current_weights = np.array(nest.GetStatus(connections, 'weight'))

  if PLOT_WEIGHTS:
    plt.figure()
    plt.hist(current_weights, bins=1000, color='green', histtype="stepfilled")
    plt.title("t = " + str(t+1) + "s")
    plt.xlabel("weight")
    plt.ylabel("occurences")
    plt.show(block=False)
    plt.savefig('weights_%d_%ds.png' % (N, t+1))

  if LOG_ACTIVITY:
    dat = nest.GetStatus(sd, keys="events")[0]
    print(col(YELLOW, '%d Neurons spiked %d times') % (len(set(dat['senders'])), len(dat['times']) ))
    nest.SetStatus(sd, [{"n_events": 0}])

  if WEIGHT_EVOL:
    hist = np.histogram(current_weights, range=[0.0, 0.3], normed=True, bins=1000)[0]
    #hist = hist - hist.min()
    #hist = hist / hist.max()
    hist = hist.tolist()
    hists.append(hist)

if WEIGHT_EVOL:
  hists = np.transpose(np.array(hists))
  plt.matshow(hists, cmap=plt.cm.gray_r)
  plt.show(block=False)
  plt.savefig('evol_%d.png' % N)

print(col(BOLD, '* Simulation finished.'))
plt.show()
