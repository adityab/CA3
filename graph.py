import subprocess
import numpy as np
import networkx as nx

from utils import *

class Graph(object):
  def __init__(self, n, p, a_rec, a_conv, a_div, cc_chain):
    self.n = n
    self.params = [p, a_rec, a_conv, a_div, cc_chain]

    # Format args upto 3 decimal places
    formatted = [format(param, '.3f') for param in self.params]

    # Run SONETS to generate graph
    subprocess.run(map(str, ['sonets/run_secorder', n] + formatted))

    # Generated filename
    graph_name = '_'.join(['', str(n)] + formatted)

    # Read graph statistics
    stats = np.loadtxt('data/stats' + graph_name + '.dat', dtype=float)
    self.params_requested = stats[1]
    self.params_detected = stats[0]

    # Read adjacency matrix
    self.matrix = np.loadtxt('data/w' + graph_name + '.dat', dtype=float)
    # Turn it into a NetworkX object
    self.G = nx.DiGraph(self.matrix)

  def adjacency_list(self):
    return self.G.adjacency_list()

#g = Graph(20, 0.1, 0, 0, 0, 0)
# print(g.G.edges())
# print(g.G.adjacency_list())
