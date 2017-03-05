import numpy as np
import networkx as nx

class Graph(object):
  def __init__(self, n, p):
    self.G = nx.fast_gnp_random_graph(n, p, seed=None, directed=True)

  def adjacency_list(self):
    return self.G.adjacency_list()

# g = Graph(4, .5)
# print(g.G.edges())
# print(g.G.adjacency_list())
