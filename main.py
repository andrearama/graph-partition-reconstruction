# -*- coding: utf-8 -*-
"""
@author: Andrea Ramazzina, Lukas Mericle
"""

from reconstruct.from_partition import *
from reconstruct.from_community import *
from reconstruct.utils import *
from community1 import *
from info import *
from mapeq_interface import *
import networkx as nx
import numpy as np


G = nx.karate_club_graph()
#G = nx.davis_southern_women_graph()
#G = nx.florentine_families_graph()
A = nx.to_numpy_matrix(G) # Adjacency matrix (might consider to use 'to_scipy_sparse_matrix' as well)
partition = louvain(G)
graph_partition = list_subgraphs(partition)
print(graph_partition)

print()

P0 = compute_Pg(A, list(range(A.shape[0])), A.shape[0])
Pgs = compute_Pgs(A, graph_partition)

o2p_lambdas1 = o2p_lambdas(P0, Pgs)
print(o2p_lambdas1)
o2p_lambdas2 = optimize_lambdas(P0, Pgs, dir="o2p")
print(o2p_lambdas2)

print()

p2o_lambdas1 = p2o_lambdas(P0, Pgs)
print(p2o_lambdas1)
p2o_lambdas2 = optimize_lambdas(P0, Pgs, dir="p2o")
print(p2o_lambdas2)


draw_graph(G, graph_partition)
