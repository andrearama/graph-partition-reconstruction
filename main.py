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
import matplotlib.pyplot as plt
import numpy as np


G = nx.karate_club_graph()
#G = nx.davis_southern_women_graph()
#G = nx.florentine_families_graph()
A = nx.to_numpy_matrix(G)
labels = G.nodes
partition = louvain(G)
graph_partition = list_subgraphs(partition)
print(labeled_partition(graph_partition, labels))

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

print()

o2p_Pstar = compute_Pstar(Pgs, o2p_lambdas1)
p2o_Pstar = compute_Pstar(Pgs, p2o_lambdas1)
print(kl_divergence(P0, o2p_Pstar)/len(P0))
print(kl_divergence(p2o_Pstar, P0)/len(P0))

run_infomap("ninetriangles")
