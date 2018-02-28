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
from graphgen import *
import networkx as nx
import numpy as np


#G = nx.karate_club_graph()
#G = nx.davis_southern_women_graph()
#G = nx.florentine_families_graph()
#G = nx.connected_caveman_graph(8, 6)
#A = nx.to_scipy_sparse_matrix(G, format='csc') # csc since we are primarily interested in multiplying/normalizing columns

G = rome_graph()
A = G.adj
G = G.to_nx()
G, labels = scrub_graph(G)

partition = louvain(G)
communities = list_subgraphs(partition)
#print(communities)

print()

P0 = compute_Pg(A, list(range(A.shape[0])), A.shape[0])
Pgs = compute_Pgs(A, communities)

o2p_lambdas1 = o2p_lambdas(P0, Pgs)
print(o2p_lambdas1)
#o2p_lambdas2 = optimize_lambdas(P0, Pgs, dir="o2p")
#print(o2p_lambdas2)

print()

p2o_lambdas1 = p2o_lambdas(P0, Pgs)
print(p2o_lambdas1)
#p2o_lambdas2 = optimize_lambdas(P0, Pgs, dir="p2o")
#print(p2o_lambdas2)

print()

o2p_Pstar = compute_Pstar(Pgs, o2p_lambdas1)
p2o_Pstar = compute_Pstar(Pgs, p2o_lambdas1)
print(kl_divergence(P0, o2p_Pstar)/len(P0))
print(kl_divergence(p2o_Pstar, P0)/len(P0))

draw_graph(G, communities)
