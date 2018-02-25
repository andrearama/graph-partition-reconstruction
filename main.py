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


#G = nx.karate_club_graph()
#G = nx.davis_southern_women_graph()
G = nx.florentine_families_graph()
A = nx.to_numpy_matrix(G) # Adjacency matrix (might consider to use 'to_scipy_sparse_matrix' as well)
partition = louvain(G)
graph_partition = list_subgraphs(partition)
print(graph_partition)

P0 = compute_Pg(A, list(range(A.shape[0])), A.shape[0])
Pgs = compute_Pgs(A, graph_partition)

lambda_gs1 = compute_lambda_gs(P0, Pgs)
print(lambda_gs1)
lambda_gs2 = optimize_lambdas(P0, Pgs)
print(lambda_gs2)

Pstar = compute_Pstar(Pgs, lambda_gs2)

mus = np.zeros(Pgs.shape[1])
for i in range(Pgs.shape[1]):
    mus[i] = compute_mug(P0, Pstar, Pgs[:,i])
print(mus.mean())
