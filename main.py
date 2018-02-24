# -*- coding: utf-8 -*-
"""
@author: Andrea Ramazzina, Lukas Mericle
"""

from reconstruct.from_partition import *
from reconstruct.from_community import *
from community import * # what is this library? --Lukas
from mapeq_interface import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


G=nx.karate_club_graph()
A = nx.to_numpy_matrix(G) # Adjacency matrix (might consider to use 'to_scipy_sparse_matrix' as well)

partition = louvain(G)
graph_partition = list_subgraphs(partition)

P0 = compute_Pg(A)
Pgs = compute_Pgs(A, graph_partition)
lambda_gs = compute_lambda_gs(P0, Pgs)

print(lambda_gs)
