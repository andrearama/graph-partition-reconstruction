# -*- coding: utf-8 -*-
"""
@author: Andrea Ramazzina, Lukas Mericle
"""
from info import *
from graphgen import *
import networkx as nx



#G = nx.karate_club_graph()
#G = nx.davis_southern_women_graph()
#G = nx.florentine_families_graph()
G = nx.connected_caveman_graph(8, 6)
#A = nx.to_scipy_sparse_matrix(G, format='csc') # csc since we are primarily interested in multiplying/normalizing columns

#G = rome_graph()
#G = swuspg_graph()
#G = jazz_graph()
#G = pgp_graph()


kl_o2p,kl_p2o = partition_and_divergence(G,mode = 'louvain')

