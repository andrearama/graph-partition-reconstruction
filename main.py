# -*- coding: utf-8 -*-
"""
@author: Andrea Ramazzina, Lukas Mericle
"""
from info import *
from graphgen import *
import networkx as nx
import matplotlib.pyplot as plt


# Graphs we want to analyze:
# The commented graphs give me an error when using "sparse.linalg.eigs"
graphs = []
graphs.append( nx.karate_club_graph())
graphs.append( nx.davis_southern_women_graph())
#graphs.append( nx.florentine_families_graph()) 
graphs.append( nx.connected_caveman_graph(8, 6))
                            
#graphs.append( rome_graph())
#graphs.append( swuspg_graph())
#graphs.append( jazz_graph())



shapes = ['^','o'] #One for each community detection algorithm
modes = ['louvain','label_prop'] 
colors = ['b','r','g'] #One for each graph

for i in range(0,len(shapes)):
    shape = shapes[i]
    mode = modes[i]
    
    for count, G in enumerate(graphs):
        kl_o2p,kl_p2o = partition_and_divergence(G,mode)
        setting = colors[count]+shape
        plt.plot(kl_o2p,kl_p2o,setting,mfc='none')
