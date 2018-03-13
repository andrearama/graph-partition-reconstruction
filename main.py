# -*- coding: utf-8 -*-
"""
@author: Andrea Ramazzina, Lukas Mericle
"""
from info import *
from graphgen import *
import networkx as nx
import matplotlib.pyplot as plt
from mapeq_interface import *


# Graphs we want to analyze:
# The commented graphs give me an error when using "sparse.linalg.eigs"
graphs = []
graphs.append( nx.karate_club_graph())
graphs.append( nx.davis_southern_women_graph())
graphs.append( nx.florentine_families_graph())
graphs.append( nx.connected_caveman_graph(8, 6))

graphs.append( rome_graph())
graphs.append( swuspg_graph())
graphs.append( jazz_graph())

names = ['karate', 'davis', 'florence', 'caveman', 'rome', 'swuspg', 'jazz']
shapes = ['^','o', '*'] #One for each community detection algorithm
modes = ['louvain', 'mapeq', 'label_prop']
#shapes = ['*']
#modes = ['mapeq']
colors = ['b','r','g', 'k', 'm', 'y', 'c'] #One for each graph


for count, G in enumerate(graphs):

    for i in range(0,len(shapes)):
        shape = shapes[i]
        mode = modes[i]
        setting = colors[count]+shape
        G.name = names[i]

        levels = get_partition_levels(G, mode=mode)
        kl_o2p,kl_p2o = get_partition_divergences(G, levels)
        clustering_coefficient = nx.average_clustering(G)
        density_coefficient = nx.density(G)

        plt.subplot(2, 2, 1)
        for i in range(len(levels)):
            plt.plot(kl_o2p[i]/len(G),kl_p2o[i]/len(G),setting,mfc='none')

        plt.subplot(2, 2, 2) #looks like as the clustering increases the difference in results decreases.
        for i in range(len(levels)):
            plt.plot(clustering_coefficient,kl_p2o[i]/len(G),setting,mfc='none')

        plt.subplot(2, 2, 3) # Similar to the other one
        for i in range(len(levels)):
            plt.plot(len(G),kl_p2o[i],setting,mfc='none')

        plt.subplot(2, 2, 4)  #looks like as the density decreases the difference in results decreases.
        for i in range(len(levels)):
            plt.plot(density_coefficient,kl_p2o[i]/len(G),setting,mfc='none')

#

#plt.subplot(2, 2, 1)
#plt.xlabel('KL[P0;P*]')
#plt.ylabel('KL[P*;P0]')
#plt.subplot(2, 2, 2)
#plt.xlabel('clustering coefficient')
#plt.ylabel('KL[P*;P0]')
plt.show()
