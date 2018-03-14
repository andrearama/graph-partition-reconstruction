# -*- coding: utf-8 -*-
"""
@author: Andrea Ramazzina, Lukas Mericle
"""
from info import *
from graphgen import *
import networkx as nx
import matplotlib.pyplot as plt
from mapeq_interface import *
import gc
import pickle
from os.path import exists

graphs = ['ergr_graph',
          'swgr_graph',
          'abgr_graph',
          'rome_graph',
          'swuspg_graph',
          'jazz_graph']

names = ['random', 'smallworld', 'prefgrowh', 'rome', 'swuspg', 'jazz']
modes = ['louvain', 'mapeq', 'label_prop']

if not exists("results.pickle"):
    with open("results.pickle", "wb") as f:
        pickle.dump({}, f)

with open("results.pickle", "rb") as f:
    output = pickle.load(f)

for (Gstr,name) in zip(graphs, names):

    print()
    print()
    G = globals()[Gstr]() # lazy, memory-efficient (https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string)
    gc.collect()
    G.name = name
    output[name] = {}
    print("Graph", name)
    clustering_coefficient = nx.average_clustering(G)
    print("CLC", clustering_coefficient)
    density_coefficient = nx.density(G)
    print("DNC", density_coefficient)
    print("NND", len(G))
    output[name]["clc"] = clustering_coefficient
    output[name]["dnc"] = density_coefficient
    output[name]["nnd"] = len(G)

    for mode in modes:

        print()
        print("Mode", mode)
        output[name][mode] = {}
        levels = get_partition_levels(G, mode=mode)
        gc.collect()
        kl_o2p, kl_p2o = get_partition_divergences(G, levels)
        print("O2P", kl_o2p)
        print("P2O", kl_p2o)

        output[name][mode]["o2p"] = kl_o2p
        output[name][mode]["p2o"] = kl_p2o
        output[name][mode]["cvg"] = []
        output[name][mode]["ncl"] = []

        for i,communities in enumerate(levels):

            print()
            print("Community", i)
            coverage = nx.algorithms.community.quality.coverage(G, communities)
            print("CVG", coverage)
            output[name][mode]["cvg"].append(coverage)
            n_clusters = len(communities)
            print("NCL", n_clusters)
            output[name][mode]["ncl"].append(n_clusters)

with open("results.pickle", "wb") as f:
    pickle.dump(output, f)
