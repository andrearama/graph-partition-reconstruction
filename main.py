# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:42:04 2018

@author: andre
"""

import community # what is this library? --Lukas
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA


def create_Ag(matrix, node_list):
    sub1 = matrix[node_list,:]
    sub_matrix = sub1[:,node_list]
    return sub_matrix
            

def compute_Ags(A, graph_partition):
    Ags = np.zeros_like(graph_partition)
    for i,node_list in enumerate(graph_partition):
        Ags[i] = create_Ag(A, node_list)
    return Ags


def compute_Pg(A):
    P = A/A.sum(axis=0)
    #Stationary probability:
    w, v = LA.eig(P.transpose())
    Pg = v[:,w.argmax()]
    Pg = Pg/Pg.sum()
    return Pg.real


def compute_Pgs(Ags):
    Pgs = np.zeros_like(Ags)
    for i,Ag in enumerate(Ags):
        Pgs[i] = compute_Pg(Ag)
    return Pgs


def compute_Qg(P0, Pg):
    Qg = np.prod(np.power(P0/Pg, Pg))
    return Qg


def compute_Qgs(P0, Pgs):
    Qgs = np.zeros_like(Pgs)
    for i,Ag in enumerate(Pgs):
        Qgs[i] = compute_Qg(P0, Pg)
    return Qgs


def compute_lambda_gs(P0, Pgs):
    Q_g = compute_Q_gs(P0, Pgs)
    return Q_g/Q_g.sum()


G=nx.karate_club_graph()

##Adjacency matrix: (might consider to use 'to_scipy_sparse_matrix' as well)
A = nx.to_numpy_matrix(G)

#Partition (Louvain algorithm):
partition = community.best_partition(G)

# Create a list of subgraphs:
graph_partition = []
for community_index in set(partition.values()):
    node_list = [node for node in partition.keys()
                                if partition[node] == community_index]
    graph_partition.append(node_list)

lambda_g = compute_lambda_gs(A, graph_partition)

print(lambda_g.sum())
