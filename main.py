# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:42:04 2018

@author: andre
"""

import community
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

def compute_stationary_probabilities(A):
    N = A.shape[0]
    #Probability matrix:
    P = np.zeros_like(A)
    for i in range(N):
        for j in range(N):
            if A[i,j] == 0:
                P[i,j] = 0
            else:
                P[i,j] = A[i,j]/(A[i].sum())
    
    #Stationary probability:
    w, v = LA.eig(P.transpose() )        
    stationary_probabilities = v[:,w.argmax()]
    stationary_probabilities = stationary_probabilities/stationary_probabilities.sum()
    stationary_probabilities = stationary_probabilities.real     

    return stationary_probabilities


def create_sub_matrix(matrix,node_list):
    n = len(node_list)
    sub_matrix = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            sub_matrix[i,j] = matrix[node_list[i],node_list[j]]
    
    return sub_matrix
                
def compute_Q_g(A,graph_partition,g):
    node_list = graph_partition[g];
    A_g = create_sub_matrix(A,node_list)
    P_g = compute_stationary_probabilities(A_g)
    P_0 = compute_stationary_probabilities(A)
    Q_g = 1
    for i_g,i_0 in enumerate(node_list):
        Q_g = Q_g *np.power( P_0[i_0]/P_g[i_g], P_g[i_g])
     
    return Q_g

def compute_lambda_g(A,graph_partition,g):
    nominator = compute_Q_g(A,graph_partition,g)
    
    denominator = 0
    for l in range(len(graph_partition)):
        denominator = denominator+compute_Q_g(A,graph_partition,l)
    
    lambda_g = nominator/denominator
    return lambda_g


G=nx.karate_club_graph()

##Adjacency matrix: (migh consider to use 'to_scipy_sparse_matrix' as well)
A = nx.to_numpy_matrix(G)

    
#Partition (Louvain algorithm):
partition = community.best_partition(G)

# Create a list of subgraphs:
graph_partition = []
for community_index in set(partition.values()) :
    node_list = [node for node in partition.keys()
                                if partition[node] == community_index]
    graph_partition.append(node_list)




lambda_1 = compute_lambda_g(A,graph_partition,0)
lambda_2 = compute_lambda_g(A,graph_partition,1)
lambda_3 = compute_lambda_g(A,graph_partition,2)
lambda_4 = compute_lambda_g(A,graph_partition,3)

print(lambda_1+lambda_2+lambda_3+lambda_4)
