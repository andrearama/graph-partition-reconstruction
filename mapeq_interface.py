import numpy as np
import subprocess as sp
from os import getcwd
from os.path import exists
import networkx as nx


def write_LL(A, fname="A"):
    """
    Write a link list file from a matrix. Saves to the mapeq/graphs/ directory.
    http://www.mapequation.org/code.html#Link-list-format
    """

    ii,jj = A.nonzero()

    with open("mapeq/graphs/"+fname+".txt", "w") as f:

        for i,j in zip(ii, jj):
            f.write("{} {} {}\n".format(j+1, i+1, A[i,j]))


def run_infomap(G, src_ext=".txt", options=[ "-dp", "0.15", "--silent"]):
    """
    Run Infomap on the given file (should be housed in mapeq/graphs/).

    Assumes that the Infomap.zip has been unzipped into a subdirectory called
    mapeq, that subdirectories called "graphs" and "out" have been created in
    there, that "graphs" is populated with graph input files, and that this
    function is called from the main directory.
    """

    src_name = G.name
    adj = nx.to_scipy_sparse_matrix(G, format='csc')

    write_LL(adj, fname=src_name)

    proc = sp.run(
        ["./mapeq/Infomap", "mapeq/graphs/"+src_name+src_ext, "mapeq/out/"]
                                                                    + options)
    results = parse_results("mapeq/out/"+src_name+".tree")
    res_w_names = enum_partitions(results)
    return res_w_names



def parse_results(fname):
    """
    Parse the results of the map equation script into a form like the other
    methods we use.
    """

    with open(fname, "r") as f:
        lines = f.readlines()

    lines = [line for line in lines if line[0]!="#"]  # remove comment lines
    nodes = [int(line.split()[3]) for line in lines]  # get node indices
    node_assgn = [line.split()[0] for line in lines]  # get node-partition assignments
    partitions = [list(map(int, p.split(":"))) for p in node_assgn]  # split node-partition assignments by level
    partitions = square_out_partitions(partitions) # not sure if needed
    max_per_lvl = np.max(partitions, axis=0)

    for i in range(1, partitions.shape[1]):
        partitions[:,i] += (partitions[:,i-1]-1)*max_per_lvl[i]
    partitions -= 1

    partition_levels = [[[] for _ in range(max(partitions[:,i])+1)] for i in range(partitions.shape[1])]
    for lvl in range(partitions.shape[1]):
        for n,p in zip(nodes, partitions):
            prt = p[lvl]
            partition_levels[lvl][prt].append(n-1)

    for i,part in enumerate(partition_levels):
        partition_levels[i] = [p for p in part if len(p)>0]

    return partition_levels

def enum_partitions(results):
    """
    Return a dict where the keys are indices (0, 1, 2, ...) and the
    values are the partitionings at those levels.
    """

    out = {} # dict
    for i,part in enumerate(results):
        out[i] = part
    return out

def square_out_partitions(partitions):
    max_lvl = max(map(len, partitions))
    for p in partitions:
        while len(p) < max_lvl:
            p.append(0)
    return np.array(partitions)
