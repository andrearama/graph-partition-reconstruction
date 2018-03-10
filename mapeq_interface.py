import numpy as np
import subprocess as sp
from os import getcwd


def write_LL(A, fname="A"):
    """
    Write a link list file from a matrix. Saves to the mapeq/graphs/ directory.
    http://www.mapequation.org/code.html#Link-list-format
    """

    ii,jj = A.nonzero()

    with open("mapeq/graphs/"+fname+".txt", "w") as f:

        for i,j in zip(ii, jj):
            f.write("{} {} {}\n".format(j+1, i+1, A[i,j]))


def run_infomap(src_name, src_ext=".txt", options=[ "-dp", "0.15", "--silent"]):
    """
    Run Infomap on the given file (should be housed in mapeq/graphs/).

    Assumes that the Infomap.zip has been unzipped into a subdirectory called
    mapeq, that subdirectories called "graphs" and "out" have been created in
    there, that "graphs" is populated with graph input files, and that this
    function is called from the main directory.
    """

    proc = sp.run(
        ["./mapeq/Infomap", "mapeq/graphs/"+src_name+src_ext, "mapeq/out/"]
                                                                    + options)
    return parse_results("mapeq/out/"+src_name+".tree")



def parse_results(fname):
    with open("mapeq/out/"+src_name+".tree", "r") as f:
        lines = f.readlines()

    lines = [line for line in lines if line[0]!="#"]  # remove comment lines
    nodes = [line.split()[3] for line in lines]  # get node indices
    node_assgn = [line.split()[0] for line in lines]  # get node-partition assignments
    partitions = [p.split(":") for p in node_assgn]  # split node-partition assignments by level
    max_lvl = max(list(map(len, partitions)))  # get max number of partitions

    partition_levels = []
    for lvl in range(max_lvl): # step through level depths
        max_prt = 0
        partition_levels.append([[]]*len(node_assgn))  # assume the limit where each node is in its own partition
        for n,p in zip(node_assgn, partitions):  # awful naming scheme, I know.
            try:  # p[lvl] won't exist for all (p, lvl) because they are not all same length
                prt = int(p[lvl])
                partition_levels[lvl][prt-1].append(int(n)) # add node to partition where it belongs
                if max_prt < prt:
                    max_prt = prt
            except:
                pass
        partition_levels[lvl] = partition_levels[lvl][:max_prt] # limit to only those partitions where nodes were added

    return partition_levels # a list (partitionings) of lists (partitions) of lists (nodes)
