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


def run_infomap(src_name, src_ext=".net", options=[ "-dp", "0", "--silent"]):
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

    with open(getcwd()+"/"+src_name+".tree", "r") as f:
        lines = f.readlines()
    for line in lines: # replace with something more helpful, like transforming into series of graph partitions
        print(line)
