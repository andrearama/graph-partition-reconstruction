import numpy as np
import subprocess as sp
from os import getcwd


def write_LL(A, fname="A"):
    """
    Write a link list file from a matrix. Saves to the mapeq/graphs/ directory.
    http://www.mapequation.org/code.html#Link-list-format
    """
    f = open("mapeq/graphs/"+fname+".txt", "w")
    ii,jj = np.where(A!=0)
    for i,j in zip(ii, jj):
        w = A[i,j]
        f.write("{} {} {}\n".format(j+1, i+1, w))
    f.close()


def run_infomap(src_name, src_ext=".net",
                            options=[ "-dp", "0", "--overlapping", "--silent"]):
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
    for line in lines:
        print(line)
