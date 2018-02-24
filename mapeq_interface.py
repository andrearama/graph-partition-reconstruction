import numpy as np


def write_LL(A, fname="A.txt"):
    """
    Write a link list file from a matrix.
    http://www.mapequation.org/code.html#Link-list-format
    """
    f = open(fname, "w")
    ii,jj = np.where(A!=0)
    for i,j in zip(ii, jj):
        w = A[i,j]
        f.write("{} {} {}\n".format(j+1, i+1, w))
    f.close()
