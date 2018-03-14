import numpy as np
from scipy import sparse
import networkx as nx
from mapeq_interface import write_LL
from os.path import exists


class MyGraph(object):


    def __init__(self):
        self.adj = None
        self.n = 0

    def load(self, m):
        self.adj = sparse.csc_matrix(m, dtype=np.float64)
        self.n = self.adj.shape[0] # assume square

    def degree_distribution(self):
        dist = self.adj.sum(axis=0).A1

        return dist

    def symmetrize(self):
        self.adj = self.adj.maximum(self.adj.transpose())

    def to_nx(self):
        return nx.convert_matrix.from_scipy_sparse_matrix(self.adj)

    def to_el(self, fname):
        ii,jj = self.adj.nonzero()

        with open("el_raw/"+fname+".txt", "w") as f:
            f.write("# N {:d}\n".format(self.adj.shape[0]))

            for i,j in zip(ii, jj):
                f.write("{} {} {}\n".format(j+1, i+1, int(self.adj[i,j])))


class ERGraph(MyGraph):

    def __init__(self, n, p):
        super().__init__()
        self.n = n
        self.p = p

        self.adj = sparse.triu(np.random.rand(n,n) < p*(n-1)/n)
        self.symmetrize()
        self.adj = self.adj.tocsc()


class SWGraph(MyGraph):

    def __init__(self, n, c, p):
        assert c%2==0
        super().__init__()
        self.n = n
        self.c = c
        self.p = p

        self.adj = sparse.dia_matrix((n,n))

        for k in range(1,int(c/2)+1):
            self.adj.setdiag(1,k) # set upper triangle for simplicity, symmetrize at end
        self.adj = self.adj.tocsc()

        self.new_connections()

        self.symmetrize()

    def new_connections(self):
        ii,jj = self.adj.nonzero()
        for i,j in zip(ii,jj):
            if np.random.rand() < self.p:
                empty_slots = np.where(self.adj.getcol(j).todense().flatten()==0)[0]
                candidate_positions = empty_slots[np.where(empty_slots<j)] # only operate on upper triangle
                new_i = np.random.choice(candidate_positions)
                self.adj[i,j] = 0
                self.adj[new_i, j] = 1


class ABGraph(MyGraph):

    def __init__(self, n, n0, m):
        assert n0 >= m
        super().__init__()
        self.n = n
        self.m = m

        self.grow(n0)

    def grow(self, n0):

        self.adj = sparse.lil_matrix((self.n,self.n))
        self.adj[:n0,:n0] = sparse.csr_matrix(np.ones((n0,n0)))
        self.adj.setdiag([0]*n0)

        dd = self.degree_distribution()
        ddsample = []

        for i,d in enumerate(dd):
            for j in range(d.astype(np.uint16)):
                ddsample.append(i)

        for i in range(n0, self.n):

            add = [np.random.choice(ddsample)]
            self.adj[i,add[0]] = 1
            self.adj[add[0],i] = 1

            while len(add) < self.m:

                new = np.random.choice(ddsample)

                if new not in add:

                    add.append(new)
                    self.adj[i,new] = 1
                    self.adj[new,i] = 1

            ddsample += add + [i]*self.m

        self.adj = self.adj.tocsc()


def from_adjlist(fname):

    net_inds = np.fromfile(fname, sep=' ').reshape(-1,2).astype(np.uint32)-1

    netn = net_inds.max()+1

    net = sparse.csc_matrix((np.ones(net_inds.shape[0], dtype=np.uint32), (net_inds[:,0], net_inds[:,1])), shape=(netn, netn))

    return net


def parse_el(fname, directed=False, weighted=False, indexing=1):
    """
    Accepts the contents of an edge list file with the comments already removed.
    Parses an edge list and returns a populated MyGraph() object. Additionally
    writes a link list file to mapeq/graphs/ if it does not exist there.
    """

    with open("el_raw/"+fname, "r") as f:
        lines = f.readlines()

    n = int(lines[0][4:])

    A = sparse.csc_matrix((n,n))

    for line in lines[1:]:

        spl = line.split()

        j = int(spl[0])-indexing
        i = int(spl[1])-indexing

        if not weighted:
            A[i,j] = 1
        else:
            A[i,j] = int(spl[2])

    G = MyGraph()
    G.load(A)

    if not directed:
        G.symmetrize()

    if not exists("mapeq/graphs/"+fname):
        write_LL(G.adj, fname=fname.split(".")[0])

    return G.to_nx()


def swuspg_graph():
    """
    https://toreopsahl.com/datasets/#uspowergrid
    """

    return parse_el("swuspg.txt")


def jazz_graph():
    """
    http://deim.urv.cat/~alexandre.arenas/data/welcome.htm
    """

    return parse_el("jazz.txt")


def pgp_graph():
    """
    http://deim.urv.cat/~alexandre.arenas/data/welcome.htm
    """

    return parse_el("pgp.txt")


def rome_graph():
    """
    http://www.dis.uniroma1.it/~challenge9
    """

    return parse_el("rome.txt", directed=True)

def ergr_graph():
    return parse_el("ergr-N2500-P0.002.txt")

def swgr_graph():
    return parse_el("swgr-N2500-C12-P0.10.txt")

def abgr_graph():
    return parse_el("abgr-N2500-N012-M3.txt")
