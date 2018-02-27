import numpy as np
from scipy import sparse
import networkx as nx

class Graph(object):

    def __init__(self):
        self.adj = None
        self.n = 0

    def load(self, m):
        self.adj = sparse.csr_matrix(m)
        self.n = self.adj.shape[0] # assume square

    def to_nx(self):
        return nx.convert_matrix.from_scipy_sparse_matrix(self.adj)


class ERGraph(Graph):

    def __init__(self, n, p):
        super().__init__()
        self.n = n
        self.p = p
        self.adj = sparse.triu(np.random.rand(n,n) < p*(n-1)/n)
        self.symmetrize()
        self.adj = self.adj.tocsr()


class SWGraph(Graph):

    def __init__(self, n, c, p):
        assert c%2==0
        super().__init__()
        self.n = n
        self.c = c
        self.p = p
        self.adj = sparse.dia_matrix((n,n))
        for k in range(1,int(c/2)+1):
            self.adj.setdiag(1,k)
            self.adj.setdiag(1,n-k)
        self.new_connections()
        self.symmetrize()
        self.adj = self.adj.tocsr()

    def new_connections(self):
        self.adj = self.adj.maximum(sparse.triu(np.random.rand(self.n, self.n) < self.p, k=1))


class ABGraph(Graph):

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
        self.adj = self.adj.tocsr()


def from_adjlist(fname):
    net_inds = np.fromfile(fname, sep=' ').reshape(-1,2).astype(np.uint32)-1
    netn = net_inds.max()+1
    net = sparse.csr_matrix((np.ones(net_inds.shape[0], dtype=np.uint32), (net_inds[:,0], net_inds[:,1])), shape=(netn, netn))
    return net
