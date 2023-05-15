import torch
import numpy as np
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.utils.num_nodes import maybe_num_nodes

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

EPS = 1e-15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Node2Vec(torch.nn.Module):
    """
    Adapted from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html

    The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    """
    def __init__(self, input_dim, hidden_dims, num_points, dropout, edge_index, embedding_dim, walk_length, context_size,
                 walks_per_node=1, p=1, q=1, num_negative_samples=1,
                 num_nodes=None, sparse=False):
        super(Node2Vec, self).__init__()

        if random_walk is None:
            raise ImportError('`Node2Vec` requires `torch-cluster`.')

        N = maybe_num_nodes(edge_index, num_nodes)
        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        self.adj = self.adj.to('cpu')

        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        self.clus = None
        self.embedding = None
        #self.embedding = torch.empty(num_points, hidden_dims).to(device)

        self.w1 = nn.Linear(input_dim, hidden_dims, bias=True)
        self.w2 = nn.Linear(input_dim, hidden_dims, bias=True)
        self.w1.bias.data.fill_(0.0)
        self.w2.bias.data.fill_(0.0)

        self.iden = nn.Parameter(data = torch.randn((num_points, hidden_dims),dtype=torch.float).to(device), requires_grad=True)

    def get_embedding(self, X1, X2):
        return normalize( self.w1(X1) + self.w2(X2) + self.iden)

    def update_B(self, X1, X2, unique=None):
        self.embedding = normalize(self.w1(X1) + self.w2(X2) + F.embedding(unique, self.iden))

    def loader(self, **kwargs):
        return DataLoader(range(self.adj.sparse_size(0)),
                          collate_fn=self.sample, **kwargs)

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)


    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        if self.clus==None:
            rw = torch.randint(self.adj.sparse_size(0),(batch.size(0), self.walk_length*self.num_negative_samples))
        else:
            rw = torch.empty(batch.size(0), self.walk_length*self.num_negative_samples, dtype=torch.long)
            for i in range(self.clus.max()+1):
                cluster = (self.clus[batch.view(-1)]==i).nonzero(as_tuple=True)[0].tolist()
                neg_ind = (self.clus!=i).nonzero(as_tuple=True)[0]
                rw[cluster] = neg_ind[torch.randint(neg_ind.size(0), (len(cluster), self.walk_length*self.num_negative_samples))]
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)


    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)


    def loss(self, pos_rw, neg_rw, mapping=None):
        r"""Computes the loss given positive and negative random walks."""

        pos_rw = F.embedding(pos_rw.view(-1), mapping.view(-1,1)).view(pos_rw.size())
        neg_rw = F.embedding(neg_rw.view(-1), mapping.view(-1,1)).view(neg_rw.size())
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = F.embedding(start, self.embedding).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = F.embedding(rest.view(-1), self.embedding).view(pos_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1)
        pos_loss = torch.logsumexp(out, dim=-1)

        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = F.embedding(start, self.embedding).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = F.embedding(rest.view(-1), self.embedding).view(neg_rw.size(0), -1, self.embedding_dim)
        out = (h_start * h_rest).sum(dim=-1)

        neg_loss = torch.logsumexp(out, dim=-1)
        neg_loss = torch.logsumexp(torch.cat((neg_loss.view(-1,1), pos_loss.view(-1,1)), dim=-1), dim=-1)
        #return -1*torch.mean(pos_loss - neg_loss)

        return -1*torch.mean(torch.exp(pos_loss-neg_loss))


def Kmeans(x, K=-1, Niter=10, verbose=False):
    #start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space
    x_temp = x.detach()

    temp = set()
    while len(temp)<K:
        temp.add(np.random.randint(0, N))
    c = x_temp[list(temp), :].clone()

    x_i = x_temp.view(N, 1, D) # (N, 1, D) samples
    cutoff = 1
    if K>cutoff:
        c_j = []
        niter=K//cutoff
        rem = K%cutoff
        if rem>0:
            rem=1
        for i in range(niter+rem):
            c_j.append(c[i*cutoff:min(K,(i+1)*cutoff),:].view(1, min(K,(i+1)*cutoff)-(i*cutoff), D))
    else:
        c_j = c.view(1, K, D) # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        #print("iteration: " + str(i))

        # E step: assign points to the closest cluster -------------------------
        if K>cutoff:
            for j in range(len(c_j)):
                if j==0:
                    D_ij = ((x_i - c_j[j]) ** 2).sum(-1)
                else:
                    D_ij = torch.cat((D_ij,((x_i - c_j[j]) ** 2).sum(-1)), dim=-1)
                    # D_ij += ((x_i - c_j[j]) ** 2).sum(-1)
        else:
            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        assert D_ij.size(1)==K

        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        # print(D_ij[0,:100])

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x_temp)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        # print(Ncl[:10])
        Ncl += 0.00000000001
        c /= Ncl  # in-place division to compute the average

    return cl, c

