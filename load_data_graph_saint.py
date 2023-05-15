import numpy as np
import json
import pdb
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import os
import yaml
import scipy.sparse as sp
import torch
#from graphsaint.globals import *


def load_data(prefix, normalize=True):
    """
    Load the various data files residing in the `prefix` directory.
    Files to be loaded:
        adj_full.npz        sparse matrix in CSR format, stored as scipy.sparse.csr_matrix
                            The shape is N by N. Non-zeros in the matrix correspond to all
                            the edges in the full graph. It doesn't matter if the two nodes
                            connected by an edge are training, validation or test nodes.
                            For unweighted graph, the non-zeros are all 1.
        adj_train.npz       sparse matrix in CSR format, stored as a scipy.sparse.csr_matrix
                            The shape is also N by N. However, non-zeros in the matrix only
                            correspond to edges connecting two training nodes. The graph
                            sampler only picks nodes/edges from this adj_train, not adj_full.
                            Therefore, neither the attribute information nor the structural
                            information are revealed during training. Also, note that only
                            a x N rows and cols of adj_train contains non-zeros. For
                            unweighted graph, the non-zeros are all 1.
        role.json           a dict of three keys. Key 'tr' corresponds to the list of all
                              'tr':     list of all training node indices
                              'va':     list of all validation node indices
                              'te':     list of all test node indices
                            Note that in the raw data, nodes may have string-type ID. You
                            need to re-assign numerical ID (0 to N-1) to the nodes, so that
                            you can index into the matrices of adj, features and class labels.
        class_map.json      a dict of length N. Each key is a node index, and each value is
                            either a length C binary list (for multi-class classification)
                            or an integer scalar (0 to C-1, for single-class classification).
        feats.npz           a numpy array of shape N by F. Row i corresponds to the attribute
                            vector of node i.
    Inputs:
        prefix              string, directory containing the above graph related files
        normalize           bool, whether or not to normalize the node features
    Outputs:
        adj_full            scipy sparse CSR (shape N x N, |E| non-zeros), the adj matrix of
                            the full graph, with N being total num of train + val + test nodes.
        adj_train           scipy sparse CSR (shape N x N, |E'| non-zeros), the adj matrix of
                            the training graph. While the shape is the same as adj_full, the
                            rows/cols corresponding to val/test nodes in adj_train are all-zero.
        feats               np array (shape N x f), the node feature matrix, with f being the
                            length of each node feature vector.
        class_map           dict, where key is the node ID and value is the classes this node
                            belongs to.
        role                dict, where keys are: 'tr' for train, 'va' for validation and 'te'
                            for test nodes. The value is the list of IDs of nodes belonging to
                            the train/val/test sets.
    """
    # adj_full = scipy.sparse.load_npz('{}/adj_full.npz'.format(prefix)).astype(np.bool)
    # adj_train = scipy.sparse.load_npz('{}/adj_train.npz'.format(prefix)).astype(np.bool)
    # role = json.load(open('{}/role.json'.format(prefix)))
    # feats = np.load('{}/feats.npy'.format(prefix))
    # class_map = json.load(open('{}/class_map.json'.format(prefix)))

    adj_full = scipy.sparse.load_npz('{}/adj_full.npz'.format(prefix)).astype(bool)
    adj_train = scipy.sparse.load_npz('{}/adj_train.npz'.format(prefix)).astype(bool)
    role = json.load(open('{}/role.json'.format(prefix)))
    feats = np.load('{}/feats.npy'.format(prefix))
    class_map = json.load(open('{}/class_map.json'.format(prefix)))


    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    '''
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    '''
    # -------------------------
    adj_coo = adj_full.tocoo()
    row = adj_coo.row
    col = adj_coo.col
    adj_full = torch.tensor(np.array([row, col]))
    adj_full = torch.torch.sparse_coo_tensor(adj_full, torch.ones(row.size), adj_coo.shape)
    feats = torch.tensor(feats)
    label = torch.tensor([class_map[i] for i in range(adj_coo.shape[0])])
    return adj_full, adj_train, feats, label, role