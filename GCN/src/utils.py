"""
Created on Sat Mar 28 21:52:08 2020

@author: mugdh
"""

import numpy as np
import scipy.sparse as sp

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    # The zeroth element of the tuple contains the cell location of each
    # non-zero value in the sparse matrix
    # The first element of the tuple contains the value at each cell location
    # in the sparse matrix
    # The second element of the tuple contains the full shape of the sparse
    # matrix
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
def encode_onehot(labels):
    classes = set(labels)
#print("total classes:",len(classes))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def get_splits_dataset(labels):
#print(labels[0:30,:])
    n_nodes = labels.shape[0]
    nb_classes = labels.shape[1]
    nodes_per_class = 50
#     for i in range(nodes_per_class):
    labels_to_keep0 = [np.random.choice(np.nonzero(labels[:, 0])[0]) for c in range(nodes_per_class)]
    labels_to_keep1 = [np.random.choice(np.nonzero(labels[:, 1])[0]) for c in range(nodes_per_class)]
#         labels_to_keep = [np.random.choice(np.nonzero(labels[:, c])[0]) for c in range(nodes_per_class)]
    labels_to_keep = labels_to_keep0 + labels_to_keep1
#     print("labels to keep:", labels_to_keep0)
#     print("labels to keep:", labels_to_keep1)
#     print("labels to keep:", labels_to_keep)
#     sys.exit()
#     print(labels[labels_to_keep0])
#     print(labels[labels_to_keep1[0],:])
    y_train = np.zeros(shape=labels.shape,
                   dtype=np.float32)
    y_val = labels.copy()
    
    train_mask = np.zeros(shape=(n_nodes,), dtype=np.bool)
    val_mask = np.ones(shape=(n_nodes,), dtype=np.bool)
    
    for l in labels_to_keep:
        y_train[l, :] = labels[l, :]
        y_val[l, :] = np.zeros(shape=(nb_classes,))
        train_mask[l] = True
        val_mask[l] = False
#     print("y shape:", labels[0],labels[10], labels[-1])
#     maxList = split_train_test(labels)   
#     all = len(maxList)
#     test = int((all/100)*40)
#     val= int((all/100)*30)
#     idx_test = maxList[0:test]
#     idx_val = maxList[test:(test+val)]
#     idx_train = maxList[(test+val):]
#     idx_test = range(10000,12448)
#     idx_train = range(0,500)
#     idx_val = range(12448,16260)
#     train_mask = sample_mask(idx_train, labels.shape[0])
#     val_mask = sample_mask(idx_val, labels.shape[0])
#     test_mask = sample_mask(idx_test, labels.shape[0])
#     y_train = np.zeros(labels.shape)
#     y_val = np.zeros(labels.shape)
#     y_test = np.zeros(labels.shape)
#     y_train[train_mask, :] = labels[train_mask, :]
#     y_val[val_mask, :] = labels[val_mask, :]
#     y_test[test_mask, :] = labels[test_mask, :]
#     print("y_train shape:",y_train.shape,"y_val type:", y_val.shape,"y test type:",y_test.shape,"train mask shape:", train_mask.shape,"val mask:",val_mask.shape,"test mask shape:", test_mask.shape)
    return y_train, y_val, train_mask, val_mask

def load_dataset():
    idx_features_labels = np.genfromtxt("{}{}.txt".format('../data/', "clustering_new"), dtype=np.dtype(int))
    np.random.shuffle(idx_features_labels)
    class_lables = np.array(idx_features_labels[:, 1], dtype = int)
    labels = encode_onehot(class_lables)
    features = sp.eye((labels.shape[0]), dtype= np.float32).tolil()
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.txt".format('../data/', "edge_list_new"), dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[1], edges.shape[0], features.shape[1]))
    return features, adj, labels, edges


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

