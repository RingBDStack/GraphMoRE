import random
import torch
import networkx as nx
import numpy as np
import torch_geometric.data
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Amazon, Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.utils import negative_sampling
import scipy.sparse as sp
import pickle as pkl
import os
import torch_geometric.transforms as T
import warnings
warnings.filterwarnings('ignore')
seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def get_mask(idx, length):
    """Create mask.
    """
    mask = torch.zeros(length, dtype=torch.bool)
    mask[idx] = 1
    return mask

def load_data(root: str, data_name: str, split='public', **kwargs):
    if data_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=root, name=data_name, split=split)
        train_mask, val_mask, test_mask = dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask
    elif data_name == "airport":
        dataset = Airport(root)
        train_mask, val_mask, test_mask = dataset.data.mask
    elif data_name == "photo":
        dataset = Amazon(root=root, name="Photo")
        labels = dataset.data.y.tolist()
        val_prop, test_prop = 0.15, 0.15
        val_mask, test_mask, train_mask = split_data(labels, val_prop, test_prop, seed=3047)
        mask = (train_mask, val_mask, test_mask)
        features = dataset.data.x
        num_features = dataset.num_features
        edge_index = dataset.data.edge_index.long()
        neg_edges = negative_sampling(edge_index)
        num_classes = dataset.num_classes
        labels = torch.tensor(labels)
        return features, num_features, labels, edge_index, neg_edges, mask, num_classes
    else:
        raise NotImplementedError
    mask = (train_mask, val_mask, test_mask)
    features = dataset.data.x
    num_features = dataset.num_features
    labels = dataset.data.y
    edge_index = dataset.data.edge_index.long()
    neg_edges = negative_sampling(edge_index)
    num_classes = dataset.num_classes
    return features, num_features, labels, edge_index, neg_edges, mask, num_classes

def load_synthetic_data(root: str, data_name: str):
    with open(f'{root}/{data_name}.pkl', 'rb') as f:
        G = pkl.load(f)
    with open(f'{root}/{data_name}_feature.pkl', 'rb') as f:
        features = pkl.load(f)
    features = torch.tensor(features).float()
    num_features = features.shape[-1]
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    neg_edges = negative_sampling(edge_index)
    perm = torch.randperm(edge_index.shape[-1])
    edge_index = edge_index[:, perm]
    perm = torch.randperm(neg_edges.shape[-1])
    neg_edges = neg_edges[:, perm]
    labels = torch.tensor([])
    mask = torch.tensor([])
    num_classes = None
    return features, num_features, labels, edge_index, neg_edges, mask, num_classes


def mask_edges(edge_index, neg_edges, val_prop, test_prop):
    n = len(edge_index[0])
    n_val = int(val_prop * n)
    n_test = int(test_prop * n)
    edge_val, edge_test, edge_train = edge_index[:, :n_val], edge_index[:, n_val:n_val + n_test], edge_index[:, n_val + n_test:]
    val_edges_neg, test_edges_neg = neg_edges[:, :n_val], neg_edges[:, n_val:n_test + n_val]
    train_edges_neg = torch.concat([neg_edges, edge_val, edge_test], dim=-1)
    return (edge_train, edge_val, edge_test), (train_edges_neg, val_edges_neg, test_edges_neg)

def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()

def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.shape[0], 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features

def split_data(labels, val_prop, test_prop, seed):
    random.seed(seed)
    num_class = np.max(labels) + 1
    label_dict = dict()
    for i in range(num_class):
        label_dict[i] = []
    for i, l in enumerate(labels):
        label_dict[l].append(i)
    idx_train, idx_val, idx_test = [], [], []
    for i in range(num_class):
        random.shuffle(label_dict[i])
        num_val = round(val_prop * len(label_dict[i]))
        num_test = round(test_prop * len(label_dict[i]))
        idx_val += label_dict[i][:num_val]
        idx_test += label_dict[i][num_val:num_val + num_test]
        idx_train += label_dict[i][num_val + num_test:]
    return idx_val, idx_test, idx_train


class Airport(InMemoryDataset):
    def __init__(self, root):
        super(Airport, self).__init__()
        val_prop, test_prop = 0.15, 0.15
        graph = pkl.load(open(f"{root}/airport/airport.p", 'rb'))
        adj = nx.adjacency_matrix(graph).toarray()
        row, col = np.nonzero(adj)
        edge_index = np.concatenate([row[None], col[None]], axis=0)
        features = np.array([graph._node[u]['feat'] for u in graph.nodes()])
        features = augment(adj, torch.tensor(features).float())
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])

        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, random.seed(3047))
        mask = (idx_train, idx_val, idx_test)

        self.data = torch_geometric.data.Data(x=features,
                                              edge_index=torch.tensor(edge_index),
                                              y=torch.tensor(labels),
                                              mask=mask)

        @property
        def num_features(self) -> int:
            return self.data.x.shape[-1]

        @property
        def raw_file_names(self):
            pass

        @property
        def processed_file_names(self):
            pass

        def download(self):
            pass

        def process(self):
            pass

