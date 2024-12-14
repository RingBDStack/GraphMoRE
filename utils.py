import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random


def cal_accuracy(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    correct = (preds == trues).sum()
    return correct / len(trues)


def cal_F1(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    weighted_f1 = f1_score(trues, preds, average='weighted')
    macro_f1 = f1_score(trues, preds, average='macro')
    return weighted_f1, macro_f1


def cal_AUC_AP(scores, trues):
    auc = roc_auc_score(trues, scores)
    ap = average_precision_score(trues, scores)
    return auc, ap

def cal_shortest_dis(edge_index):
    dis_shortest = {}
    edge_index_ = edge_index.cpu().numpy().astype(int).tolist()
    G = nx.Graph()
    for i in range(len(edge_index_[0])):
        G.add_edge(edge_index_[0][i], edge_index_[1][i])
    d = dict(nx.shortest_path_length(G))
    for i in range(len(edge_index_[0])):
        dis = d[edge_index_[0][i]][edge_index_[1][i]]
        if dis == 0:
            dis = np.inf
        dis_shortest[(edge_index_[0][i], edge_index_[1][i])] = dis
        dis_shortest[(edge_index_[1][i], edge_index_[0][i])] = dis

    return dis_shortest

