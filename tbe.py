from torch_geometric.nn import WLConv
import torch
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T
from time import time

def normalize_labels(labels):
    label_map = {}
    normalized_labels = torch.zeros_like(labels, device='cuda:0')
    next_label = 0
    for label in labels:
        if label not in label_map:
            label_map[label] = next_label
            next_label += 1
        normalized_labels[labels == label] = label_map[label]
    return normalized_labels

def partition_refinement_WL(data, prepartition, iters=float('inf')):
    """Performs partition refinement using 1-WL.
    Returns the coarsest stable refinement of prepartition or the
    refinement after iters steps"""
    wl = WLConv()
    x = prepartition
    x_norm = normalize_labels(x)
    #print('x', x)
    k = 0
    while True:
        x_new = wl(x, data.edge_index)
        k += 1
        print(k)
        x_new_norm = normalize_labels(x_new)
        if k==iters or torch.equal(x_norm,x_new_norm):
            return x_new_norm
        else:
            # reset the hashmap not to accumulate ids in memory
            wl.reset_parameters()
            x = x_new
            x_norm = x_new_norm

start = []
end = []
num_nodes = int(input())
edges = int(input())
for _ in range(edges):
    s,w,e = map(int, input().split(" "))
    start.append(s)
    end.append(e)

edge_index = torch.tensor([start,
                           end,
                           ], dtype=torch.long, device='cuda:0')
x = torch.ones(edge_index.shape[1], dtype=torch.float, device='cuda:0')
data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, device='cuda:0')
prepartition = torch.zeros(data.num_nodes, dtype=torch.long, device='cuda:0')
start = time()
wl_infty = partition_refinement_WL(data, prepartition)
end = time()

print(end - start)
print(1)
print(torch.unique(wl_infty).shape[0])
print(0)
