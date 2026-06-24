# Task 5 - converting a dataset I've used before into a PyG graph
#
# I picked the Iris dataset because I've used it before for basic classification
# (sklearn, logistic regression stuff from earlier). It's simple enough that I can
# understand what's happening when I convert it.
#
# The idea: treat each sample (flower) as a node, and connect two nodes with an
# edge if they're "close" to each other in feature space (k-nearest neighbors).
# So similar flowers will be connected.
#
# Why does this make sense as a graph?
# Because in plain kNN classification, we already implicitly say "you are what your
# neighbors are". Making it a graph and running a GNN on it is basically the same
# intuition but now the model can learn HOW to use neighbor info, not just vote.
# Also this lets us do semi-supervised stuff - label a few nodes and let info
# propagate through the graph to unlabeled ones.

import torch
from torch_geometric.data import Data
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np


# compute knn edges
# for each node find its k nearest neighbors and add edges to them
def make_knn_edges(X, k=5):
    n = X.shape[0]
    src = []
    dst = []

    for i in range(n):
        # compute euclidean distances from node i to all others
        dists = np.sum((X - X[i]) ** 2, axis=1)
        dists[i] = np.inf  # don't connect to self

        # get k closest
        nearest = np.argsort(dists)[:k]
        for j in nearest:
            src.append(i)
            dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


# load iris
iris = load_iris()
X_raw = iris.data    # shape (150, 4) - 150 flowers, 4 features each
y = iris.target      # 0=setosa, 1=versicolor, 2=virginica

# normalize features - important so distance calculations arent dominated by scale
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# build the graph
edge_index = make_knn_edges(X, k=5)

data = Data(
    x=torch.tensor(X, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(y, dtype=torch.long)
)

print("--- Iris as a PyG Graph ---")
print(f"num nodes: {data.num_nodes}")
print(f"num edges: {data.num_edges}")
print(f"node feature dim: {data.num_node_features}")
print(f"has self loops: {data.has_self_loops()}")
print()
print("first 5 node features:")
print(data.x[:5])
print()
print("first 10 edges (edge_index):")
print(data.edge_index[:, :10])

# simple train/test split - label 10 nodes per class for training
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
for c in range(3):
    class_nodes = (data.y == c).nonzero(as_tuple=True)[0]
    train_mask[class_nodes[:10]] = True

data.train_mask = train_mask
data.test_mask = ~train_mask

print(f"\ntrain nodes: {train_mask.sum().item()}")
print(f"test nodes: {data.test_mask.sum().item()}")

# the point of this is that with a GNN on this graph, a node labeled "setosa"
# can share that info with its neighbors, helping classify unlabeled ones nearby
# which a regular MLP would have no way of doing
print("\ndata:", data)
