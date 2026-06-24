# Task 7 - Build my own Graph class
# I want to store nodes, edges, and their attributes
# keeping it simple, not going too deep into abstractions

# thought about what a graph actually needs:
#   - a way to store nodes (with optional features/labels)
#   - a way to store edges between nodes (with optional weights/attributes)
#   - some basic methods to query things (neighbors, etc.)


class Graph:
    # directed=False means undirected (default)
    def __init__(self, directed=False):
        self.directed = directed
        self.nodes = {}   # node_id -> dict of attributes
        self.edges = {}   # (src, dst) -> dict of attributes

    def add_node(self, node_id, **attrs):
        if node_id not in self.nodes:
            self.nodes[node_id] = {}
        self.nodes[node_id].update(attrs)

    def add_edge(self, u, v, **attrs):
        # add the nodes too if they don't exist already
        self.add_node(u)
        self.add_node(v)

        self.edges[(u, v)] = attrs
        if not self.directed:
            # for undirected graphs store both directions
            self.edges[(v, u)] = attrs

    def get_neighbors(self, node_id):
        neighbors = []
        for (u, v) in self.edges:
            if u == node_id:
                neighbors.append(v)
        return neighbors

    def get_node_attr(self, node_id):
        return self.nodes.get(node_id, {})

    def get_edge_attr(self, u, v):
        return self.edges.get((u, v), {})

    # convert to adjacency matrix (just a 2d list, not numpy)
    def to_adj_matrix(self):
        node_ids = sorted(self.nodes.keys())
        n = len(node_ids)
        # map node id to matrix index
        idx = {nid: i for i, nid in enumerate(node_ids)}

        mat = [[0]*n for _ in range(n)]
        for (u, v) in self.edges:
            mat[idx[u]][idx[v]] = 1
        return mat

    def __repr__(self):
        # count edges properly for undirected (each edge stored twice)
        num_edges = len(self.edges) if self.directed else len(self.edges) // 2
        return f"Graph(nodes={len(self.nodes)}, edges={num_edges}, directed={self.directed})"


# --- testing it out ---

g = Graph(directed=False)

# add some nodes with features and labels
g.add_node(0, feature=[1.0, 0.5], label="A")
g.add_node(1, feature=[0.3, 0.8], label="B")
g.add_node(2, feature=[0.9, 0.1], label="A")

# add edges with weights
g.add_edge(0, 1, weight=0.9)
g.add_edge(1, 2, weight=0.4)
g.add_edge(0, 2, weight=0.7)

print(g)
print("node 0 features:", g.get_node_attr(0))
print("neighbors of node 1:", g.get_neighbors(1))
print("edge (0,1) weight:", g.get_edge_attr(0, 1))
print("adjacency matrix:", g.to_adj_matrix())


# -----------------------------------------------------------------------
# How PyG does it vs how I did it
# -----------------------------------------------------------------------
#
# PyG uses torch_geometric.data.Data which looks like this:
#
#   from torch_geometric.data import Data
#   import torch
#
#   data = Data(
#       x = torch.tensor([[1.0, 0.5], [0.3, 0.8], [0.9, 0.1]]),   # node features [N, F]
#       edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]]),          # edges in COO format [2, E]
#       edge_attr = torch.tensor([[0.9], [0.4], [0.7]]),            # edge features [E, D]
#       y = torch.tensor([0, 1, 0])                                 # node labels
#   )
#
# The big difference is the edge_index format. Instead of storing edges as (u,v) tuples
# like I did, PyG stores them as a [2 x num_edges] tensor where row 0 is all source nodes
# and row 1 is all destination nodes. This is called COO (coordinate) format.
#
# Why? Because this format works really well with PyTorch's scatter operations
# and can run on GPU. My dict-based approach is fine for understanding but
# you can't really do GPU matrix ops on it.
#
# Also PyG handles batching automatically (multiple graphs into one big disconnected graph)
# which is super useful for training. My class doesn't do any of that.
#
# Basically PyG is a much more production-ready version of what I built here,
# but the core idea - nodes have features, edges connect them - is the same.
# -----------------------------------------------------------------------
