# Task 6 - NetworkX warmup
# converting the karate club graph to adjacency matrix and edge list
# karate club is like the hello world of graph datasets apparently

import networkx as nx

# load the graph
G = nx.karate_club_graph()

print(f"nodes: {G.number_of_nodes()}")
print(f"edges: {G.number_of_edges()}")
print()

# --- 1. Adjacency Matrix ---
# networkx has a built in function for this which is nice
# weight=None means treat all edges as 1 (binary), otherwise it uses edge weights
adj = nx.to_numpy_array(G, weight=None, dtype=int)

print("Adjacency Matrix shape:", adj.shape)
print("showing just the top left 8x8 corner (34x34 is too big to print):")
print(adj[:8, :8])
print()

# just verifying its symmetric (undirected graph so it should be)
# print("is symmetric:", np.array_equal(adj, adj.T))  # yeah it is


# --- 2. Edge List ---
edges = list(G.edges())
print(f"Edge list ({len(edges)} edges total):")
for u, v in edges:
    print(f"  {u} -- {v}")

# tried doing it manually at first with a nested loop but that was way too slow
# networkx's G.edges() is just cleaner

# bonus: converting back from edge list to a graph just to see if it works
G2 = nx.from_edgelist(edges)
print(f"\nreconstructed graph has {G2.number_of_edges()} edges - matches original: {G.number_of_edges() == G2.number_of_edges()}")
