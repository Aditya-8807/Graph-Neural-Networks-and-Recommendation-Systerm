# Task 1 - implementing message passing from scratch using just pytorch
# okay so the idea is pretty simple - each node collects info from its neighbors
# and then transforms it. thats basically it i think.

# i'm using an adjacency matrix to represent the graph
# rows and cols are nodes, entry is 1 if there's an edge between them

import torch
import torch.nn as nn
import torch.nn.functional as F


# normalize the adjacency so we get a mean of neighbors instead of a sum
# (if we dont do this the values explode with high degree nodes)
def normalize_adj(A):
    degree = A.sum(dim=1, keepdim=True)
    degree[degree == 0] = 1  # avoid divide by zero for isolated nodes
    A_norm = A / degree
    return A_norm


# one layer of message passing
# step 1: aggregate (multiply adj matrix with features, gives avg neighbor features)
# step 2: apply linear layer + relu
class MPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MPLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, A_norm, X):
        # aggregate neighbor features - this is the "message passing" part
        aggregated = torch.mm(A_norm, X)
        # transform and activate
        out = F.relu(self.linear(aggregated))
        return out


# two layer gnn - stack two mp layers
class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(SimpleGNN, self).__init__()
        self.layer1 = MPLayer(in_dim, hidden_dim)
        self.layer2 = MPLayer(hidden_dim, num_classes)

    def forward(self, A_norm, X):
        h = self.layer1(A_norm, X)
        out = self.layer2(A_norm, h)
        return out


# --- main ---

torch.manual_seed(42)

# small toy graph with 6 nodes
# edges: 0-1, 0-2, 1-2, 1-3, 3-4, 4-5
num_nodes = 6
edge_list = [(0,1), (0,2), (1,2), (1,3), (3,4), (4,5)]

# build adjacency matrix manually
A = torch.zeros(num_nodes, num_nodes)
for u, v in edge_list:
    A[u][v] = 1.0
    A[v][u] = 1.0  # undirected so both directions

print("Adjacency matrix:")
print(A)
print()

A_norm = normalize_adj(A)

# random node features, each node has 4 features
X = torch.randn(num_nodes, 4)

# labels - first 3 nodes are class 0, last 3 are class 1
# just a simple binary task
labels = torch.tensor([0, 0, 0, 1, 1, 1])

model = SimpleGNN(in_dim=4, hidden_dim=8, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Starting training...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    logits = model(A_norm, X)
    loss = F.cross_entropy(logits, labels)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        acc = correct / num_nodes
        print(f"epoch {epoch+1} | loss: {loss.item():.4f} | acc: {acc:.2f}")

# look at what the node embeddings look like after training
print("\nnode embeddings after training (from layer 1):")
model.eval()
with torch.no_grad():
    embeddings = model.layer1(A_norm, X)
    print(embeddings)

# quick sanity check - nodes in same class should have similar embeddings
# hard to tell just by looking but at least it runs lol
