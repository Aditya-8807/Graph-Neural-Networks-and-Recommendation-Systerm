# Week 5 GCN Assignment Implementation Plan

> **For agentic workers:** This is a tutoring/content plan, not a software feature — execute inline, conversationally, one question at a time with the user (per user's explicit choice: "explain then I write, you review"). Do not dispatch subagents or write unit tests for markdown proofs; "tests" here mean running notebook cells and checking numeric/plot output against known-correct values.

**Goal:** Complete Week 5's four GCN questions (adjacency normalization, NumPy GCN layer, spectral self-loop proof, over-smoothing on Cora) as `week-5/assignment/Answers.md` + `week-5/assignment/Week5_GCNs.ipynb`, written in the user's own voice, with the user understanding each answer before it's final.

**Architecture:** Two deliverable files. Math/conceptual answers go in Answers.md; all runnable code goes in one Jupyter notebook, run top-to-bottom before commit so outputs are visible on GitHub.

**Tech Stack:** NumPy, NetworkX, PyTorch, PyTorch Geometric (Cora via `torch_geometric.datasets.Planetoid`), Matplotlib. `nn` conda env.

## Global Constraints
- Notebook must be fully executed (all cells run, outputs saved) before commit.
- Code and prose should read in the user's own informal first-person voice (see `week-4/assignment/conceptual_answers.md` and `message_passing.py` for tone reference).
- User must confirm understanding of each answer before it's committed.
- `torch_geometric` install happens once, in the `nn` conda env.

---

### Task 1: Environment setup — install torch_geometric

**Files:** none (environment only)

- [ ] Install into `nn` env: `/opt/anaconda3/envs/nn/bin/python -m pip install torch_geometric`
- [ ] Verify: `/opt/anaconda3/envs/nn/bin/python -c "import torch_geometric; print(torch_geometric.__version__)"`
  Expected: prints a version string (e.g. `2.6.1`), no ImportError.
- [ ] Create `week-5/assignment/` directory.

### Task 2: Q1 — Normalizing the adjacency matrix (Answers.md)

**Files:**
- Create: `week-5/assignment/Answers.md`

**Content requirements:**
- 3-node path graph (1-2-3): write out $A$ (3x3, edges 1-2 and 2-3) and $D$ (degree matrix: deg 1,2,1).
- Add self-loops: $\tilde{A} = A + I_3$, $\tilde{D}$ (degrees become 2,3,2).
- Compute $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ by hand, show the arithmetic, give the final 3x3 matrix with numeric entries.
- Conceptual answer: explain that $\hat{A}_{ij} = 1/\sqrt{\tilde{d}_i \tilde{d}_j}$, so a high-degree neighbor's message gets down-weighted relative to a low-degree neighbor's — this stops hub nodes from dominating aggregation just because they have more edges, which is a reasonable inductive bias (per-neighbor-importance shouldn't scale with how popular the neighbor is).

- [ ] **Step 1:** Walk the user through the by-hand computation verbally first (explain what $D^{-1/2}$ does to each row/col), confirm they follow.
- [ ] **Step 2:** Write the answer into `week-5/assignment/Answers.md` under a `## Q1` heading, in the user's voice.
- [ ] **Step 3:** Cross-check the hand computation numerically:
  ```python
  import numpy as np
  A = np.array([[0,1,0],[1,0,1],[0,1,0]])
  A_tilde = A + np.eye(3)
  D_tilde = np.diag(A_tilde.sum(axis=1))
  D_inv_sqrt = np.diag(1/np.sqrt(np.diag(D_tilde)))
  A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
  print(A_hat)
  ```
  Run: `/opt/anaconda3/envs/nn/bin/python -c "..."` (inline check, not part of notebook)
  Expected: matches the by-hand matrix in Answers.md (symmetric, diagonal entries 0.5, off-diagonal ~0.408).
- [ ] **Step 4:** Get user confirmation they understand the derivation before moving on.

### Task 3: Q2.1 — NumPy GCN layer (notebook)

**Files:**
- Create: `week-5/assignment/Week5_GCNs.ipynb`

**Interfaces:**
- Produces: `gcn_layer(A_hat, X, W)` function used only within this notebook (no cross-file reuse needed).

- [ ] **Step 1:** Explain the forward pass $H = \text{ReLU}(\hat{A}XW)$ conceptually — aggregate neighbor features via $\hat{A}$, then linearly transform via $W$, then nonlinearity.
- [ ] **Step 2:** Write the notebook cell (NumPy only):
  ```python
  import numpy as np

  def relu(x):
      return np.maximum(0, x)

  def gcn_layer(A_hat, X, W):
      return relu(A_hat @ X @ W)

  # graph from Q1
  A = np.array([[0,1,0],[1,0,1],[0,1,0]])
  A_tilde = A + np.eye(3)
  D_tilde = np.diag(A_tilde.sum(axis=1))
  D_inv_sqrt = np.diag(1/np.sqrt(np.diag(D_tilde)))
  A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

  np.random.seed(0)
  X = np.random.randn(3, 4)   # 3 nodes, 4 input features
  W = np.random.randn(4, 2)   # project to 2 output features

  H = gcn_layer(A_hat, X, W)
  print(H)
  ```
- [ ] **Step 3:** Run the cell in the notebook (via Jupyter kernel using `nn` env). Expected: prints a 3x2 matrix, all entries >= 0 (ReLU applied).
- [ ] **Step 4:** Q2.2 conceptual: explain in Answers.md that when $N \gg F$, compute $\hat{A}(XW)$ first — $XW$ is $N \times F'$ (cheap: $O(NFF')$), then $\hat{A}(XW)$ is $O(N^2 F')$; whereas $(\hat{A}X)W$ forces materializing an $N \times F$ intermediate multiplied by dense $\hat{A}$ first at $O(N^2F)$ — same asymptotic order but doing the small matrix multiply ($W$) first minimizes the width carried through the expensive $O(N^2 \cdot \cdot)$ step. Write this into Answers.md under `## Q2`.
- [ ] **Step 5:** Get user confirmation before moving on.

### Task 4: Q3.1 — Eigenvalue proof (Answers.md) and Q3.2 — numerical check (notebook)

**Files:**
- Modify: `week-5/assignment/Answers.md` (add `## Q3` section)
- Modify: `week-5/assignment/Week5_GCNs.ipynb` (add cell)

- [ ] **Step 1:** Explain the proof approach: $L = I_N - D^{-1/2}AD^{-1/2}$ has eigenvalues in $[0,2]$ (given). If $\lambda$ is an eigenvalue of $D^{-1/2}AD^{-1/2}$ with eigenvector $v$, then $Lv = (1-\lambda)v$, so $1-\lambda$ is an eigenvalue of $L$, so $1-\lambda \in [0,2]$, so $\lambda \in [-1,1]$.
- [ ] **Step 2:** Write the proof into Answers.md, in the user's voice, showing the algebra explicitly.
- [ ] **Step 3:** Write the Q3.3 conceptual answer (what's lost without self-loops): without $I_N$ added to $A$, a node's own features never get included in its own aggregation — $H = \hat{A}XW$ would only mix in neighbors' features and completely discard the node's own signal at every layer, which is a real information loss (the node's identity/original features vanish after one layer).
- [ ] **Step 4:** Add notebook cell for Q3.2 numerical check:
  ```python
  import networkx as nx

  G = nx.karate_club_graph()
  A = nx.to_numpy_array(G)
  D = np.diag(A.sum(axis=1))
  D_inv_sqrt = np.diag(1/np.sqrt(np.diag(D)))
  norm_no_selfloop = D_inv_sqrt @ A @ D_inv_sqrt

  A_tilde = A + np.eye(A.shape[0])
  D_tilde = np.diag(A_tilde.sum(axis=1))
  D_tilde_inv_sqrt = np.diag(1/np.sqrt(np.diag(D_tilde)))
  renormalized = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

  eig_unnorm = np.linalg.eigvalsh(np.eye(A.shape[0]) + norm_no_selfloop)
  eig_renorm = np.linalg.eigvalsh(renormalized)

  print("I + D^-1/2 A D^-1/2 range:", eig_unnorm.min(), eig_unnorm.max())
  print("Renormalized range:", eig_renorm.min(), eig_renorm.max())
  ```
- [ ] **Step 5:** Run the cell. Expected: first range extends toward/above ~2 (unstable), second range stays within roughly [0, 1] (confirming the paper's claim).
- [ ] **Step 6:** Get user confirmation before moving on.

### Task 5: Q4 — Over-smoothing on Cora (notebook)

**Files:**
- Modify: `week-5/assignment/Week5_GCNs.ipynb` (add cells)
- Modify: `week-5/assignment/Answers.md` (add `## Q4` conceptual section)

**Interfaces:**
- Consumes: `torch_geometric.datasets.Planetoid` (Task 1 install), `torch_geometric.nn.GCNConv`.

- [ ] **Step 1:** Explain over-smoothing conceptually before coding: each layer averages a node with its neighbors; stack enough layers and every node's receptive field covers most of the graph, so embeddings converge toward the same value — the model loses the ability to distinguish nodes, which tanks classification.
- [ ] **Step 2:** Add cell to load Cora:
  ```python
  from torch_geometric.datasets import Planetoid
  dataset = Planetoid(root='./data/Cora', name='Cora')
  data = dataset[0]
  ```
- [ ] **Step 3:** Add cell defining a configurable-depth GCN and train/eval loop:
  ```python
  import torch
  import torch.nn.functional as F
  from torch_geometric.nn import GCNConv

  class GCN(torch.nn.Module):
      def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
          super().__init__()
          self.convs = torch.nn.ModuleList()
          if num_layers == 1:
              self.convs.append(GCNConv(in_channels, out_channels))
          else:
              self.convs.append(GCNConv(in_channels, hidden_channels))
              for _ in range(num_layers - 2):
                  self.convs.append(GCNConv(hidden_channels, hidden_channels))
              self.convs.append(GCNConv(hidden_channels, out_channels))

      def forward(self, x, edge_index):
          for conv in self.convs[:-1]:
              x = F.relu(conv(x, edge_index))
          x = self.convs[-1](x, edge_index)
          return x

  def train_and_eval(num_layers, epochs=200, lr=0.01, hidden=16):
      model = GCN(dataset.num_features, hidden, dataset.num_classes, num_layers)
      optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
      for epoch in range(epochs):
          model.train()
          optimizer.zero_grad()
          out = model(data.x, data.edge_index)
          loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
          loss.backward()
          optimizer.step()
      model.eval()
      pred = model(data.x, data.edge_index).argmax(dim=1)
      correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
      acc = int(correct) / int(data.test_mask.sum())
      return acc
  ```
- [ ] **Step 4:** Add cell that runs it for K = 2, 4, 8, 16 and plots:
  ```python
  import matplotlib.pyplot as plt

  Ks = [2, 4, 8, 16]
  accs = [train_and_eval(k) for k in Ks]
  print(list(zip(Ks, accs)))

  plt.plot(Ks, accs, marker='o')
  plt.xlabel('Number of GCN layers (K)')
  plt.ylabel('Test accuracy')
  plt.title('Over-smoothing: test accuracy vs depth on Cora')
  plt.savefig('oversmoothing_plot.png')
  plt.show()
  ```
- [ ] **Step 5:** Run all cells. Expected: accuracy for K=2 highest (~0.80), decreasing noticeably by K=16 (over-smoothing visible). If K=2 isn't the best or the drop isn't visible, that's a real experimental finding — do not fudge; report what actually happened and discuss why in the conceptual answer.
- [ ] **Step 6:** Write the Q4 conceptual answer in Answers.md referencing the actual observed numbers.
- [ ] **Step 7:** Get user confirmation before moving on.

### Task 6: Finalize, run notebook top-to-bottom, commit

**Files:**
- Modify: `week-5/assignment/Week5_GCNs.ipynb`
- Modify: `week-5/assignment/Answers.md`

- [ ] **Step 1:** Restart kernel and run all cells top-to-bottom (Restart & Run All) so outputs are fresh and in order.
- [ ] **Step 2:** Verify Answers.md has all four `##` sections (Q1-Q4) and reads in the user's voice.
- [ ] **Step 3:** Remind user of the manual step: upload notebook to Google Colab, set sharing to "Anyone with the link can view", add the link to the top of `week-5/README.md` (not automatable — needs the user's Google account).
- [ ] **Step 4:** Commit:
  ```bash
  git add week-5/assignment/Answers.md week-5/assignment/Week5_GCNs.ipynb
  git commit -m "Complete week-5 GCN assignment (Q1-Q4)"
  ```
