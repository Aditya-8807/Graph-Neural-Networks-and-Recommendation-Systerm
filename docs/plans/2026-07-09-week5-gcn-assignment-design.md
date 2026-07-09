# Week 5 Assignment Design — GCNs & Spectral Graph Theory

## Goal
Complete `week-5/Assignment/Assignments.md` (Q1-Q4 on GCN normalization, spectral
proofs, and over-smoothing), in the user's own voice, with the user understanding
every line before it's final (per the assignment's honor-code note).

## Deliverables
- `week-5/assignment/Answers.md` — math/conceptual write-ups:
  - Q1: adjacency/degree matrices for the 3-node path graph, self-loop versions,
    symmetric-normalized $\hat{A}$ computed by hand, conceptual answer on
    degree weighting.
  - Q2.2: conceptual answer on multiplication order $(\hat{A}X)W$ vs $\hat{A}(XW)$.
  - Q3.1: proof that eigenvalues of $D^{-1/2}AD^{-1/2}$ lie in $[-1,1]$ via the
    normalized Laplacian.
  - Q3.3: conceptual answer on what's lost without self-loops.
- `week-5/assignment/Week5_GCNs.ipynb` — code:
  - Q2.1: NumPy-only GCN layer, tested on the Q1 graph with random $X$, $W$.
  - Q3.2: eigenvalue comparison ($I_N + D^{-1/2}AD^{-1/2}$ vs renormalized) on
    `networkx.karate_club_graph()`.
  - Q4: PyTorch Geometric GCN on Cora, K = 2/4/8/16, test-accuracy-vs-K plot
    (over-smoothing demo).
- Notebook run top-to-bottom before commit so outputs/plots are visible in GitHub.
- User uploads the notebook to Colab and adds the public view link to
  `week-5/README.md` (manual step, not automatable).

## Naming deviation
The assignment doc says to name the notebook `Week4_GCNs.ipynb` — a leftover
copy-paste from last week's template upstream (confirmed via upstream git log:
the whole file's title is "Week 4" despite living in the `week-5` folder). Using
`Week5_GCNs.ipynb` instead, matching the folder convention used in weeks 1-4.

## Environment
`torch_geometric` is not installed anywhere yet. Installing into the `nn` conda
env (already has torch 2.2.2 + torchvision/torchaudio, looks like the course env).

## Process
Tutor-style, question by question: explain the math/reasoning first, then write
the answer/code in the user's voice (informal, first-person, matching week-4
style), user confirms understanding before moving to the next question.
