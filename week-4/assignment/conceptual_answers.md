# Week 4 - Conceptual Questions

---

## Task 2: What are Knowledge Graphs?

So from what I understood reading chapter 7 of the GRL book, a knowledge graph is basically a way to store facts about the real world in graph form.

The core idea is that everything is represented as **triples** - like three-tuples of the form:

```
(head, relation, tail)
```

for example: `(Paris, isCapitalOf, France)` or `(Eiffel Tower, locatedIn, Paris)`

- the **head and tail** are entities (nodes in the graph) - things like people, places, concepts
- the **relation** is a typed directed edge between them - it describes *how* they're connected

So its basically a directed graph where edges have types, not just weights. That's what makes it "multi-relational" - there are many different kinds of edges, not just one.

A few things I noted about KGs:
- they are **always incomplete** - you can never have all the facts, so one big task is predicting missing links (will this triple be true?)
- direction matters - `(A, isParentOf, B)` is NOT the same as `(B, isParentOf, A)`
- they can get enormous - Freebase apparently had like billions of triples

I think the most famous example is Google's Knowledge Graph which powers those info boxes you see on the right side when you search for something. Recommendation systems also use them a lot apparently.

The two main tasks people do on KGs are:
1. **Link prediction** - given (h, r, ?) predict the tail entity
2. **Entity alignment** - matching entities across two different KGs that refer to the same thing

---

## Task 3: Over-smoothing and Over-squashing

These are two problems that come up when you try to make GNNs deeper. I'll try to explain both.

### A bit of context first

Each GNN layer expands how far a node can "see" - after k layers a node has access to information from nodes k hops away. So naturally you'd think: more layers = better, right? Turns out no, for two different reasons.

---

### Over-smoothing

**What is it?**

If you stack too many GNN layers, all the node embeddings eventually become the same (or very similar). The model loses the ability to tell nodes apart.

**Why does it happen?**

Message passing is like averaging - each node takes the average of its neighbors' features. If you keep doing this over and over again across many layers, everyone ends up with kind of the same blended average value. Its like if you mixed paints together repeatedly - you eventually just get a muddy brown no matter what colors you started with.

More formally, the repeated multiplication with the normalized adjacency matrix makes all representations converge to the same subspace (something to do with the dominant eigenvector of the matrix, I don't fully get the math behind this yet).

**How to fix it:**

- Add **skip/residual connections** - like in ResNet, let the original features pass through unchanged so they're not completely overwritten
- **JK-Net** (Jumping Knowledge Networks) - connect all intermediate layer outputs to the final layer, not just the last one
- **DropEdge** - randomly drop edges during training so the averaging doesn't converge so fast
- honestly just not using too many layers (2-3 is usually enough for most tasks)

---

### Over-squashing

**What is it?**

This one is a bit harder to understand. The problem is that as you go deeper, the number of nodes a node needs to "receive messages from" grows exponentially (its all nodes within k hops). But those messages all have to get compressed into one fixed-size vector. So information gets "squashed".

**Why does it happen?**

Think about it - if a node has 10 neighbors and each of them has 10 neighbors, at 2 hops you're trying to fit info from 100 nodes into one vector. At 5 hops that's potentially 100,000 nodes. The fixed embedding size just can't hold all of that. So the gradients flowing back through those paths become tiny and distant nodes effectively have no influence.

The book I found ([this one](https://link.springer.com/book/10.1007/978-981-16-6054-2)) talks about this being related to graph curvature - basically nodes connected through narrow "bottleneck" edges suffer the most from squashing.

**How to fix it:**

- **Graph rewiring** - add extra edges to create shortcuts so information doesn't have to travel as far (DIGL does this)
- **Graph Attention Networks (GAT)** - let the model learn which neighbors to actually pay attention to
- **Graph Transformers** - just connect every node to every other node with attention, skip message passing entirely (this feels like cheating but apparently it works)

---

### The tension between depth and distinctiveness

This is basically the core dilemma:

- you WANT more layers to see more of the graph (bigger receptive field)
- but more layers → over-smoothing (nodes lose identity) AND over-squashing (can't compress enough info)

So you're stuck. If you go shallow, nodes can only see their immediate neighbors. If you go deep, all the embeddings collapse into the same thing.

The way most people deal with this is by separating the "how far does info travel" question from the "how many learned transformations" question. Like APPNP does propagation based on personalized pagerank (can go far) but only applies learned weights once at the start. That way you get a large receptive field without repeated smoothing.

I think graph transformers are also a response to this - since they use full attention they dont have the squashing/smoothing problem at all, but they're way more expensive computationally.

---

## Task 4: PyG Node Classification Colab

I ran the notebook linked in the assignment. It uses the **Cora** dataset which is a citation network (papers = nodes, citations = edges).

The notebook builds a 2-layer GCN using PyG's `GCNConv` and trains it on the semi-supervised node classification task. Only a small fraction of nodes have labels during training and the rest are predicted using the graph structure.

Main things I noticed:
- the Data object in PyG is really clean - everything (features, edges, labels, masks) lives in one object
- the train/val/test split is done using boolean masks on the node tensor, which is different from the usual dataset splitting I've seen
- GCN got decent accuracy on Cora pretty fast (around 80%) even with very few labeled nodes
- the t-SNE visualization at the end is cool - you can see the node embeddings cluster by class

---

*Note: answers for tasks 1, 5, 6, 7 are in the .py files in this folder*
