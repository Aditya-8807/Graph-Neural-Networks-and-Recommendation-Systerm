# 🚀 Assignment for the Week 🚀

Welcome to your first deep dive into the world of Graph Machine Learning! This week is all about building rock-solid intuition, breaking open the abstractions, and seeing graphs everywhere.

Get your Python environments ready, and let's get tracking. 💻🔥

---

## 🛠️ The Deliverables

### 1️⃣ Message Passing from Scratch 🧩

* **The Task:** Implement a message passing algorithm using **nothing but PyTorch** (and `networkx` if you want, but honestly, you might not even need it if you simply take the Graph to be an adjacency matrix).
* **The Goal:** Solve everything solely through `torch`. Keep it simple! The simplest form of message passing is perfectly fine.
* A .py file for me please.

### 2️⃣ Demystifying Knowledge Graphs (KGs) 🕸️

* **The Task:** What exactly *are* knowledge graphs?
* **The Goal:** Write up a concise conceptual summary.
* *Pssst:* Maybe check out the *Graph Representation Learning (GRL)* book by William L. Hamilton for some crisp definitions!

### 3️⃣ The GNN Killers: Over-smoothing vs. Over-squashing 📉☠️

* **The Task:** Deep dive into these two infamous GNN bottlenecks.
* **The Questions:** * What are they?
* Why do they even happen in the first place?
* How can we solve them?
* *Hmm...* could Hamilton help? Or maybe this guy here?: 
[THE BOOK](https://link.springer.com/book/10.1007/978-981-16-6054-2)


* Explain the tension between stacking layers for a larger receptive field versus losing node distinctiveness.

<!--
```
  When you add the 10th GNN layer expecting a massive receptive field...
  [ Meme Idea: Over-smoothing ]
  "And... everyone is identical. All node embeddings are now the exact same vector." 
  🫠 🤝 🫠 🤝 🫠

```
-->

### 4️⃣ The Colab 🧪

* **The Task:** Open up, run, and implement this notebook if you haven't already done so:
👉 [PyTorch Geometric Node Classification Colab](https://colab.research.google.com/drive/1AaNEIaIZhRNMueJDdrnNLdwiYuwwfFP9) 📊

<!--
>Insert Meme of Sweating(It's not like I got no other questions, I just want you to relax)
-->

### 5️⃣ Paradigm Shift: Anything is a Graph 🌌

* **The Task:** The thing about graphs is that **anything is a graph**. 🧬📦🛣️
* **The Action:** Pick out a dataset you have *already used* for some other machine learning task (images, tabular data, text, text structures, whatever!) and convert it into a PyTorch Geometric (`PyG`) dataset.
* **The Reflection:** Think deeply about *why* thinking about this specific dataset as a graph would make more sense, and report your findings, by doing a task on it!

### 6️⃣ The NetworkX Warmup & C++ Nostalgia 🐍

* **The Task:** I have been getting into a bit of C++ lately, so I would have loved to have you guys code this up in C++... but sadly we don't do that here! 😂 So, let's go with everyone's close buddy: **Python**.
* **The Action:** Take the classic Karate Club graph from `networkx` (or any other graph data) and write a regular old `networkx` script to convert it into:
1. An **Adjacency Matrix** 🟦
2. An **Edge List** 📜



<!-- ```
  Me: "Can we code the graph data structures in modern, high-performance C++?"
  The Curriculum: "We have Python at home."
  Python at home: 🐍 🤝 "Hey buddy, let's write an edge list!"

``` -->

### 7️⃣ Build Your Own Graph Class (Week 1 Vibes) 🏛️

* **The Task:** Create a custom class in Python of your own for Graphs.
* **The Blueprint:** Think about what is fundamentally needed in it. Edge attributes? Node attributes? Add them explicitly.
* **Compare & Contrast:** *Then*, look at how `PyG` handles this under the hood.
* **⚠️ Caution:** Don't go too deep though!! `PyG` is built upon multiple underlying libraries and is essentially an abstraction over abstractions. Getting a solid grasp of the high level should be completely sufficient. This is kinda week 1 stuff, so keep your sanity intact! 😉

---

## 📬 Submission Guidelines

* Document your code properly in your fork or directory.
* For the conceptual questions, a clean markdown file or notebook cells detailing your answers will do perfectly.

Good luck, and happy hacking! Let's make everything a graph. 🎛️⚡