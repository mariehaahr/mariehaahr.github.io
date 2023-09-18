---
title: 'Exercises - Week 1'
description: 'Exercises for week 1 - NA'
pubDate: 'Sep 08 2023'
heroImage: 'pink-network.avif'
---

### Exercise 3.4
*Draw a correlation network for the given vectors, by only drawing edges with positive weights, ignoring self loops.*

Pay attention: This dataset is not a network, it is vectors, which we will turn into a correlation network eventually.

```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv("data.txt", sep = "\t")

# calculate the correlations, and replace them in the dataframe
df = df.corr()
# Take only positive correlations
df = df.unstack().reset_index()
df.columns = ("source", "target", "edge")
df = df[df["edge"]]

# Remove self loops
df = df[df["source"] != df["target"]]

# Make networkx object
G = nx.from_pandas_edgelist(df)

# Draw
nx.draw(G, with_labels = True)
plt.savefig("correlation_network.png")
plt.show()
```
The correlation network looks like:

<img src="/graph_ex1.png" alt="graph" width="400" height="250">

### Exercise 4.1 

*This network is bipartite. Identify the nodes in either type and find the nodes, in either type, with the most neighbors.*

```python
# Load the data
G2 = nx.read_edgelist("data2.txt")

# create the two bipartite sets of nodes
nodes = nx.algorithms.bipartite.basic.sets(G)

# Find the node in type 0 with the most neighbors
node_neighbors = {n: len(set(G.neighbors(n))) for n in nodes[0]}
maxnode = max(node_neighbors, key = node_neighbors.get)
print(maxnode, node_neighbors[maxnode])
```

<span style="color:grey;">131 4</span>

```python
# Find the node in type 1 with the most neighbors
node_neighbors = {n: len(set(G.neighbors(n))) for n in nodes[1]}
maxnode = max(node_neighbors, key = node_neighbors.get)
print(maxnode, node_neighbors[maxnode])
```

<span style="color:grey;">2 59</span>

### Exercise 4.4

*This network is dynamic, the third and fourth columns of the edge list tell you the first and last snapshot in which the edge was continuously present. An edge can reappear if the edge was present in two discontinuous time periods. Aggregate it using a disjoint window of size 3.*

```python
# Load the data. We need to import as multigraph, or networkx will collapse the edges.
# We also need to make sure to import the edge type information.
G = nx.read_edgelist("data3.txt", create_using = nx.MultiGraph(), data = [("start", int), ("end", int)])
# Since the window is disjoint and of size three, we need to group the 1-3, 4-6, and 7-9 snapshots.
first_window = nx.Graph()
second_window = nx.Graph()
third_window = nx.Graph()

# split the different time slots into the three graphs
for e in G.edges(data = True):
   if e[2]["start"] <= 3:
      first_window.add_edge(e[0], e[1])
   if e[2]["start"] <= 6 and e[2]["end"] > 3:
      second_window.add_edge(e[0], e[1])
   if e[2]["end"] > 6:
      third_window.add_edge(e[0], e[1])

print(first_window.nodes)
print(second_window.nodes)
print(third_window.nodes)
```

<span style="color:grey;">['1', '2', '3', '4', '5', '6', '7', '8', '9', '13', '12']</span>

<span style="color:grey;">['1', '2', '3', '4', '5', '7', '8', '9', '6', '12']</span>

<span style="color:grey;">['1', '2', '3', '5', '4', '6', '7', '8', '10', '11', '12']</span>

