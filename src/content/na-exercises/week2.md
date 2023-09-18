---
title: 'Exercises - Week 2'
description: 'Exercises for week 2 - NA'
pubDate: 'Sep 09 2023'
heroImage: 'pink-network.avif'
---

### 5.1  
*Calculate the adjacency matrix, the stochastic adjacency matrix, and the graph Laplacian for the given network*

```python
# import necessary libraries
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite

# Load the data
G1 = nx.read_edgelist("data_ex1.txt")

# turn into numpy array adjacency matrix
adj_mat = nx.to_numpy_array(G1)
```
**Converting an adjacency matrix** into a stochastic matrix is normalising it (dividing each entry by the sum of its corresponding row, each of the rows sums to one). 

```python
# stochastic adjacency matrix
sto_adj_mat = adj_mat / adj_mat.sum(axis=1)
```

**Laplacian Matrix** To calculate the laplacian matrix, you subtract the adjacency matrix from the degree matrix. L = D - A. Where the degree matrix D is a diagonal matrix, and that is a matrix whose nonzero values are exclusively on the main diagonal.

```python
nx.laplacian_matrix(G1)
```

### 5.1 

*Calculate the stochastic adjacency matrix of its projection. Project along the axis of size 248.*

**Projection of bipartite netwroks**: To transform a bipartite network into a uni-partite network, you need to perform the task of network projections. Bipartite network projection is an extensively used method for compressing information about bipartite networks. Since the one-mode projection is always less informative than the original bipartite graph, an appropriate method for weighting network connections is often required.

Here we have methods like:

1. Simple weighting
2. Hyperbolic weigthing
3. Resource allocation

**Remember:** The keys lies in recognising that not all edges have the same importance. Two movies that are watched by three common users are more related to each other than two movies that only have one common spectator. 

```python
# load the data as a bipartite network
G2 = nx.read_edgelist("data_ex2.txt", data = [("weight", float),])

# divide nodes into the two bipartite sets
nodes = nx.algorithms.bipartite.basic.sets(G2)

# turn the bipartite network into an adjacency matrix
adjmat = nx.algorithms.bipartite.matrix.biadjacency_matrix(G2, nodes[1])


```

**Simple weighting** Simple weigthing is a way of projecting a bipartite network into a unipartite. For each pair of nodes you identify the number of common neighbours they have, and that’s the weight og the edge. In term of matrices, this is equivalent to multiplying the bipartite adjacency matrix with its transpose.

```python
# we then use the simple weighting method
projected = adjmat.dot(adjmat.T)
projected.shape
```

<span style="color:grey;">(248, 248)</span>

*We make sure the shape of the matrix is 248x248, since thats the shape of the projected adjacency matrix.*

### 5.3

*Calculate the eigenvalues and the right and left eigenvectors of the stochastic adjacency of this bipartite network, using the same procedure applied in Exercise 5.2. Make sure to sort the eigenvalues in descending order (and sort the eigenvectors accordingly). Only take the real part of eigenvalues and eigenvectors, ignoring the imaginary part.*

**Eigenvalues and eigenvectors:** There is n eigenvalues in a matrix of n rows. In a stochastic adjacency matrix the largest eigenvalue of the n eigenvalues, will always be 1.

**Eigenvalues equal to 1**: If there are two eigenvalues that are equal to 1, there are two components in the network.

```python
# read the data as bipartite, and do the same as last exercise
G3 = nx.read_edgelist("data_ex3.txt", data = [("weight", float),])

# divide nodes into the two bipartite sets
nodes_G3 = nx.algorithms.bipartite.basic.sets(G3)

# turn the bipartite network into an adjacency matrix
adjmatG3 = nx.algorithms.bipartite.matrix.biadjacency_matrix(G3, nodes[1])
 
# do the same as in previous ex
projectedG3 = adjmatG3.dot(adjmatG3.T)

adjmat_proj_stoc_G3 = projectedG3 / projectedG3.sum(axis = 1)

# now we calculate the Eigenvalues and Eigenvectors. 
# Right eigenvectors first.
values1, vectors_r = np.linalg.eig(adjmat_proj_stoc_G3.todense())
# We need to sort the eigenvalues and eigenvectors, 
# since numpy returns them in random order
sorted_index = values1.argsort()[::-1]
values = np.real(values1[sorted_index])
vectors_r = np.real(vectors_r[:,sorted_index])
print(values)
print(vectors_r)
```

Output:

<img src="/vector.png" alt="graph" width="400" height="250">

```python
# Now for the left ones:
values, vectors_l = np.linalg.eig(adjmat_proj_stoc_G3.todense().T)
sorted_index = values.argsort()[::-1]
values = np.real(values[sorted_index])
vectors_l = np.real(vectors_l[:,sorted_index])
print(vectors_l)
``` 

Output

<img src="/vector2.png" alt="graph" width="400" height="250">

### 7.3
*What is the average reciprocity in the network used in Exercise 7.2? How many nodes have a reciprocity of zero?*

**Reciprocity**:
In a social network, it is interesting to know the probability that, if
I consider you my friend, you also consider me your friend – which
hopefully is 100%, but it rarely is so. This is an important quantity
in network analysis, and we give it a name. We call it reciprocity,
because it is all about reciprocating connections.
To calculate reciprocity we count the number of connected pairs of
the network: pairs of nodes with at least one edge between them. Then we count the number
of connected pairs that have both possible edges between them. The ones reciprocating the connection.
Reciprocity is simply the second count over the first one.

```python
# create the directed network
G73 = nx.read_edgelist("data_ex73.txt", create_using=nx.DiGraph())
# Overall reciprocity
print(nx.reciprocity(G73))
```

<span style="color:grey;">0.9720337038129897</span>

```python
# The node by node reciprocity
# get the list of nodes
nodes_73 = list(G73.nodes())

# node by node reciprocity
reci_dict = nx.reciprocity(G73, nodes = list(G73.nodes))

# count how many 0 reciprocity
counter = 0
for i in reci_dict.values():
    if i == 0:
        counter +=1
print(counter)
```

<span style="color:grey;">117</span>

### 7.4

*How many weakly and strongly connected components does the network used in Exercise 7.2 have? Compare their sizes, in number of nodes, with the entire network. Which nodes are in these two components?*

```python
# Load the data
G = nx.read_edgelist("data_ex74.txt", create_using = nx.DiGraph())

# Weak connectivity
wccs = list(nx.weakly_connected_components(G))
print("# Weak connected components: %s" % len(wccs))
wccs_largest = max(wccs, key = len)
print("%% Nodes in largest WCC: %1.2f%%" % (100 * len(wccs_largest) / len(G.nodes)))
print("Nodes in largest WCC: %s" % ' '.join(wccs_largest))
```


<span style="color:grey;">Weak connected components: 11</span>

<span style="color:grey;">% Nodes in largest WCC: 98.84%</span>

<span style="color:grey;">Nodes in largest WCC: 1701 2657 2201 1846 2843 ... </span>


```python
# Strong connectivity
sccs = list(nx.strongly_connected_components(G))
print("# Strong connected components: %s" % len(sccs))
sccs_largest = max(sccs, key = len)
print("%% Nodes in largest SCC: %1.2f%%" % (100 * len(sccs_largest) / len(G.nodes)))
print("Nodes in largest SCC: %s" % ' '.join(sccs_largest))
```

<span style="color:grey;">Strong connected components: 52</span>

<span style="color:grey;">% Nodes in largest SCC: 97.58%</span>

<span style="color:grey;">Nodes in largest SCC: 1701 2657 1846 2201 2843</span>