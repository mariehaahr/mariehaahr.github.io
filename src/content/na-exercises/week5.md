---
title: 'Exercises - Week 5'
description: 'Exercises for week 5 - NA'
pubDate: 'Sep 25 2023'
heroImage: 'pink-network.avif'
---

<style>
   .social-links,
   .social-links a{
   display: center;
   color: black;
   text-align: center;
   }
   .social-links a:hover{
      text-align: center;
      color: hotpink;
   }
   @media (max-width: 720px) {
   .social-links {text-align: center; display: none; }
   }
</style>

##### <center>  Exercise notebook with tips </center>

<div class="social-links">
   <a href="https://github.com/mariehaahr/Network-Analysis-Hints/tree/main/week-5/tips" target="_blank">
      <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 489 510">
         <path fill="currentcolor" d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg>
      </a>
</div>

## 23.1

*Perform a network projection of this bipartite network using simple weights. The unipartite projection should only contain nodes of type 1 (|V1| = 248). How dense is the projection?*

**Recall:** In order to create this weighted link, you simply multiply the bipartite adjacency matrix with its transpose. You have to decide which set of nodes you want to project onto. 

```python
import random
import pandas as pd
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt
from collections import Counter

# Load the data
G = nx.read_edgelist("../data/23_1.txt")

# Let's figure out which node is of which type
# Tip: use nx.algorithms.bipartite.basics.sets
nodes = nx.algorithms.bipartite.basic.sets(G)
print(nodes)
rows = sorted(list(nodes[0]))
cols = sorted(list(nodes[1]))

# We'll project on nodes of type a
if "a1" in rows:
   nodes = rows
else:
   nodes = cols

# Let's get the bipartite adjacency matrix
# Tip: use nx.algorithms.bipartite.matrix.biadjacency_matrix
T = nx.algorithms.bipartite.matrix.biadjacency_matrix(G, row_order = nodes)

# If we multiply the bipartite adjacency matrix with its transpose we obtain the number of
# common ones between two rows, i.e. the number of common neighbors between the nodes.
U = T * T.T

# We set the diagonal to zero because we don't care about self loops
# Tip: use setdiag() and eliminate_zeros()
U.setdiag(0)
U.eliminate_zeros()

# We get the projected graph back. ( nx.from_scipy_sparse_matrix)
# We relabel to recover the original node IDs (cab be done, but is not necessary)
G = nx.from_scipy_sparse_matrix(U)
G = nx.relabel_nodes(G, {i: nodes[i] for i in range(len(nodes))})

# Graph is super dense! 40%
print(nx.density(G))
```

<span style="color:grey;">0.4097231291628575</span>

## 23.3

*Perform a network projection of the bipartite network used for Exercise 23.1 using hyperbolic weights. Draw a scatter plot comparing hyperbolic and simple weights.*

**Recall:** This method reminds of wimple weight, but here it exaggerates the differences, so that thresholding becomes easier (thresholding, as in establishing a threshold and drop the edges below this minimum acceptable weight).

Each common neighbour $z$ contributes $k_z^{-1}$ to the weight in the projection.

```python
#  Make the code from the previous question into its own function
# Tip: Should take a network and a list of nodes as input and return a network
def simple(network, nodes):
   T = nx.algorithms.bipartite.matrix.biadjacency_matrix(network, row_order = nodes)
   U = T * T.T
   U.setdiag(0)
   U.eliminate_zeros()
   G = nx.from_scipy_sparse_matrix(U)
   return nx.relabel_nodes(G, {i: nodes[i] for i in range(len(nodes))})

# Function calculating hyperbolic weights. basically, we just need to normalize
# the bipartite adjacency matrix by its degree, because each neighbo now counts
# not as one, but as one over the degree. Then the rest is as before.

def hyper(network, nodes):
   T = nx.algorithms.bipartite.matrix.biadjacency_matrix(network, row_order = nodes)
   T /= T.sum(axis = 0)
   T = sparse.csr_matrix(T)
   U = T * T.T
   U.setdiag(0)
   U.eliminate_zeros()
   G = nx.from_scipy_sparse_matrix(U)
   return nx.relabel_nodes(G, {i: nodes[i] for i in range(len(nodes))})

# Load the data and get the correct node list

# Tip: The data is in the format of an edgelist
# Tip 2: Remember to sort the nodes
# Tip 3: Use nx.algorithms.bipartite.basic.sets to get the nodes

G = nx.read_edgelist("../data/23_1.txt")
nodes = nx.algorithms.bipartite.basic.sets(G)
rows = sorted(list(nodes[0]))
cols = sorted(list(nodes[1]))
if "a1" in rows:
   nodes = rows
else:
   nodes = cols


# Use the functions to project the networks (simple and hyperbolic) 
# Tip: Get all edges from both networks for the plotting
G_simple = simple(G, nodes)
G_hyper = hyper(G, nodes)
all_edges = set(G_simple.edges) | set(G_hyper.edges)

# Make a dataframe with the edges and the weights from the two networks
# Tip: You can use a list comperhension to do this
# Tip 2: The columns should be something like (edge, simple, hyperbolic)
df = pd.DataFrame(data = [(e, G_simple.edges[e]["weight"], G_hyper.edges[e]["weight"]) for e in all_edges], columns = ("edge", "simple", "hyper"))

# plot a scatter plot of the two weights 
# Tip: x should be the simple weights and y the hyperbolic weights
df.plot(kind = "scatter", x = "simple", y = "hyper", color = "#e41a1c")
plt.show()
```

<img src="/ex23_3.png" alt="dd" width="600" height="450">


## 24.1

*Plot the CCDF edge weight distribution of this network. Calculate its average and standard deviation. NOTE: this is a directed graph!*

```python
# Load the data
# Tip: The data is in the form of an edgelist
# Tip 2: The arguments data=[("weight", float),] should be used to load the weights
# Tip 3 : The argument create_using=nx.DiGraph() should be used to load the data as a directed graph
G = nx.read_edgelist("../data/24_1.txt", data = [("weight", float),], create_using = nx.DiGraph())

# Get the weights
# Tip: nx.get_edge_attributes to get the weights attribute
# Tip 2 This returns a dictionary, so use list(dict.values()) to get the values

edgeweights = list(dict(nx.get_edge_attributes(G, "weight")).values())

# Let's use our degree CCDF code for the edge weights
# Tip Same aproach as the privious CCDF exercises but now with the edge weights instead of the degrees

wd = pd.DataFrame(list(Counter(edgeweights).items()), columns = ("weight", "count")).sort_values(by = "weight")
ccdf = wd.sort_values(by = "weight", ascending = False)
ccdf["cumsum"] = ccdf["count"].cumsum()
ccdf["ccdf"] = ccdf["cumsum"] / ccdf["count"].sum()
ccdf = ccdf[["weight", "ccdf"]].sort_values(by = "weight")
ccdf.plot(kind = "line", x = "weight", y = "ccdf", color = "#e41a1c", loglog = True)
plt.show()
```

<img src="/ex24_1.png" alt="dd" width="600" height="450">

```python
 # Average & stdev of the weights
# Tip: Use numpy to calculate the mean and standard deviation of the weights
edgeweights = np.array(edgeweights)
print("Edge weight average: %1.4f" % np.mean(edgeweights))
print("Edge weight stdev: %1.4f" % np.std(edgeweights))
```

<span style="color:grey;">Edge weight average: 17191.3325</span>  
<span style="color:grey;">Edge weight stdev: 149924.3810</span>

## 24.3

*Can you calculate the doubly stochastic adjacency matrix of the network from Exercise 24.1? Does the calculation eventually converge? (Limit the normalization attempts to 1,000. If by 1,000 normalizations you don't have a doubly stochastic matrix, the calculation didn't converge)*

```python
# Load the data
# Tip: The data is in the form of an edgelist
# Tip 2: The arguments data=[("weight", float),] should be used to load the weights
# Tip 3 : The argument create_using=nx.DiGraph() should be used to load the data as a directed graph
G = nx.read_edgelist("../data/24_1.txt", data = [("weight", float),], create_using = nx.DiGraph())

# Let's start by getting the adjacency matrix
# Tip: Use the function nx.to_numpy_matrix()
A = nx.to_numpy_matrix(G)

# If we want to get the doubly stochastic, we need to alternatively
# normalize by row and column sum. We stop only when the deviation
# from one is very little, thus the row/column sum is very close to
# one. We also need to keep track of how many times we performed
# the normalization. If we keep going back and forth between the
# same two values, it means we're not converging.

# Tip: Do a while loop that checks if the standard deviation of the 1 axis sums is greater than 1e-12
# Tip 2: Normalize by row sum and by column sum
# Tip 3: Remember to keep track of the number of attempts ( if greater than 1000 then break)
attempts = 0
row_sums = A.sum(axis = 1)
while np.std(A.sum(axis = 1)) > 1e-12:
   A /= A.sum(axis = 1)
   A /= A.sum(axis = 0)
   attempts += 1
   if attempts > 1000:
      print("Calculation didn't converge. The matrix cannot be made doubly stochastic. Aborting.")
      break

print("Calculation converged. Here's A:")
print(A)

# YES! Calculation converges! Phew!
```
<span style="color:grey;">Calculation converged. Here's A:</span>  
<span style="color:grey;">[[0.00000000e+00 9.29197199e-07 9.75734853e-05 ... 0.00000000e+00</span>  
<span style="color:grey;">0.00000000e+00 0.00000000e+00]</span>  
<span style="color:grey;">[0.00000000e+00 0.00000000e+00 8.73760713e-04 ... 0.00000000e+00</span>  
...

## 25.1

*Perform a random walk sampling of this network. Sample 2,000 nodes (1% of the network) and all their connections (note: the sample will end up having more than 2,000 nodes).*

```python
# Load data
# Tip: The data is in the format of edgelist
# Tip 2: The node type is int (nodetype=int)
G = nx.read_edgelist("../data/25_1.txt", nodetype = int)
# We need to keep track of the nodes and the edges we have sampled, so we store them in sets.
# Then we need a seed node where to start the exploration. Here I decided to just pick a random node.
sampled_nodes = set()
sampled_edges = set()
curnode = random.choice(list(G.nodes))

# We continue until we sampled 2000 nodes.
while len(sampled_nodes) <= 2000:
   # First we get the neighbors of the node we're currently exploring
   neighbors = list(G.neighbors(curnode))
   if not curnode in sampled_nodes: # This is true if we never sampled this node before. This means we never added its connections to sampled_edges
      
      sampled_nodes.add(curnode) # This will allow us to remember we sampled this node
      
      # We update the set of sampled edges. We need to have a canonical representation of the edge because the network is undirected,
      # so if we already saw the edge because we sampled the neighbor, we might have stored the edge as (neighbor, curnode) rather than
      # (curnode, neighbor). With this min-max trick, this is not an issue.
      
      new_edges = set([(min(curnode, neighbor), max(curnode, neighbor)) for neighbor in neighbors])
      sampled_edges.update(new_edges) 
   
   # We move on to sampling a random neighbor of the current node, because we're doing a random walk.
   curnode = random.choice(neighbors) 

# Make a graf of the sampled edges
G_smpl = nx.Graph(list(sampled_edges))

# Print the number of nodes
print(len(G_smpl.nodes))
```
<span style="color:grey;">64294</span>

## 25.2

*Compare the CCDF of the degree distribution of your sample of the network from Exercise 25.1 with the one of the original network by fitting a log-log regression and comparing the exponents. You can take multiple samples from different seeds to ensure the robustness of your result.*

```python
from scipy.stats import linregress

# Function implementing the random walk logic from the previous question
# Tip: Should take in a graph and a number of nodes to sample
# Tip: Should return a graph with the sampled nodes and the edges between them
def rw(G, n):
   sampled_nodes = set()
   sampled_edges = set()
   curnode = random.choice(list(G.nodes))
   while len(sampled_nodes) <= n:
      neighbors = list(G.neighbors(curnode))
      if not curnode in sampled_nodes:
         sampled_nodes.add(curnode)
         sampled_edges |= set([(min(curnode, neighbor), max(curnode, neighbor)) for neighbor in neighbors])
      curnode = random.choice(neighbors)
   return nx.Graph(list(sampled_edges))

def ccdf(dd):
   dd = pd.DataFrame(list(dd.items()), columns = ("k", "count")).sort_values(by = "k")
   ccdf = dd.sort_values(by = "k", ascending = False)
   ccdf["cumsum"] = ccdf["count"].cumsum()
   ccdf["ccdf"] = ccdf["cumsum"] / ccdf["count"].sum()
   ccdf = ccdf[["k", "ccdf"]].sort_values(by = "k")
   return ccdf

def dd_exponent(degdistr):
   logcdf = np.log10(degdistr[["k", "ccdf"]])
   slope, log10intercept, r_value, p_value, std_err = linregress(logcdf["k"], logcdf["ccdf"])
   return slope

# Load the data
# The data is in the form of an edgelist (nodetype=int)
G = nx.read_edgelist("../data/25_1.txt", nodetype = int)

#Tip: Get the degree distribution of the original graph
#Tip 1: turn the degree view into a dictionary and then get the values
#Tip 2: You can use the Counter function from the collections package
#Tip 3 Use the ccdf on the degree distribution to get the ccdf
dd = Counter(dict(G.degree).values())
G_ccdf = ccdf(dd)

print("Original Exponent: %1.4f" % dd_exponent(G_ccdf))
```

<span style="color:grey;">Original Exponent: -1.6013</span>

```python
# Let's take 100 samples and store their degree exponent in a list
# This will take a while
smpl_exponents = []
for _ in range(100):
   G_smpl = rw(G, 2000)
   G_smpl_ccdf = ccdf(Counter(dict(G_smpl.degree).values()))
   smpl_exponents.append(dd_exponent(G_smpl_ccdf))

# Find the mean and standard deviation of the exponents
smpl_exponents_mean = np.mean(smpl_exponents)
smpl_exponents_std = np.std(smpl_exponents)
print("Sample Exponent: %1.4f (+/- %1.4f)" % (smpl_exponents_mean, smpl_exponents_std)) # The exponent of the sample is different! ~1.125 vs 
```
<span style="color:grey;">Sample Exponent: -1.1251 (+/- 0.0097)</span>