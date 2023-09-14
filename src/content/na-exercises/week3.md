---
title: 'Exercises - Week 3'
description: 'Lorem ipsum dolor sit amet'
pubDate: 'Sep 10 2023'
heroImage: '/na_ex/third.png'
---

### 6.4
*Plot the degree distribution of this network. Start from a plain degree distribution, then in log-log scale, finally plot the complement of the cumulative distribution.*

```python
import pandas as pd
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# first load the data
G1 = nx.read_edgelist("data_64.txt")

# we use the function Counter() such that we can plot the degree distrobution. We are
# only interested in the number of nodes that have that specific number of neighbours
# so Counter() is a perfect method for this exercise.

# make a dictionary of the degrees of the network
d = dict(G1.degree)
# turn into a count of degree (the values of the dict)
counter = Counter(d.values())
# Make it into a dataframe
counter = pd.DataFrame(list(counter.items()), columns = ("k", "count")).sort_values(by = "k")

# We plot the degree distr. and save it to a file
plt.scatter(counter['k'], counter['count'], c = 'blue', s=10, alpha=0.5)
plt.title("plain degree distr.")
plt.savefig("degree_distribution.png")
plt.show()
```

<img src="/degree_distribution.png" alt="dd" width="700" height="450">

```python
# lets plot the same thing in log-log scale
#d_log = pd.DataFrame((counter['k']), np.log(counter['count']))
plt.scatter(np.log2(counter['k']), np.log2(counter['count']), c = 'blue', s=10, alpha=0.5)
plt.title("log-log degree distr.")
plt.savefig("loglog_degree_distribution.png")
plt.show()
```

<img src="/loglog_degree_distribution.png" alt="dd" width="700" height="450">

```python
# finally plotting the CCDF - the complementary cumulative distribution function

# To make the CCDF we need to know how many nodes have degree equal to or higher
# than a specific value. So we sort the dataframe in descending degree order, so
# that the pandas cumsum function will calculate that for us. Then we normalize by
# the total degree sum, so that the count becomes a probability. We then sort in
# ascending degree value, to respect the convention.
ccdf = counter.sort_values(by = "k", ascending = False)
ccdf["cumsum"] = ccdf["count"].cumsum()
ccdf["ccdf"] = ccdf["cumsum"] / ccdf["count"].sum()
ccdf = ccdf[["k", "ccdf"]].sort_values(by = "k")

# Plot as usual and save it for later, since it's very pretty.
ccdf.plot(kind = "line", x = "k", y = "ccdf", color = "#e41a1c", loglog = True)
plt.savefig("degree_distribution_ccdf.png")
```

<img src="/degree_distribution_ccdf.png" alt="dd" width="700" height="450">

### 6.5

*Estimate the power law exponent of the CCDF degree distribution from Exercise 6.4. First by a linear regression on the log-log plane, then by using the powerlaw package. Do they agree? Is this a shifted power law? If so, what's k min? (Hint: powerlaw can calculate this for you)*

```python
# estimating the CCDF by a linear regression
from scipy.stats import linregress
# do a linear regr.
logcdf = np.log10(ccdf[["k", "ccdf"]])
slope, log10intercept, r_value, p_value, std_err = linregress(logcdf["k"], logcdf["ccdf"])
print("CCDF Fit: %1.4f x ^ %1.4f (R2 = %1.4f, p = %1.4f)" % (10 ** log10intercept, slope, r_value ** 2, p_value))
```

Looking at the R-squared value, we can see that the linear regression fits very well on the cumulative degree distribution.

```python
# estimating the CCDF by a powerlaw package
import powerlaw as pl

# With the powerlaw package, fitting the CCDf is simple. 
# It will store results in the .power_law property. To
# get the actual k_min, we need to find the degree value 
# corresponding to the probability in .power_law.xmin:
# pandas makes it easy. This is definitely a shifted power 
# law. (Kappa contains the intercept information)

results = pl.Fit(ccdf["ccdf"])
k_min = ccdf[ccdf["ccdf"] == results.power_law.xmin]["k"]
print("Powerlaw CCDF Fit: %1.4f x ^ -%1.4f (k_min = %d)" % (10 ** results.power_law.Kappa, results.power_law.alpha, k_min))

# Let's plot the best fit.
ccdf["fit"] = (10 ** results.power_law.Kappa) * (ccdf["k"] ** -results.power_law.alpha)
ax = plt.gca()
ccdf.plot(kind = "line", x = "k", y = "ccdf", color = "#e41a1c", loglog = True, ax = ax)
ccdf.plot(kind = "line", x = "k", y = "fit", color = "#377eb8", loglog = True, ax = ax)
plt.savefig("ccdf_fit.png")
```


<img src="/ccdf_fit.png" alt="dd" width="700" height="450">

*We can clearly see that the powerlaw is very shifted, since the its head (the beginning of the plot) is not following the fit.*

### 6.6

*Find a way to fit the truncated power law of this network. Hint: use the scipy.optimize.curve_fit to fit an arbitrary function and use the functional form I provide in Chapter 6 of the book.*

Reading a graph in Pajek format from a path refers to the process of loading a network graph that has been saved or stored in the Pajek file format from a specified file path. Pajek is a popular software tool for network analysis and visualization, and it uses its own file format to represent network data.

```python
from scipy.optimize import curve_fit

# load the data
G2 = nx.read_pajek("data_66.net")
# Redo the CCDF just like the 2 previous exercises
dd = Counter(dict(G2.degree).values())
dd = pd.DataFrame(list(dd.items()), columns = ("k", "count")).sort_values(by = "k")
ccdf = dd.sort_values(by = "k", ascending = False)
ccdf["cumsum"] = ccdf["count"].cumsum()
ccdf["ccdf"] = ccdf["cumsum"] / ccdf["count"].sum()
ccdf = ccdf[["k", "ccdf"]].sort_values(by = "k")

# Let's define a custom function which is a power law with its exponential truncation. We
# also define its logarithm, because we fit it to the log of the CCDF in curve_fit. This
# is done because we want to minimize the relative error, not the absolute error (since
# the tail of the distribution is very important, but it contributes very little to the
# absolute error). Then we plot.
def f(x, a, l):
   return (x ** a) * np.exp(-l * x) 

def log_f(x, a, l):
   return np.log10(f(x, a, l))

# we use the defined functions above to fit the curve to the CCDF
popt, pcov = curve_fit(log_f, ccdf["k"], np.log10(ccdf["ccdf"]), p0 = (1, 1))
ccdf["fit"] = ccdf.apply(lambda x: f(x["k"], popt[0], popt[1]), axis = 1)

# plot it
ax = plt.gca()
ccdf.plot(kind = "line", x = "k", y = "ccdf", color = "#e41a1c", loglog = True, ax = ax)
ccdf.plot(kind = "line", x = "k", y = "fit", color = "#377eb8", loglog = True, ax = ax)
plt.savefig("ccdf_fit.png")
```

<img src="/curve_ccdf2.png" alt="Alt Text" width="400" height="600">


### 9.3

*Calculate the global, average and local clustering coefficient for this network.*

The clustering coefficient is a measure to distinguish between different cases of a network with the same number of nodes and the same density (they can look very different). The clustering coefficient is a number return that describes quantitatively how “clustered” a network looks. 

$$CC=\frac{3 \cdot \text{\#Triangles}}{\text{\#Triads}}$$

**Global Clustering Coefficient**: 
The global clustering coefficient, quantifies the overall tendency of nodes in the network to form clusters or triangles. A high global clustering coefficient indicates that the network is highly clustered, and nodes tend to form tightly interconnected groups. A low coefficient suggests a more random or loosely connected network.

**Average Clustering Coefficient**
The average clustering coefficient, provides the average clustering tendency of nodes in the network. It's the average of the local clustering coefficients of all nodes in the network. The average clustering coefficient provides an overview of how clustered the network is on average. A higher value indicates more local clustering within the network. *Here you calculate all of the local clustering coefficients, and then take the average. This is what makes it so different from the Global Clustering Coefficient.*

**Local Clustering Coefficient**
The local clustering coefficient of a specific node measures how well its neighbors are connected to each other. It quantifies the likelihood that the neighbors of a node form a cluster around that node. The local clustering coefficient of a node provides insight into how tightly its immediate neighborhood is connected. A high local clustering coefficient indicates that the node's neighbors are well-connected, while a low coefficient suggests that the neighbors are not well-connected to each other.

```python
# load data
G3 = nx.read_edgelist('data_93.txt')

# Global Clustering Coefficient
global_clustering_coefficient = nx.transitivity(G3)
print("Global Clustering Coefficient:", global_clustering_coefficient)

# Average Clustering Coefficient
average_clustering_coefficient = nx.average_clustering(G3)
print("Average Clustering Coefficient:", average_clustering_coefficient)

# Local Clustering Coefficient (for a specific node 'node_id')
node_id = 0  # Replace with the node you want to calculate for
local_clustering_coefficient = nx.clustering(G3)
print(f'dictinary of local clustering coefficients: {local_clustering_coefficient}')
```

<span style="color:grey;">Global Clustering Coefficient: 0.12443636088060324</span>  
<span style="color:grey;">Average Clustering Coefficient: 0.6464630921565044</span>  
<span style="color:grey;">dictinary of local clustering coefficients: {'100': 1.0, '101': 1.0, '10': 0.6, ...</span>  

### 9.4

*What is the size in number of nodes of the largest maximal clique of the network used in Exercise 9.3? Which nodes are part of it?*

```python
# first load the data
G4 = nx.read_edgelist('data_94.txt')

# networkx has a function find_cliques, which returns all maximal cliques in an undirected graph
cliques = nx.find_cliques(G4)
cliques = list(cliques)

# the function nx.graph_clique_number() returns the size of the maximal clique.
largest_clique = nx.graph_clique_number(G4, cliques = cliques)
print(f'The largest clique is of size {largest_clique}')

# if we want to find out which nodes are in the cliques, we iterate iver the list of all cliques
# and find the one who have a size of 9 nodes

for i in cliques:
    if len(i) == largest_clique:
        print(i)
```

<span style="color:grey;">The largest clique is of size 9</span>  
<span style="color:grey;">['15', '5', '2', '38', '4', '86', '12', '13', '82']</span>  
<span style="color:grey;">['15', '5', '2', '97', '51', '57', '58', '55', '56']</span>  
<span style="color:grey;">['15', '5', '2', '97', '51', '57', '58', '4', '131']</span>  
<span style="color:grey;">['15', '5', '2', '97', '51', '57', '58', '183', '56']</span>  
<span style="color:grey;">['15', '5', '2', '97', '51', '57', '58', '183', '131']</span>  
<span style="color:grey;">['15', '5', '2', '13', '12', '87', '86', '4', '82']</span>  

### 10.4

*What's the diameter of the graph below? What's its average path length?*

<img src="/graph_104.png" alt="Alt Text" width="300" height="400">

**Diameter**: The rightmost column of the histogram of shortest paths, we have the number of shortest paths of maximum length. This is the diameter of the network. The worst case for reachability in the network.

In this case the diameter is 4.

**Average path length**: What we calculate, then, is not the longest shortest path, but the typical path length, which is the average of all shortest path lengths.  
1 1 2 2 3 3 4 4 = 20  
1 1 1 1 2 2 3 3 = 14  
1 1 1 1 2 2 3 3 = 14  
2 1 1 1 2 2 3 3 = 15\
2 1 1 1 1 1 2 2 = 11  
3 2 2 2 1 1 1 1 = 13  
3 2 2 2 1 1 1 1 = 13  
4 3 3 3 2 1 1 1 = 18  
4 3 3 3 2 1 1 1 = 18  
(20 + 14+ 14 + 15 + 11 + 13 + 13 + 18 + 18) / (8 * 9) = 1.8888888888888888

### 11.4

*What's the most central node in the network 'used for Exercise 11.3 according to PageRank? How does PageRank compares with the in-degree? (for instance, you could calculate the Spearman and/or Pearson correlation between the two)*

```python
# load data as a directed network
G5 = nx.read_edgelist('data_114.txt', create_using=nx.DiGraph())

# we can use networkx's nx.pagerank() to find the node with the maximum value
pagerank = nx.pagerank(G5)
max_node = max(pagerank, key = pagerank.get)
# key=pagerank.get is used as a custom sorting key for the max function. 
# It tells Python to use the values from the pagerank dictionary 
# (the PageRank scores) to determine the maximum value.

print(f'The most central node is {max_node}, with a pagerank of {round(pagerank[max_node], 5)}')
```

<span style="color:grey;">The most central node is 836, with a pagerank of 0.00322</span>  

```python
# How does PageRank compares with the in-degree? (for instance, you could calculate the 
# Spearman and/or Pearson correlation between the two)
from scipy.stats import pearsonr, spearmanr
indegree = dict(G5.in_degree)
# we'll make a numpy array of the pagerank scores
p_arr = np.array([pagerank[v] for v in G5.nodes])
# we'll do the same for the in degree dict
d_arr = np.array([indegree[w] for w in G5.nodes])

# now lets calculate the spearman and pearson correlation
# both function will return 2 values, the correlation coefficient and p-value for testing non-correlation
pearson_c, rval_p = pearsonr(p_arr, d_arr)
spearman_c, rval_s = spearmanr(p_arr, d_arr)

print(f'the pearson corr. {pearson_c} with a p-val of {rval_p}')
print(f'the pearson corr. {spearman_c} with a p-val of {rval_s}')
```

<span style="color:grey;">the pearson corr. 0.869335082603482 with a p-val of 0.0</span>  
<span style="color:grey;">the pearson corr. 0.8965286651584561 with a p-val of 0.0</span>  

*So we can conclude that the in-degree of a node almost says the same as the pagerank score of a node, since they have a very high correlation.*

### 11.5  
*Which is the most authoritative node in the network used for Exercise 11.3? Which one is the best hub? Use the HITS algorithm to motivate your answer (if using networkx, use the scipy version of the algorithm).*

**HITS** is an algorithm designed to estimate a node’s centrality in a directed network. Differently from other centrality measures, HITS assigns two values to each node, you can say it assigns to one of two roles.
 
<img src="/hub.png" alt="hub" width="350" height="200">

Hubs and authorities are an instance in which the quantitative
approach of the centrality measures and the qualitative approach
of the node roles meet. There is a way to estimate the degree of
“hubbiness” and “authoritativeness” in a network. *This is what the
HITS algorithm does. The underlying principle is very simple. A
good hub is a hub that points to good authorities.* A good authority
is an authority which is pointed by good hubs. These recursive
definitions can be solved iteratively – or, more efficiently, with clever
linear algebra – and they eventually converge.

```python
# read the data and load it a directed graph
G6 = nx.read_edgelist("data_115.txt", create_using = nx.DiGraph())

# now we use the HITS algorithm
hits = nx.hits_scipy(G6)

# the nx.hits_scipy() returns a dictionary with the hubs and authority
best_hub = max(hits[0], key = hits[0].get)
best_auth = max(hits[1], key = hits[1].get)

print("Best Hub: %s" % best_hub)
print("Best Authority: %s" % best_auth)
```

<span style="color:grey;">Best Hub: 2375</span>  
<span style="color:grey;">Best Authority: 2056</span>  

### 11.7
*Calculate the k-core decomposition of this network. What's the highest core number in the network? How many nodes are part of the maximum core?*

When it comes to node centrality, one common term you’ll hear thrown around is one of “core” node. This is usually a qualitative distinction, but sometimes we need a quantitative one. With k-core centrality we look for a way to say that a node is “more core” than another.  One can easily identify the k-core of a network via the k-core decomposition algorithm. 

*K-Core Definition*: In a graph, a k-core is a maximal subgraph where all nodes have a degree of at least k within that subgraph. In other words, it's a subset of nodes that are tightly connected to each other.

*K-Core Decomposition*: To find the k-cores within a graph, you perform a k-core decomposition. This involves iteratively removing nodes with degrees less than k until no more such nodes can be removed. The remaining nodes and edges constitute the k-core.

```python
#load the data
G7 = nx.read_edgelist('data_117.txt')

# Calculating the k-core decomposition and storing the maximum value.
kcore = nx.core_number(G7)
highest_core = max(kcore.values())
print("# of nodes in the maximum core: %d" % (len([v for v in kcore if kcore[v] == highest_core])))
```

<span style="color:grey;">\# of nodes in the maximum core: 41</span>  