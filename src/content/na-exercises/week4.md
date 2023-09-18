---
title: 'Exercises - Week 4'
description: 'Exercises for week 4 - NA'
pubDate: 'Sep 18 2023'
heroImage: 'pink-network.avif'
---

```python
import pandas as pd
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import powerlaw as pl
import numpy as np
```

## 13.1

*Consider this network. Generate an Erdos-Renyi graph with the same number of nodes and edges. Plot both networks' degree CCDFs, in log-log scale. Discuss the salient differences between these distributions.*

**G(n,m)** (Erdos-Renyi): You decide first how many nodes the graph should have – which is the $n$ parameter – then you decide how many edges - which is the $m$ parameter. You can imagine this model as a bunch of buttons (nodes) and pieces of yarn (edges).

```python
# first, read the data
G1 = nx.read_edgelist('data_131.txt')

# make the random graph G(n,m) based in the loaded network
G_nm = nx.gnm_random_graph(len(G1.nodes), len(G1.edges))

# we make a function that returns the CCDF of a network, since we have to do it twice.
# you can simply copy the code from previous exercises. 
# see week 3 for explanation

def generate_ccdf(G):
   dd = Counter(dict(G.degree).values())
   dd = pd.DataFrame(list(dd.items()), columns = ("k", "count")).sort_values(by = "k")
   ccdf = dd.sort_values(by = "k", ascending = False)
   ccdf["cumsum"] = ccdf["count"].cumsum()
   ccdf["ccdf"] = ccdf["cumsum"] / ccdf["count"].sum()
   ccdf = ccdf[["k", "ccdf"]].sort_values(by = "k")
   return ccdf

# Now we use the function to create the CCDF for both networks
G_ccdf = generate_ccdf(G1)
G_rnd_ccdf = generate_ccdf(G_nm)
# renaming the random one, so we can distinguish them on the plot
G_rnd_ccdf["random ccdf"] = G_rnd_ccdf["ccdf"]

# Plot the distributions. The real world network (red) take a much larger maximum value and has
# fewer nodes with an average degree.
ax = plt.gca()
G_ccdf.plot(kind = "line", x = "k", y = "ccdf", color = "#e41a1c", loglog = True, ax = ax)
G_rnd_ccdf.plot(kind = "line", x = "k", y = "random ccdf", color = "#377eb8", loglog = True, ax = ax)
plt.show()
```

<img src="/output1.png" alt="dd" width="700" height="450">

*Here we can clearly see that even though the G(n,m) model is useful when the exact number of edges is of interest and for modelling certain real-world networks where the number of connections is fixed or known, it is not useful for modelling other properties of the network, such as degree distribution.*

## 13.4

*Generate an Erdos-Renyi graph with the same number of nodes and edges as the network used for Exercise 13.1. Calculate and compare the networks' clustering coefficients. Compare this with the connection probability p of the random graph (which you should derive from the number of edges and number of nodes using the formula I show in the chapter).*

**Connection Probability P:** $p = \frac{\bar k(\bar k-1)}{2}$, where $\bar k$ is number of nodes.

```python
# here we just reuse the network from previous exercise, G1
# we make a new random graph G(n,m)
G_nm2= nx.gnm_random_graph(len(G1.nodes), len(G1.edges))

# we compare the clustering coefficients with nx.transitivity
cc = nx.transitivity(G1)
cc_random = nx.transitivity(G_nm2)

print(f"CC for real network: {cc}, and CC for random network: {cc_random}")
```

<span style="color:grey;">CC for real network: 0.23143166252210473, and CC for random network: 0.005770218820398645</span>  
*As we can see, the clustering coefficient is much lower for the random network, which we would expect to see with a $G(n,m)$ network.*


```python
# at last, derive the parameter p with the above formula
def p(Graph):
    return len(Graph.edges) / ((len(Graph.nodes) * (len(Graph.nodes)-1)) / 2)

print(p(G1))
```

<span style="color:grey;">0.005653870016403056</span>  

## 14.1

*Generate a connected caveman graph with 10 cliques, each with 10 nodes. Generate a small world graph with 100 nodes, each connected to 8 of their neighbors. Add shortcuts for each edge with probability of 0.05. The two graphs have approximately the same number of edges. Compare their clustering coefficients and their average path lengths.*

**Recall** 
*The small world model* models high clustering, but its primary target is to explain small distances, which is present in real world networks. Imagine this picture from Michele’s book, where we have a lot of people standing in a circle, only being able to talk to you neighbours that can hear you, so maybe 2 people to your left, and 2 people to your right. Now we establish a rewiring probability $\bold{p}$ - the second parameter of the model (people can call eachother on the phone).

*The caveman model* is a simple network.

- Step a) decide the size of the cave
- Step b) decide the number of caves
- Step c) make each cave in a clique
- Step d) connect the nearest caves via random cave members

```python
# (Micheles comment): We use newman_watts_strogatz_graph instead of watts_strogatz_graph,
# because the latter rewires edges, while the question explicitly asked
# for adding shortcuts, rather than rewiring.

G_cm = nx.connected_caveman_graph(10, 10) # caveman
G_sw = nx.newman_watts_strogatz_graph(100, 8, p = .05) #small world


# Calculate properties.
G_sw_cc = nx.transitivity(G_sw) # clustering coeff.
G_cm_cc = nx.transitivity(G_cm)
G_sw_apl = nx.average_shortest_path_length(G_sw) # avg path length
G_cm_apl = nx.average_shortest_path_length(G_cm)

print(f'clustering coeff of caveman: {G_cm_cc}, and for small world: {G_sw_cc}')
print(f'avg path length of caveman: {G_cm_apl}, and for small world: {G_sw_apl}')
```

<span style="color:grey;">clustering coeff of caveman: 0.9307479224376731, and for small world: 0.583413693346191</span>  
<span style="color:grey;">avg path length of caveman: 5.9363636363636365, and for small world: 3.4183838383838383</span>  
*As we can see from the results, the Caveman Model gives us communities and a high clustering coeff. We can also see that the Small World Model gives a shorter avg. path length than the caveman, this is due to the rewiring.*

## 14.2

*Generate a preferential attachment network with 2,000 nodes and average degree of 2. Estimate its degree distribution exponent (you can use either the powerlaw package, or do a simple log-log regression of the CCDF).*

**Recall:** Preferential attachment is a model that can generate a scale-free network (a network that follows a power-law) This means that a few nodes become extremely well-connected (hubs), while most nodes have only a few connections. You can use this model to kind of explore the formation of hubs, by the way the nodes prefer to attach to more popular nodes, the “rich get richer” saying.

```python
# Generate the preferential attachment network
# it is called nx.barabasi_albert_graph, since those were the
# guys who invedted it :)
G2 = nx.barabasi_albert_graph(2000, 2)

# now we make the the CCDF, I will use the function i made in exercise 13.1
G2_ccdf = generate_ccdf(G2)

# we do exactly like in exercise 6.5 (see week 3 exercises) and fit the powerlaw
results = pl.Fit(G2_ccdf["ccdf"])
k_min = G2_ccdf[G2_ccdf["ccdf"] == results.power_law.xmin]["k"]
print("Powerlaw CCDF Fit: %1.4f x ^ -%1.4f (k_min = %d)" % (10 ** results.power_law.Kappa, results.power_law.alpha, k_min))

# we can also plot the powerlaw, in order to see if the powerlaw exists
G2_ccdf["fit"] = (10 ** results.power_law.Kappa) * (G2_ccdf["k"] ** -results.power_law.alpha)
ax = plt.gca()
G2_ccdf.plot(kind = "line", x = "k", y = "ccdf", color = "#e41a1c", loglog = True, ax = ax)
G2_ccdf.plot(kind = "line", x = "k", y = "fit", color = "#377eb8", loglog = True, ax = ax)
plt.savefig("ccdf_fit.png")
```

<span style="color:grey;">Powerlaw CCDF Fit: 11.0631 x ^ -1.7376 (k_min = 19)</span>

<img src="/ccdf_fit_.png" alt="dd" width="700" height="450">

## 15.1

*Generate a configuration model with the same degree distribution as this network. Perform the Kolmogorov-Smirnov test between the two degree distributions.*

**Recall:** The Configuration Model provides a way to generate a random graph that follows an exactly specified degree distribution. The way it works, is that:  
1. You specify a degree sequence (matching the degree sequence of the network you want to regenerate)
2. Then you force each node to pick a value from the sequence as it degree, until you have chosen all edges from the sequence.

**Kolmogorov-Smirnov test:** In the context of network theory, the Kolmogorov-Smirnov (KS) test can be applied to compare various network properties or distributions, providing insights into the differences between observed network characteristics and expected or theoretical distributions.

```python
from scipy.stats import ks_2samp # the kolmogorov-Smirnov test

# first we read the data
G3 = nx.read_edgelist('data_151.txt')

# Get the degree distribution
degdistr = sorted(dict(G3.degree).values(), reverse = True)

# Generate configuration model
G3_config = nx.configuration_model(degdistr)

# Get the degree distribution of the CM
degdistr_cm = sorted(dict(G3_config.degree).values(), reverse = True)

# Perform the KS test. 
ks_2samp(degdistr, degdistr_cm)
```
<span style="color:grey;">KstestResult(statistic=0.0, pvalue=1.0, statistic_location=1, statistic_sign=1)</span>  
*The two distributions are exactly the same!*

## 15.2

*Remove the self-loops and parallel edges from the synthetic network you generated in Exercise 15.1 as the configuration model of this network. Note the % of edges you lost. Re-perform the Kolmogorov-Smirnov test with the original network's degree distribution.*

```python
# Remove parallel edges from network from previous exercise
G4_config = nx.Graph(G3_config)

# Remove self loops
G4_config.remove_edges_from(nx.selfloop_edges(G4_config))

# get the degree distr. of the new network
dd_G4 = sorted(dict(G4_config.degree).values(), reverse = True)

# Perform the KS test. 
ks_2samp(degdistr, dd_G4)
```
<span style="color:grey;">KstestResult(statistic=0.006996770721205597, pvalue=0.9999999999817853, statistic_location=71, statistic_sign=-1)</span>  
*They are almost the same!*

## 15.3

*Generate an LFR benchmark with 100,000 nodes, a degree exponent alpha = 3.13, a community exponent of 1.1, a mixing parameter mu = 0.1, average degree of 10, and maximum degree of 10000. (Note: there's a networkx function to do this). Can you recover the alpha value by fitting the degree distribution?*

**Recall:** If you want an even more real-world like way of generating a network, you can use the LFR, since it allows for much more complex network properties. The LFR is good at generating community structure, making it suitable for benchmarking community detection algorithms and studying community-like aspects of networks. (see more about LFR on lecture 4 notes)

```python
# Plugging in all the parameters
G5 = nx.LFR_benchmark_graph(100000, 3.13, 1.1, 0.1, average_degree = 10, max_degree = 10000)

# now we make the the CCDF, I will use the function i made in exercise 13.1
G5_ccdf = generate_ccdf(G5)

# Again, we fit the powerlaw (you can also just do a linear regression)
results = pl.Fit(G5_ccdf["ccdf"])
k_min = G5_ccdf[G5_ccdf["ccdf"] == results.power_law.xmin]["k"]
print("Powerlaw CCDF Fit: %1.4f x ^ -%1.4f (k_min = %d)" % (10 ** results.power_law.Kappa, results.power_law.alpha, k_min))

# we can also plot the powerlaw, in order to see if the powerlaw exists
G5_ccdf["fit"] = (10 ** results.power_law.Kappa) * (G5_ccdf["k"] ** -results.power_law.alpha)
ax = plt.gca()
G5_ccdf.plot(kind = "line", x = "k", y = "ccdf", color = "#e41a1c", loglog = True, ax = ax)
G5_ccdf.plot(kind = "line", x = "k", y = "fit", color = "#377eb8", loglog = True, ax = ax)
plt.savefig("ccdf_fit.png")
```

<span style="color:grey;">Powerlaw CCDF Fit: 10.3873 x ^ -1.5235 (k_min = 88)</span>

<img src="/output2.png" alt="dd" width="700" height="450">

## 16.1

Perform 1,000 edge swaps, creating a null version of this network. Make sure you don't create parallel edges. Calculate the Kolmogorov-Smirnov distance between the two degree distributions. Can you tell the difference?

**Recall:** Network Shuffling:  
    - Here you create a null version of your network by something called edge swapping. It works by picking two pairs of connected nodes and rewire the edges, swapping them.  
    - Once you have thousands of these null networks, you can test some property and get an idea of how statistically significant the observation is.

```python
import random

# load the data
G6 = nx.read_edgelist('data_161.txt')
edgeset = set(G6.edges)
# now we create a for loop that does edgeswapping (Network shuffling)
for _ in range(1000):
   es = None
   # We reject a swap if the edges we were going to add are already in the edge set
   while es is None or (es[0][0], es[1][0]) in edgeset or (es[0][1], es[1][1]) in edgeset:
      es = list(random.sample(sorted(edgeset), 2))
   # Remove the old edges from the edge set
   edgeset -= set(es)
   # Add the swapped edges to the edge set
   edgeset.add((es[0][0], es[1][0]))
   edgeset.add((es[0][1], es[1][1]))

G_shuffle = nx.Graph(list(edgeset))

# Get the degree distributions & do the KS test 
dd_G6 = sorted(dict(G6.degree).values(), reverse = True)
dd_rnd = sorted(dict(G_shuffle.degree).values(), reverse = True)
a1 = ks_2samp(dd_G6, dd_rnd)
print(a1)
```

<span style="color:grey;">KstestResult(statistic=0.0005347593582887701, pvalue=1.0, statistic_location=3, statistic_sign=-1)</span>

## 16.2

*Do you get larger KS distances on the network from Exercise 16.1 if you perform 2,000 swaps instead of 1,000? Do you get smaller KS distances if you perform 500?*

```python
# repeat the exact same steps as before, just change the 
# range. results are:
```
<span style="color:grey;">KstestResult(statistic=0.0005347593582887701, pvalue=1.0, statistic_location=3, statistic_sign=-1)</span>  
<span style="color:grey;">KstestResult(statistic=0.0010695187165775401, pvalue=0.9999999999999991, statistic_location=3, statistic_sign=-1)</span>  
<span style="color:grey;">KstestResult(statistic=0.07918650785691075, pvalue=0.00010704742275723213, statistic_location=2, statistic_sign=-1)</span>  
*There really isn't much difference between the different shuffled versions and the original network.*