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

<img src="../../../public/na_ex/degree_distribution.png" alt="dd" width="700" height="450">


```python
# lets plot the same thing in log-log scale
#d_log = pd.DataFrame((counter['k']), np.log(counter['count']))
plt.scatter(np.log2(counter['k']), np.log2(counter['count']), c = 'blue', s=10, alpha=0.5)
plt.title("log-log degree distr.")
plt.savefig("loglog_degree_distribution.png")
plt.show()
```

<img src="../../../public/na_ex/loglog_degree_distribution.png" alt="dd" width="700" height="450">

```python
# finally plotting the CCDF

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

<img src="../../../public/na_ex/degree_distribution_ccdf.png" alt="dd" width="700" height="450">

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

# With the powerlaw package, fitting the CCDf is simple. It will store results in the .power_law property. To
# get the actual k_min, we need to find the degree value corresponding to the probability in .power_law.xmin:
# pandas makes it easy. This is definitely a shifted power law. (Kappa contains the intercept information)
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


<img src="../../../public/na_ex/ccdf_fit.png" alt="dd" width="700" height="450">

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
```

### 9.3

*Calculate the global, average and local clustering coefficient for this network.*

The clustering coefficient is a measure to distinguish between different cases of a network with the same number of nodes and the same density (they can look very different). The clustering coefficient is a number return that describes quantitatively how “clustered” a network looks. 

$$CC=\frac{3 \cdot \text{\#Triangles}}{\text{\#Triads}}$$

**Global Clustering Coefficient**: 
The global clustering coefficient, quantifies the overall tendency of nodes in the network to form clusters or triangles. A high global clustering coefficient indicates that the network is highly clustered, and nodes tend to form tightly interconnected groups. A low coefficient suggests a more random or loosely connected network.

**Average Clustering Coefficient**
The average clustering coefficient, provides the average clustering tendency of nodes in the network. It's the average of the local clustering coefficients of all nodes in the network. The average clustering coefficient provides an overview of how clustered the network is on average. A higher value indicates more local clustering within the network.

**Local Clustering Coefficient**
The local clustering coefficient of a specific node measures how well its neighbors are connected to each other. It quantifies the likelihood that the neighbors of a node form a cluster around that node. The local clustering coefficient of a node provides insight into how tightly its immediate neighborhood is connected. A high local clustering coefficient indicates that the node's neighbors are well-connected, while a low coefficient suggests that the neighbors are not well-connected to each other.