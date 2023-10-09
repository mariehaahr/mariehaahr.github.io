---
title: 'Lecture 7'
description: 'Community discovery via random walks, label propagation, or comparison with a null model, Modularity, temporal community discovery, normalised mutual information'
pubDate: 'Sep 25 2023'
heroImage: '/lec7_na.png'
---

**Readings**: "The Atlas for the Aspiring Network Scientist", Chapters 31 & 32 (minus sections 31.5, 31.6, 32.3)

#### Table of contents
- [**Graph Partitions**](#graph-partitions)
  - [Stochastic Blockmodels](#stochastic-blockmodels)
    - [Classical Community Definition](#classical-community-definition)
    - [Maximum Likelihood](#maximum-likelihood)
  - [Random Walks](#random-walks)
    - [Infomap](#infomap)
  - [Label Percolation (Label Propagation)](#label-percolation-label-propagation)
  - [Temporal Communities](#temporal-communities)
    - [Evolutionary Clustering](#evolutionary-clustering)
- [**Community Evaluation**](#community-evaluation)
  - [Modularity](#modularity)
    - [As a quality measure](#as-a-quality-measure)
    - [As a maximisation target](#as-a-maximisation-target)
    - [Expanding modularity](#expanding-modularity)
    - [Issues with modularity](#issues-with-modularity)
  - [Other Topological Measures](#other-topological-measures)
    - [Conductance](#conductance)
    - [Internal density](#internal-density)
    - [Cut](#cut)
    - [Out-degree fraction](#out-degree-fraction)
  - [Normalised Mutual Information](#normalised-mutual-information)

# **Graph Partitions**

**What are communities?** Communities are groups of nodes densely connected to each other and sparsely connected to nodes outside the community. If two nodes are in the same community they are more likely to connect than if they are in two distinct communities. *Community discovery* is then the process of identifying these groups of nodes (communities) that have strong internal connections and weaker connections with nodes in other communities. It is a very fundamental task in network analysis, and can help us find the underlying structure and organisation of a network.

**Why find communities?** Because it is the equivalent of performing data clustering in data mining, machine learning etc. You can also use it to condense a complex network into a simpler view, if you need an “overall look” of the structures.

## Stochastic Blockmodels

### Classical Community Definition

> *Communities are groups of nodes densely connected to each other and sparsely connected to nodes outside the community.*
> 

We assume from this definition that a node only can be a part if a single community. When we say we “partition” we mean assigning a node to a community. In early community discovery days, the **stochastic block model** (SBM) was the main approach (we know them from week 4, as a random graph generator) so the question is, how do we apply that method in finding communities?

### Maximum Likelihood

<div style="text-align: center;">
    <img src="/Screenshot_2023-10-09_at_12.31.53.png" alt="dd" width="500" height="350">
</div>

Looking at the figure above, at (a) we see a network’s adjacency matrix, and (b) shows a possible SBM representation. These blocks represent the communities, and therefore the model have the name “Block” model.

We want to find the SBM model that most accurately reconstructs the adjacency matrix. We can do that by **maximum likelihood estimation** (MLE). With MLE we define the block structure and connections between communities in the network. It's a probabilistic framework that aligns the generated network based on the SBM with the actual observed network, helping in uncovering the underlying community structure.

## Random Walks

Looking at the network “process” is a common way of detecting communities. One way to do that, is with a random walk. Traversing randomly through a network with communities, the probability of going from one community to another should be pretty low, since there are only a few connections between distinct communities. And vice versa, if nodes appear together in a random walk, then we can also assume that they are in the same community.

### Infomap

The most known and best performing random walk approach is Infomap (map equation approach). The infomap is what you use to encode the random walk information with the minimum possible number of bits - meaning you are minimising the “code length”.

Infomap has been adapted to numerous scenarios. Many involve hierarchical, multilayer and overlapping community detection.

**Be aware**: Since these methods are random, running the algorithm several times will likely give different results.

## Label Percolation (Label Propagation)

Another dynamic approach is having nodes deciding for themselves to which community they belong by looking at their neighbour’s community assignments. 

- At first, the labels will be random, and nodes will switch their labels randomly.
- Then, at some point, by chance some nodes will adopt the same label, and if these nodes are in the same community, it starts being the majority label and eventually will be adopted by everybody in that community.

So the labels will percolate (propagate) through the network until we reach a state in which no more significant changes can happen.

This approach is computationally cheap, and it runs linearly in terms of number of edges $O(|E|)$.

**Be aware**: Again, since this method is random, running the algorithm several times will likely give different results.

## Temporal Communities

### Evolutionary Clustering

What do we do with networks that change over time? **Dynamic community discovery!**

Over time there can happen different things with your communities. They can either **grow** or **shrink**, they can also **merge** (e.g. going from 2 to 1 community) or **split** (e.g. going from 1 to 2 communities). New communities can be **born** or communities can **die**.

**How do your detect communities in an evolving graph?** You perform evolutionary clustering. You simply add a second term to whatever criterion we use to find communities in a snapshot of time - this is called **smoothing**. 

**Another approach** is instead of redoing the clustering and then do the smoothing for every snapshot, you simply only do this for the first snapshot, and when your receive a new snapshot, you can adapt the previous communities to the new network, and use some specific rules or methods to update the communities. 

# **Community Evaluation**

How do you know if the communities your found were good partitions? Or how do you figure out which method is best?

**Remember:** This is more data exploration than it is to find an ultimate truth. Make sure your can reason for why this exact method makes sense for your network.

## Modularity

Here we focus only on the topological information of our network.

### As a quality measure

> *Modularity is a measure following closely the classical definition of community discovery. It is all about the internal density of your communities.*
> 

Modularity compares the observed network with a random expectation, in order to mitigate having one community per edge with a density of one.

So we compare the observed number of edges inside a community with the expected number of edges.

<div style="text-align: center;">
    <img src="/Screenshot_2023-10-09_at_13.32.19.png" alt="dd" width="500" height="350">
</div>

The modularity ranges from -0.5 (disassortative structure) to 1 (fully made by cliques). A positive modularity happens when our partitions finds nodes whose number of edges exceeds null expectation. Whereas negative modularity is achieved by trying to group nodes together that connect to each other less than chance.

Each node $v$ has a number of connection opportunities equal to its degree. The number of possible wirings we can make in the network is twice the number of edges. In a configuration model (see week 4 notes), the probability of connecting $u$ and $v$ is $(k_u k_v) / 2|E|$.

So, modularity:

$$
M = \underbrace{\frac{1}{2|E|}}_{\text{Normalise}}\sum_{v,u \in V} \left[\underbrace{A_{vu}}_{\text{their observed relation}} - \frac{k_vk_u}{2 |E|} \right] \cdot \underbrace{\delta (c_v, c_u)}_{\text{Kronecker delta}}
$$

For every pair of nodes in the same community subtract from their observed relation the expected number of relations given the degree of the two nodes and the total number of edges in the network, then normalise.

**Kronecker delta:** $\delta(c_v,c_u)$ is 1 if they are in the same community and 0 otherwise

So now we know, the higher modularity → Better partition. When you do a partition, the one that gets the higher modularity score is the better partition.

### As a maximisation target

Instead of simply evaluating a partition with modularity, you can use it as an optimisation where you modify your partition in a smart way to get the highest possible modularity.

You could, for instance, progressively condense your network such that you preserve its modularity, or using modularity to optimise the encoding of information flow in the network.

There are many more ways to do it.

### Expanding modularity

The standard definition of modularity wirjs exclusively with ******************undirected******************, ********************unweighted********************, ****************disjoint partitions****************. 

Modularity for directed networks:

$$
M_{\text{directed}} =\underbrace{\frac{1}{2|E|}}_{\text{Normalise}}\sum_{v,u \in V} \left[A_{vu} - \frac{k_v^{out}k_u^{in}}{ |E|} \right] \cdot \underbrace{\delta (c_v, c_u)}_{\text{Kronecker delta}}
$$

In the undirected case we used the degree for both nodes $u$ and $v$, but here we instead use their in-and out- degree alternatively.

If the directed network is also weighted, we expand the modularity to:

$$
M_{\text{weighted}} =\underbrace{\frac{1}{2|E|}}_{\text{Normalise}}\sum_{v,u \in V} \left[w_{vu} - \frac{w_v^{out}w_u^{in}}{ \sum_{u,v \in V} w_{uv}} \right] \cdot \underbrace{\delta (c_v, c_u)}_{\text{Kronecker delta}}
$$

### Issues with modularity

Modularity has a tendency to prefer communities with a higher number of edges.

- The resolution limit. It wants communities to be a certain size.
- It prefers to merge to neighbouring cliques, rather than splitting each clique, like it would do.
    

<div style="text-align: center;">
    <img src="/Screenshot_2022-11-03_at_11.03.24.png" alt="dd" width="500" height="350">
</div>    

Modularity cannot “see” long range communities. If your communities are very large and span
across multiple degrees of separation, modularity will “over-partition” them.

## Other Topological Measures

### Conductance

The idea behind conductance is that communities should not be conductive: whatever falls into, or originates from, them should have a hard time getting out. So we compare the volume of edges pointing outside the cluster.

### Internal density

The other side of the conductance coin is the internal density measure. How many edges are inside the community over the total possible number of edges the community could host.

### Cut

> ******************The cut ratio is the fraction of all possible edges leaving the community.******************
> 

We minimise this, in order to solve the min-cut problem:

$$
f(C) = \frac{|E_{C,B}|}{|C|(|V|-|C|)}
$$

But it is often modified to be a normalised min-cut:

$$
f(C) = \frac{|E_{C,B}|}{2|E_C| + |E_{B,C}|} + \frac{|E_{B,C}|}{2(|E|- |E_C|)+|E_{B,C}|}
$$

### Out-degree fraction

The out degree fraction (ODF), as the name suggests, looks at the share of edge pointing outside the cluster. 

## Normalised Mutual Information

In the case where we have a network with node attributes, we know by following the homophily assumption (see lecture 6) nodes with the same attributes will tend to connect with each other, and therefore be in the same community. 

Normalised mutual information (NMI) is another way to evaluate your partition when you have metadata about your nodes – if you assume that communities should be used to recover latent node attributes. Be aware, though, that not always nodes with similar attributes connect to each other.

When evaluating community partitions, you need a standardised yardstick to know whether a network has communities more tightly knit than another - or if it has communities at all!

1. **Get node sttributes**:
    - Nodes are associated with attributes based on some features or characteristics relevant to the problem (e.g., features describing a person in a social network).
2. **Apply a community detection algorithm**:
    - Use a community detection algorithm (e.g., clustering) to partition the nodes based on both network structure and node attributes.
3. **Calculate NMI**:
    - Calculate the Normalised Mutual Information (NMI) between the discovered communities (obtained from the algorithm) and the ground truth communities (if available), considering both network structure and node attributes.
    
    $$
    NMI(A,B) = \frac{2 \times I(AB)}{H(A)+ H(B)}
    $$
    
    - $I(A,B)$ is the mutual information between the discovered communities (A) and the ground truth communities (B) considering both network structure and node attributes.
4. **Interpretation**:
    - Higher NMI values indicate better agreement or similarity between the discovered communities and the ground truth communities, considering both network structure and node attributes.