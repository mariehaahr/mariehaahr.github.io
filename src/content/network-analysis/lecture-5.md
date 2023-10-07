---
title: 'Lecture 5'
description: 'Bipartite Network Projection, Backboning, Sampling'
pubDate: 'Sep 25 2023'
heroImage: '/lec5_na.png'
---


Bipartite network projection, Weighted network back-boning, Sampling of large scale networks

#### Table of contents
- [Hairballs](#hairballs)
- [Bipartite Projections](#bipartite-projections)
  - [Simple Weights](#simple-weights)
  - [Vectorised Projection](#vectorised-projection)
    - [The saturation problem](#the-saturation-problem)
  - [Hyperbolic Weights](#hyperbolic-weights)
  - [Resource Allocation](#resource-allocation)
  - [Random Walks](#random-walks)
- [Network Backboning](#network-backboning)
  - [Structural back-boning approaches](#structural-back-boning-approaches)
    - [Naive Thresholding](#naive-thresholding)
    - [Doubly Stochastic](#doubly-stochastic)
    - [High-Salience Skeleton](#high-salience-skeleton)
    - [Convex Network Reduction](#convex-network-reduction)
  - [Statistical backboning approaches](#statistical-backboning-approaches)
    - [Disparity Filter](#disparity-filter)
    - [Noise-Corrected](#noise-corrected)
- [Network Sampling](#network-sampling)
  - [Induced Strategies](#induced-strategies)
    - [Node induced](#node-induced)
    - [Edge induced](#edge-induced)
  - [Topological Strategies](#topological-strategies)
  - [Topological BFS Variants](#topological-bfs-variants)
    - [Snowball](#snowball)
    - [Forest Fire](#forest-fire)
  - [Random Walk Variants](#random-walk-variants)
    - [Vanilla](#vanilla)
    - [Metropolis-Hastings](#metropolis-hastings)
    - [Re-Weighted](#re-weighted)
    - [Neighbour Reservoir Sampling](#neighbour-reservoir-sampling)


# Hairballs

Hairballs are network structures that are useless. When you visualise them, they often look like a hairball. When you run into a hairball, one of these three things has happened.



<div style="text-align: center;">
    <img src="/hairball.png" alt="dd" width="900" height="350">
</div>

a) Many networks are inferred, and not observed directly.  
b) Your observation of the network is subject to noise.  
c) Your sample of the network is bad or incomplete.

In order to avoid creating these hairballs, we will in this post, talk about bipartite projections, network back-boning and network sampling.

# Bipartite Projections

What is a bipartite projection of a network? When you have a bipartite network with nodes of type $V_1$ and $V_2$ you create a new unipartite network with nodes of just type $V_1$ or $V_2$.

## Simple Weights

Lets demonstrate this method with a Netflix example, a network connecting each user to the movie they have watched. Now the key in Simple Weights lies in recognising that not all edges have the same importance. Two movies that are watched by three common users are more related to each other than two movies that only have one common spectator.

So to project a network with Simple Weights, for each pair of nodes you identify the number of common neighbours they have, and that’s the weight of the edge. In order to create this weighted link, you simply multiply the bipartite adjacency matrix with its transpose. You have to decide which set of nodes you want to project onto. If $A$ is a $|V_1|\times |V_2|$ matrix, then $AA^T$ is a $|V_1| \times |V_1|$ matrix, while $A^TA$ is a $|V_2| \times |V_2|$ one. 

Optionally, you can add an extra step, where you evaluate the statistical significance of the edge weights from your projection, where you create a series of null networks, and keep in your projection only those links significantly exceeding random expectation.  

## Vectorised Projection

### The saturation problem

In order to understand this problem, lets demonstrate with an authors example, a network of authors connected if they have coauthored a scientific paper together. If you only have written one single paper before this, and this is your second collaboration, you are a strong contributor since this represents 50% of your entire scientific output. But if instead, this was your hundredth collaboration, this paper only adds a little to your connection strength. Giving the same weight in these two different scenario is the saturation problem.

<aside>
📌 *We can exploit edge weights to solve the saturation problem.*

</aside>

In a vectorised projection, we want the adjacency matrix to contain non-zero values different than one. Here, we are considering the nodes as vectors, where each row is a node of type $V_1$, and each entry tells us whether or not it is connected to a node of type $V_2$. Simply a vector of zeros and ones. These vectors represent $V_1$ nodes, and we can then calculate the distance between 2 vectors, and this is how similar they are to each other. You can use the **********************euclidean distance**********************, **********************************cosine similarity**********************************, and ****************************************Pearson correlation****************************************.

******************************************************Problems with this approach****************************************************** - first, it is not always immediately obvious how you translate a distance into a similarity while preserving its properties. Also, none of these measures where developed with network data in mind.

## Hyperbolic Weights

In hyperbolic weight, we recognise that hubs contribute less to the connection weight than non-hubs. This method reminds of wimple weight, but here it exaggerates the differences, so that thresholding becomes easier (thresholding, as in establishing a threshold and drop the edges below this minimum acceptable weight). Each common neighbour $z$ contributes $k_z^{-1}$ to the weight in the projection.

## Resource Allocation

here we do the same as in hyperbolic weight, but considering a second step where we also look at the degree of the originating node.

So each common neighbour $z$ that node $u$ has with node $v$ contributes $(k_uk_z)^{-1}$.

For weighted bipartite networks, the entries of $W$ are:

$$
w_{u,v} = \sum_{z \in N_u \cap N_v} \frac{B_{uv}}{k_uk_z}
$$

Where $B$ is your weighted bipartite adjacency matrix. Note that $W$ is not symmetric.

## Random Walks

Here we take the resource allocation to the extreme. Rather than looking at 2-step walks, we look a infinite length random walks. 
Here we use the stationary distribution to estimate the edge weight:
$w_{u,v} = \pi_v A_{u,v}$
Where $A$ is a transition probability matrix.
Note that $A$ here is different than a simple binary adjacency matrix, as it encodes the probabilities of all random walks of length two. 

# Network Backboning

When we are dealing with a hairball, it is possible we want to reduce the network a bit. Network backboning is the process of taking a network that is too dense and removing the connections that are likely to be not significant or strong enough.

There are two categories in network back-boning methods. *Structural* and *statistical* approaches.

- So if you have a hairball network, you might want to sparsify you network. (also called pruning)
- Maybe your network simply has too many connections to be computationally tractable and so you need to filter out the ones that are unlikely to affect your computations.

<aside>
📌 **The difference between summarisation and back-boning:**
Network Back-boning can easily be confused with graph summarisation. Graph summarisation is the task of taking a large complex network and reducing its size so it can be described better.

Network Back-boning you don’t merge nodes like in summarisation, you want to let the strong connections emerge, and preserve all the entities in your data.  
In a nutshell, network back-boning wants to allow you to perform node- and global-level analyses, while graph summarisation only focuses on empowering meso-level analysis, where you lose sight of the single individual nodes.

</aside>

## Structural back-boning approaches

### Naive Thresholding

If we have a weighted network and we want to keep the “strongest connections”, we sort them in decreasing order of intensity. We then decide a threshold (a minimum strength we accept in the network). Everything not meeting the threshold, we ignore.

There are two problems with naive thresholding:

- The first problem is ********broad weight distributions********, that in real world networks, the edge weights distribute broadly in a “fat-tail” highly skewed fashion. This makes it hard to motivate the choice of a threshold.
- ************************************************************Local edge weight correlations************************************************************ - The second problem is that edge weights are usually correlated. This means that there are areas of the network with high edge weights and areas with low weights. If we impose the same threshold everywhere, some nodes will retain all their connections and others will lose all of theirs, without making the structure any clearer.

### Doubly Stochastic

The doubly stochastic approach solves some of the before-mentioned problems. Recall, that a normal ‘stochastic matrix’ is the adjacency matrix normalised, such that the sum of the columns is 1.

A doubly stochastic matrix is a matrix in which the sums of **both rows and columns** sum to 1. We can with this approach threshold the edges without fearing for the issues we mentioned before.

- You should pick the threshold that allows your graph to be a single connected component.
- The downside of this approach is that not all matrices can be transformed into a doubly stochastic matrix. Only strictly positive matrices can.
- Note also, a doubly stochastic matrix must be square!

### High-Salience Skeleton

Some connections are more important than others because they keep the network together in a single component. 

- To build a ********************************************high salience skeleton******************************************** (HSS) we loop over the nodes and we build their shortest-path tree.
- Then we start exploring the graph with a BFS and note the total edge weight of each path.  **Note** a constraint of the structure: It cannot contain a triangle, since it is a tree
- We perform this operation for all nodes in the network and we obtain a set of shortest path trees. We sum them so that each edge now has a new weight: the number of shortest oath trees in which it appears. Te network can now be thresholded with these new weights.

The HSS makes a lot of sense for networks in which paths are meaningful, like infrastructure networks.But it is computationally expensive. 

### Convex Network Reduction

**A subgraph** of a network $G$ is convex if it contains all shortest paths existing in the main network $G$ between its $V' \subseteq V$.

************************************An induced subgraph************************************ is a graph formed from a subset of the vertices
of the graph and all of the edges connecting pairs of vertices in that
subset.

A network is convex if all its induced subgraph are convex. No matter which set of nodes you pick: as long as they are a part of a single connected component, they are all going to be convex. You can make a real world network into a convex network by finding the minimal set of edges to remove to reduce the network into a tree of cliques.

## Statistical backboning approaches

Generally, it is a better idea to find out the edge-generating process, design a null model for it, and test for statistical significance (instead of naive structural approaches).

### Disparity Filter

It takes a *node-centric approach,* so each node has a different threshold to accept or reject its own edges. You model an expected node strength, by for instance taking the average of its edge weights. This method does not take into account that some nodes have inherently stringer connections, DF ignores the weights of the neighbours of a node when deciding whether to keep an edge or not.

### Noise-Corrected

It takes a *edge-centric approach*, but it is very similar to DF. Here each edge has a different threshold it has to clear if it wants to be included in the network.

# Network Sampling

Edge induction and network back-boning is not always good enough. Sometimes the network is just a hairball no matter what you do, and here you extract from your network a smaller version of it. You want to make sure that this subset of the graph is as representative as possible for the whole structure.

There are two network sampling strategies, *******induced******* and ***********topological***********.

## Induced Strategies

### Node induced

In this method, when focusing on the nodes, it means that your are specifying the IDs of a set of nodes that must be in your sample. But, not at all times you know which nodes to choose, so an alternative approach is to select nodes at random. This randomised method might end up reconstructing a disconnected sample. 

### Edge induced

Here you focus on the edges, and generate an induced sample by selecting edges in a network and then crawl their immediate neighbours. You can do this with a random approach (just like the node induced) or use more sophisticated approaches, **********Totally Induced Edges Samples********** (TIES) and **************Partially Induced************** (PIES).

Edge sampling counteracts the degree bias we meet at the node induced approaches. Here it is very likely that the random edge you pick is a low degree node, which usually is very representative. 

Though, a downside with the edge induced approaches is that you cannot easily use it when interfacing yourself with a social media API system. 

## Topological Strategies

In topological strategies, just like in induced strategies, you also start from a random seed, but here you start exploring the graph. You are not limited to the immediate neighbourhood of your seed like in induced sampling. Topological strategies works will with API systems.

There are two families of topological sampling

- A modification of the BFS approach
- Based on random walks

## Topological BFS Variants

### Snowball

This approach is often used in the context of social networks or online communities. The snowball sampling method is useful for collecting data from networks, especially when the complete network is not accessible or when it is impractical to obtain information from every node.

Snowball is BFS, but imposing a cap in the number of connections at a time $k$. 

### Forest Fire

This is like snowball, however, once we get all neighbours of a node, we do not explore them all. Instead, for each of them, we flip a coin and we explore the node with probability $p$.

This method s inspired by the way a forest fire spreads in nature. It's particularly useful for sampling large online social networks, and capturing their evolving structures.

However, it's important to note that the Forest Fire algorithm introduces a bias towards highly connected nodes, as it tends to explore hubs in the network.

## Random Walk Variants

These methods works by taking a random walk through the network, sampling the nodes it encounters.

### Vanilla

You take a individual node, ask who its neighbours are, pick a random, and ask that random node the same question. 

This is a very simple and easy approach, but it comes with some problems. You can end up trapped in an area of the network where you have explored all nodes, and this method suffers from degree bias. This means that high degree nodes are very likely to be sampled, while low degree not so much.

### Metropolis-Hastings

Here we select a neighbour of the currently visited node in the random walk, and then we do not accept it with probability 1. We instead look at its degree, and if it is higher than the one of the node we are visiting, we have a chance of rejecting its neighbour and trying a different one.

The Metropolis-Hastings Random Walk is widely used for sampling from complex probability distributions, especially in Bayesian inference and other statistical applications.

### Re-Weighted

Here we don’t modify the way random walk is performed, but modify the way we look at it. We correct the result of a *vanilla random walk* for the property of interest.

### Neighbour Reservoir Sampling

Neighbour Reservoir Sampling (NRS) method is a mix between the two families, Random Walk and BFS. It happens in two phases.

- First, NRS builds its set of core nodes and connections, then performs a random walk, including in the sample all nodes and edges it finds during this. We then modify this core.
- Then you make a loop. At each iteration, you pick two nodes at random $u$ and $v$. Node $v$ is a member of set $V'$ and $u$ is not, but it is a neighbour of a node in the set.

We want our sample to be a single connected component.