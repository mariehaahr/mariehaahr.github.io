---
title: 'Lecture 8'
description: 'Hierarchical communities, overalpping communities, bipartite communities and multilayer communities'
pubDate: 'Oct 23 2023'
heroImage: '/lec8_na.png'
---

**Mandatory Readings**: "The Atlas for the Aspiring Network Scientist", Chapters 33 to 36 (minus sections 33.2, 34.3, 34.4, 35.3, 35.4, and pages 503-505)

#### Table of contents
- [**Hierarchical Community Discovery**](#hierarchical-community-discovery)
  - [Recursive approaches](#recursive-approaches)
    - [Merging](#merging)
    - [Splitting](#splitting)
  - [Density vs Hierarchy](#density-vs-hierarchy)
- [**Overlapping Coverage**](#overlapping-coverage)
  - [Evaluating overlapping communities](#evaluating-overlapping-communities)
    - [What about modularity?](#what-about-modularity)
  - [Adapted approaches](#adapted-approaches)
  - [Other approaches](#other-approaches)
  - [The overlap paradox](#the-overlap-paradox)
- [**Bipartite Community Discovery**](#bipartite-community-discovery)
  - [Evaluating Bipartite Communities](#evaluating-bipartite-communities)
    - [Bipartite modularity](#bipartite-modularity)
  - [Via Projecting](#via-projecting)
- [**Multilayer Community Discovery**](#multilayer-community-discovery)
  - [Flattening](#flattening)
  - [Layer by layer](#layer-by-layer)
  - [Multilayer adaptions](#multilayer-adaptions)
    - [Multilayer modularity](#multilayer-modularity)
  - [Multilayer Density](#multilayer-density)

# **Hierarchical Community Discovery**

How do we consider situations where a network can be divided into communities in different ways and they are all valid?

After finding communities of nodes, you could find communities of communities. This is the **************************hierarchical************************** community problem; how ti create a hierarchy of communities that best describes the structure of your network.

## Recursive approaches

To put it simple: when running a discovery algorithm on your network to find a number og communities, you can interpret those communities as nodes, and then recursively apply the discovery algorithm until you cannot do that anymore.

<div style="text-align: center;">
    <img src="/Screenshot_2023-10-21_at_16.19.55.png" alt="dd" width="650" height="350">
</div>

### Merging

You start from a condition where all your nodes are isolated in their own community and you create a criterion to merge communities. 

It calculates, for each edge, the modularity gain one would get if they were to merge the two nodes in the same community. Then it merges all edges with a positive modularity gain. Now we have a different network for which the expensive modularity gains need to be recomputed.

You repeat the process until you have all nodes in the same community. 

What the algorithm does, in practice, is building a dendrogram of communities from the **bottom up**.

<div style="text-align: center;">
    <img src="/Screenshot_2023-10-21_at_16.29.18.png" alt="dd" width="600" height="350">
</div>

### Splitting

In the splitting approach you do the opposite. You start with all nodes in the same community and you use a criterion to split it up in different communities. This is a ********top-down approach.********

An algorithm using this approach is the Girvan-Newman algorithm, which uses edge betweenness as its criterion to split communities. But since the edge betweenness has to be recalculated every time you alter the network topology, it is extremely computationally heavy.

The top-down algorithm will normally perform all the possible splits and will return you the full structure, rather than the cut that maximises the modularity. This means that yu will have to calculate the modularity of each split yourself, afterwards. 

<div style="text-align: center;">
    <img src="/Screenshot_2023-10-21_at_16.29.05.png" alt="dd" width="600" height="350">
</div>

💡 ****************************************************************************************Higher modularities are better partitions, thus better cuts.****************************************************************************************

## Density vs Hierarchy

As we know, our classical definition of a good partition is when you maximise the internal density within the community. But these hierarchical approaches poses some issues with this.

Not all communities at all levels of the hierarchy maximise the internal density, so even if totally valid hierarchical communities spite our classical definition of communities based on edge density.

# **Overlapping Coverage**

Communities normally seem to imply that communities are a clear cut case. This is often not the fact. This means that nodes can be part of multiple communities at the same time.

None of the methods we have seen this far allow for such overlapping. The formula for modularity doesn’t work either, because of the Kronecker delta.

## Evaluating overlapping communities

Here we can used Normalised mutual information (NMI), we just need to make it accept an overlapping coverage. We compare two bipartite matrices.


<div style="text-align: center;">
    <img src="/Screenshot_2023-10-22_at_17.22.49.png" alt="dd" width="600" height="350">
</div>

### What about modularity?

In order to account for overlapping communities with the modularity measure, we have different solutions. One solution is replacing the binary Kronecker delta with a continuous node similarity measure based on the either the product or the average of how much a node really belongs to a community. Another simple solution could be to simply calculate the average modularity of all communities; to a version incorporating both overlap and directed edges.

## Adapted approaches

Instead of using our algorithms that maximise modularity (which we know from normal community discovery) we adapt the approaches to make them maximise overlapping modularity instead. 

It is also possible to adapt multiple other algorithms, such as Infomap and label propagation.

## Other approaches

An interesting approach to overlapping community discovery is the Order Statistics Local Optimisation Method (OSLOM). Shortly explained, its philosophy is close to modularity, and it builds an expected number of edges in a community by means of a configuration model.

************Recall:************ A configuration model is a random model that provides a way to generate a random graph that follows an exactly specified degree distribution.

OSLOM attempts to establish the statistical significance of the partition, so how likely is it to observe the given community subgraphs in a configuration model. So therefore, the less likely a node is to be included in a community in a null model (the configuration model), the more likely it is that we should add it to the community. The OSLOM can be used as a post-processing strategy, to refine the communities you have found using another method.

## The overlap paradox

The paradox illustrates the complexity of overlapping communities in real-world networks, where nodes have multiple affiliations or roles. Striking the right balance between keeping communities dense internally and sparser outside, particularly when dealing with nodes that belong to more than one community, poses a significant challenge.

If it is true that the more communities two nodes share
the more likely they are connected, then the overlap of multiple communities is denser than the communities themselves, i.e. there are more links going outside communities than inside.

# **Bipartite Community Discovery**

In bipartite networks, nodes of the same type cannot connect to each other. However, they could still be in the same community, because they have lots of common neighbours. Thus we need to adapt modularity and other community quality measures to take this into account.

## Evaluating Bipartite Communities

### Bipartite modularity

We know from the ********************************original version******************************** of modularity, it follows that we want to partition the network into communities that are denser than what we would expect given a null model (the configuration model). So the thing we need to change for it to work on bipartite networks, is the configuration model.

The only thing we change is the ****configuration model connection probability**** from the original, which was $\frac{k_uk_v}{2|E|}$, and change it to $\frac{k_vk_u}{|E|}$, with the added constraints that $u$ and $v$ needs to be nodes of from different sets of nodes in the bipartite network.

Now you can use any of the modularity maximisation algorithms we know from normal community discovery.

## Via Projecting

One way to perform bipartite community discovery is by projecting the bipartite network into uni-partite form and then perform normal community discovery there. However, the resulting network will be too dense and we will lose information in the projection due to maybe unclear mapping.

# **Multilayer Community Discovery**

***********In multilayer networks we want to find communities that span across layers.***********

## Flattening

This is the simple solution to multilayer community discovery. You simply collapse all nodes across the layers, so you reduce the network into a single-layer network with some weighted egdes.

There are different approaches to weight these edges. You can either simply count the number of layers where the connection between nodes appear, and use that as weight. Another approach is counting the number of common neighbours that two nodes have and use that as weight of the layer. The last method is **differential flattening**, where you flatten the multilayer graph into the single layer version of it such that its clustering coefficient if maximised. 

The problem with these simple flattening methods is the same as with bipartite projection in community discovery, we lose information. 

## Layer by layer

In this approach, we perform community discovery separately on each layer of your network and then combine your results. Again, this is also quite a simple way of finding communities in a multilayer network, and it is not ideal.

In one approach you build a matrix where each row is a node and each column is the partition assignment for that node in a specific layer. Then you perform kMeans clustering on it, finding clusters of nodes that are similar across layers,

A similar approach uses frequent pattern mining, where each node can be represented as a simple list of community affiliations, and then we look for sets of communities that are frequently together: these are communities sharing nodes across layers.

## Multilayer adaptions

### Multilayer modularity

If we were to represent the multilayer network as a flat network, the new node is not densely connected to the rest of the triangle: a node couples only with itself, not with its community fellows. So the coupling edges have to count in some special way. So, a node couples only with itself in a different layer, not connecting to its community members, making a multilayer community sparser than it actually is, which we need to account for. 

The major complication in multilayer modularity is that we have many Kronecker deltas $({\delta})$.



<div style="text-align: center;">
    <img src="/Screenshot_2023-10-23_at_13.47.18.png" alt="dd" width="550" height="350">
</div>

## Multilayer Density

Multilayer density is a ambiguous concepts, since you have to decide what it means to be dense yourself. Is a group of nodes densely connected when they are densely connected looking at all layers, or only one layer at a time?



<div style="text-align: center;">
    <img src="/Screenshot_2023-10-23_at_13.52.32.png" alt="dd" width="650" height="350">
</div>

You have to make your own judgement on how you want to interpret the density in your multilayer network, depending on your type of analysis and type of data.