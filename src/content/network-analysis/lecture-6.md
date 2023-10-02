---
title: 'Lecture 6'
description: 'Homophily, Assortativity, Core-periphery structure'
pubDate: 'Sep 25 2023'
heroImage: '/continuous_CP.png'
---


Homophily, Assortativity, Core-periphery structure

#### Table of contents
- [Mesoscale Analysis](#mesoscale-analysis)
  - [Homophily](#homophily)
  - [Ego Networks](#ego-networks)
  - [Assortativity \& Disassortativity (Qualitative)](#assortativity--disassortativity-qualitative)
    - [More than 1 attribute](#more-than-1-attribute)
    - [Strength of weak ties](#strength-of-weak-ties)
  - [Homophily \& Social Contagion](#homophily--social-contagion)
- [Quantitative Assortativity](#quantitative-assortativity)
  - [Degree Correlations](#degree-correlations)
    - [The Friendship Paradox](#the-friendship-paradox)
- [Core-Periphery](#core-periphery)
  - [Models](#models)
    - [Discrete](#discrete)
    - [Continuous](#continuous)
  - [Nestedness](#nestedness)
- [Hierarchies](#hierarchies)
  - [Types of Hierarchies](#types-of-hierarchies)
  - [Order](#order)
  - [Nested](#nested)
  - [Flow](#flow)
    - [Cycles](#cycles)
    - [Global Reach Centrality](#global-reach-centrality)
    - [Arborescences](#arborescences)


# Mesoscale Analysis

In the mesoscale analysis, we operate at the level that lies between the global analysis (topological characteristics, degree distr., CC etc.) and the local analysis (individual node’s characteristics, closeness centrality, degree etc.).

At the mesoscale we want to describe groups of nodes. **Community discovery** is the most popular mesoscale analysis to do, but this section will be about all the other ones.

## Homophily

**********The concept of homophily:********** people will tend to associate with people with similar characteristics as their own. In a social network, homophily implies that nodes establish edges among them if they have similar attributes. (Real world: segregation is a negative example of this).

> ***“Birds of a feather flock together”***

## Ego Networks

This is a simple way of exploring the meso-level around a node. You simply select a node which will be your “ego” node, and you select all of its neighbours, and the resulting network is you ego network.

Ego networks are frequently used in social network analysis (e.g. by estimating a person’s social capital).

## Assortativity & Disassortativity (Qualitative)

If we want to quantify the homophily, we can use Assortativity. Here I will explain the scenario where the attributes driving the connections are qualitative (e.g. gender etc.).

In a simple scenario, you can estimate the probability of an edge to connect alike nodes (nodes with the same attribute), and compare it to the probability of connection in the network. 

If we have a probability higher than 1 it implies homophily, meaning that nodes tend to connect to other nodes with the same attribute. And, of course, a probability below 1 then implies that the network is disassortative, and nodes don’t like to connect to similar nodes.

<aside>
💡 This simple method does not work when you have more that 1 possible attributes in your network.

</aside>

### More than 1 attribute

In this case you want to look at the probability of edges connecting alike nodes per attribute value $i$, and then compare it to the probability of an edge to have at least one node with attribute value $i$.

$$
r =\frac{\sum_i e_{ii}- \sum_i a_ib_i}{1- \sum_i a_ib_i} \\ -1 : \text{ perfect disassortativity} \\ 1:\text{ perfect assortativity}
$$

Where $e_{ii}$ is the probability of an edge to connect two nodes which both have value $i$, $a_i$ is the probability that an edge has as origin a node with value $i$, and lastly, $b_i$ is the probability that an edge has as destination a node with value $i$.

### Strength of weak ties

> *It’s rarely your closest friends who make you find a job, but that far acquaintance with whom you rarely speak, because your close friends usually access the same information as you do and so cannot tell you anything new*
> 

## Homophily & Social Contagion

**Heterophily**: *“The love of the different”,* people tend to find partners with the same attributes (hobbies etc.) but we also see some of the attributes being the opposite, like gender.

Since humans are social animals and tend to succumb to peer pressure, homophily can be a channel for behavioural changes. Such as obesity, if you have an obese friend, it increases the likelihood of you becoming obese, because habits are contagious. A social virus.

# Quantitative Assortativity

Before we talked about qualitative homophily, people with the same sex, gender, race tend to like each other. Now we talk about quantitative homophily, people with the same age, number of friends etc. tend to like each other.

<aside>
💡 Here we refer to quantitative homophily as assortativity and of course we refer to quantitative heterophily as disassortativity.

</aside>

## Degree Correlations

When measuring the degree correlations, we measure the extent to which nodes in a network tend to connect to nodes with a similar degree.

In a degree-assortative network we see that hubs connect preferentially to hubs, while peripheral nodes connect preferentially to other peripheral nodes.

We measure whether nodes with high degrees tend to be connected to other nodes with high degrees (positive assortativity) or to nodes with low degrees (negative assortativity).

- Visualising: edge scatterplot
- A way to visualise degree assortativity is to consider each edge as an observation. We create a scatter-gram, recording the degree of one node on the x-axis and of the other node on the y-axis.
- Another way to visualise the degree assortativity: The scatter plot has a point for each node in the network, reporting its degree (x axis) against the average degree of its neighbours (y axis).

### The Friendship Paradox

**Your friends are, on average, more popular than you.** It is a funny phenomenon that highlights the statistical “quirks” present in social networks.

# Core-Periphery

Usually large scale networks have a common topology: a very densely connected set of core nodes, and a bunch of casual nodes attaching only to few neighbours (you know, a hairball). 

************Core:************ being part of a more dense part of the network (the rich club), compared to the ******************periphery****************** where the network is not dense. 

Core-periphery networks emerge when all nodes belong to a single group. Some nodes are well connected while others, are not (even though they are still a part of that group).

## Models

Different way to extract core-periphery structures from your networks. But the two methods that are used the most, are the **********continuous model *********************and the ***************discrete model.***************

### Discrete

In a pure idealised core-periphery network we can classify the nodes into 2 classes; the first one is the core nodes (C) with high interconnectedness. The second one is the periphery nodes (P), the ones that are only sparsely connected in the network. 

The discrete model is **********************very strict**********************, and there can only be 2 types of connections; either C→C or C→P. We rarely see real world networks to be able to follow this standard. 

You want to maximise a simple quality measure, in order to detect the core-periphery structure:

Maximise: $\sum_{uv} A_{ij} \Delta_{uv}$, where $A$ is the adjacency matrix, $\Delta$ is a matrix with a value per node pair. So an entry in $\Delta$ is equal to 1 if either of the two nodes is part of the core. 

### Continuous

Here we take in to consideration a “semi-periphery”. Mathematically speaking, there is not such a big difference between the discrete and the continuous model. 

<div style="text-align: center;">
    <img src="/continuous_CP.png" alt="dd" width="500" height="350">
</div>

Looking at the example, we have a densely connected core (blue), a pure periphery whose nodes do not connect to each other (green), and an intermediate stage which is not dense enough to be part of the core, but whose nodes still connect to each other
(purple).

The quality measure we want to maximise is still $\sum_{uv} A_{ij} \Delta_{uv}$ . **************However************** the entries of $\Delta$ are not binary anymore. Instead we have $\Delta_{uv} = c_u c_v$, with $c_u$ being the coreness value of $u$.  

## Nestedness

An ideal nested system has a hub which is connected to all nodes in the network. The node with the second largest degree is then connected to a subset of the hub’s neighbours. And again, the third largest degree node is then connected to a subset of the second largest degree node’s neighbour. 

A typical nested network is ********************bipartite.******************** The nestedness in core-periphery structure helps us identifying the dynamics of complex systems, such as ecological networks, social networks, or biological interactions. It's a valuable concept for analysing bipartite networks and finding hierarchical patterns.

# Hierarchies

Hierarchical networks arise from both social systems and especially in biological systems. If we compare this to the ****************************core-periphery**************************** structure where horizontal connections between different levels are allowed, in hierarchies, this is not allowed! In an organisation, this would be a worker can only connect to a lower level )hierarchical networks are usually directed). 

## Types of Hierarchies

We categorise hierarchical networks by; order, nested, and flow hierarchy.

## Order

In an order hierarchy, we want to determine the order in which to sort the nodes. We usually sort them according to some topology of the node’s connections, usually a centrality score. The most central nodes are then placed on top and the least central on the bottom.


<div style="text-align: center;">
    <img src="/lec6_1.png" alt="dd" width="500" height="350">
</div>

Here we see a network on the left, and it’s order hierarchy on the right, ordered by betweenness centrality.

## Nested

Nested hierarchy we start off with all nodes in one group, then divide the network into higher-order groups, which each contain lower-order groups, and continuing this until where down to each of the single nodes.

<div style="text-align: center;">
    <img src="/lec6_2.png" alt="dd" width="500" height="350">
</div>

A network on the left showing the different groups (blue: higher-order, green: lower-order) and it’s nested hierarchy on the right.

## Flow

In a flow hierarchy, nodes in a higher level connect to nodes at the level directly beneath it, and can be seen as managers spreading information to the lower levels in a company. It is called “flow” hierarchy because you can see the highest level node as the “origin” of flow.

<div style="text-align: center;">
    <img src="/lec6_3.png" alt="dd" width="500" height="350">
</div>

**The rest of the section will be about flow hierarchies in a directed network.**

### Cycles

Cycles are an enemy of a hierarchies, since in a perfect hierarchy, you can always tell who’s your boss and your boss will never take order from you, nor from ant of your underlings.

To estimate the “hierarchical-ness” of your directed network, you count the number of edges involved in a cycle. The fewer edges are part of cycle, the more hierarchical it is.

****************************************************************Calculating the flow hierarchical-ness:**************************************************************** here you simply condense the graph. Wherever the graph has cycles (strongly connected components) you condense that component into a single node. Here you have to keep track if there are multiple edges going from A to B that are being merged, then you have to increase the edge weight to the number of edges being merged. The figure below shows an example (see the 2 at the merged edges).

 
<div style="text-align: center;">
    <img src="/lec6_4.png" alt="dd" width="500" height="350">
</div>

You then calculate the flow hierarchical-ness by taking the ratio between the sum of the edge weight in the condensed version (the one to the right) and the sum of the edge weight in your normal network (simply the number of edges). In this case this would be $12/20 = 0.6$, since there are 20 edges with a weight of 1 in the original graph and a sum of 12 in the condensed version of the network.

### Global Reach Centrality

************Recall:************ the local reach centrality of node $v$ in a directed network is the fraction of nodes it can reach using directed paths originating from itself.

Now the global reach centrality **is not a measure for nodes** (like the local one), it is a way to estimate the hierarchical-ness of a network.

**************Intuition:************** A network has a strong “global reach centrality ($GRC$) hierarchical-ness” if there is a node which has an overwhelming reaching power compared to the average of all other nodes (A little CEO that sees all and knows all).

To calculate this you estimate the local reach centrality of all nodes, and then find the maximum value among them $LRC_{MAX}$. From there, you can calculate the $GRC$:

$$
GRC= \frac{1}{|V|-1} \sum_{v \in V} LRC_{MAX} - LRC_{v}
$$

### Arborescences

**************Recall:************** an arborescence is a directed tree in which all nodes have in-degree of one, except the root, which has in-degree of 0. Every arborescence is a perfect hierarchy: all nodes have a single boss, there are no cycles, and there is one node with no bosses: the CEO.

So Arborescences is well-suited to inform us about hierarchies. 

**Using** Arborescences **to estimate hierarchical-ness (Michele’s own score):**

- Take you directed graph and condense it, ignoring edge weights in merged edges. This gives us a directed acyclic graph version of the original network.
- Then we remove all of the edges going against the flow. We cannot have nodes with an in-degree higher than 1, so we remove the edges that contributes to this. This is done by removing the ones that point towards the root. You can use closeness centrality to determine which edges to keep.
- Now we count how many connections survived all of the above pruning. So taking the previous example, we would have preserved 9 edges out of the 20 from the original graph, giving us $9/20 = 0.45$.