---
title: 'Lecture 4'
description: 'Random Graphs, Small World Model, Preferential Attachment, Link Selection, Configuration Model, LFR Benchmark, Statistical Signififance'
pubDate: 'Sep 18 2023'
heroImage: '/Screenshot_2022-09-22_at_11.10.02.png'
---

#### Table of contents
- [What are random graphs?](#what-are-random-graphs)
  - [Building Random Graphs](#building-random-graphs)
    - [$G\_{n,m}$](#g_nm)
    - [$G\_{n,p}$](#g_np)
    - [comparison](#comparison)
- [Small World Model](#small-world-model)
- [Preferential Attachment](#preferential-attachment)
    - [Link selection](#link-selection)
- [The Configuration Model](#the-configuration-model)
- [LFR Benchmark Model](#lfr-benchmark-model)
- [Evaluating Statistical Significance](#evaluating-statistical-significance)

Readings: "The Atlas for the Aspiring Network Scientist", Chapters 13 to 16 (minus sections 15.3, 15.4, 16.2)

## What are random graphs?

Random graphs are synthetic graphs, graphs that we create, in order to discover how some properties have arised, or in order to test your algorithms and analyses.

### Building Random Graphs

Random graphs can be split into two categories:

- $G_{n,p}$
- It allows you to define the probability $p$ that two random nodes will connect.
- $G_{n,m}$ allows to fix the number of edges in your final network, m.

#### $G_{n,m}$

You decide first how many nodes the graph should have – which is the *n* parameter – then you decide how many edges - which is the $m$ parameter. You can imagine this model as a bunch of buttons (nodes) and pieces of yarn (edges).

#### $G_{n,p}$

In the $*G_{n,p}$* variation we still say how many nodes we want: *n*. However, rather than saying how many edges we want, we just decide what’s the probability that two nodes are connected to each other: $*p*$.  

Then we consider all possible pairs of nodes, and for each one we toss the coin. If it lands heads we connect the nodes, if it lands tails we do not. 

#### comparison

- $G(n,p)$ is more suitable for studying properties related to probability and random graphs, where the focus is on edge existence and related probabilities.
- $G(n,m)$ is useful when the exact number of edges is of interest, which is beneficial for modelling certain real-world networks where the number of connections is fixed or known.

Since $G_{n,p}$ and $G_{n,m}$  generate graphs with the same properties, it means that $p$ and $m$ must be related.

## Small World Model

In our random generated networks, the clustering coefficient is usually pretty low, and such low clustering usually implies also the absence of communities (which we almost always see in real networks). Real world networks are expected to have a very high clustering.

**The small world model models high clustering**, but its primary target is to explain small distances, which is present in real world networks.

Imagine this picture from Michele’s book, where we have a lot of people standing in a circle, only being able to talk to you neighbours that can hear you, so maybe 2 people to your left, and 2 people to your right. Now we establish a **rewiring probability $\bold{p}$** - the second parameter of the model. 

 However, high clustering does not necessarily mean that you are going to have communities. In fact, a small world model typically doesn’t have them.

- High clustering
- Short path length


<div style="text-align: center;">
    <img src="/Screenshot_2022-09-22_at_11.10.02.png" alt="dd" width="600" height="350">
</div>

Small World:

- Number of nodes in space
- connect nodes with $n$ nearest neighbours
- Rewire connections with probability $p$

## Preferential Attachment

Preferential attachment is a model that can generate a scale-free network (a network that follows a power-law) **This means that a few nodes become extremely well-connected (hubs), while most nodes have only a few connections.** This is important, because the model helps explain the emergence of hubs in real-world networks. You can use this model to kind of explore the formation of hubs, by the way the nodes prefer to attach to more popular nodes, the “rich get richer” saying.

***The key idea is that the probability of a new node forming a connection with an existing node is not uniform; it depends on the degree (number of connections) of the existing nodes.***

<aside>
💡 In other words, popular or well-connected nodes are more likely to attract new connections.

</aside>

#### Link selection

An alternative to preferential attachment is **link selection**. In link selection, the new coming node selects a link at random from the ones that exist in the network (Figure 14.8(a)). Then, it connects with one of the two nodes connected by that edge – choosing uniformly at random between the two (Figure 14.8(b)).

- When you add a node to the network, you randomly select an edge.
- You then flip a coin on which of the two nodes the edge connects, and connect your new node to that.
- Of course the hubs will have a higher probability of getting new nodes, since they have more edges.

## The Configuration Model

Now you can’t reproduce all of the features of a real world networks with these (above) models.

The Configuration Model provides a way to generate a random graph that follows an exactly specified degree distribution.

The way it works, is that:

1. You specify a degree sequence (matching the degree sequence of the network you want to regenerate)
2. Then you force each node to pick a value from the sequence as it degree, until you have chosen all edges from the sequence.

The Configuration Model provides a useful tool for statistical analysis and comparisons, allowing you to study various properties of networks based on a given degree sequence.

<aside>
💡 **The clustering coefficient of a configuration model tends to zero, which is very unrealistic**

</aside>

## LFR Benchmark Model

If you want an even more real-world like way of generating a network, you can use the LFR, since it allows for much more complex network properties. The LFR is good at generating community structure, making it suitable for benchmarking community detection algorithms and studying community-like aspects of networks. 

**Differences from the Configuration Model:**

1. **Degree Distribution vs. Specific Structure**:
    - The Configuration Model focuses primarily on generating random graphs with a specified degree sequence, without considering community structure or other specific network properties.
2. **Community Structure Emulation**:
    - The LFR Benchmark model is designed to generate realistic community structures, you create the networks based on defined communities and inter-community connections.

## Evaluating Statistical Significance

So a problem with analysing very complex networks, it that you’re only working with a single network, so you can’t really conclude if any observation is statistically interesting. We want to have multiple versions if your network, which all have the same fixed properties, (a null model) to test this interesting observation. 

A method to do so can be:

- **************Network Shuffling**************
    - Here you create a null version of your network by something called edge swapping. It works by picking two pairs of connected nodes and rewire the edges, swapping them.
    - Once you have thousands of these null networks, you can test some property and get an idea of how statistically significant the observation is.