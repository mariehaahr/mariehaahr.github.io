---
title: 'Lecture 10'
description: 'Link prediction (Preferential Attachment, Adamic-Adar, etc.), Signed networks, Multilayer link predtiction, Applying ML to link prediction'
pubDate: 'Nov 06 2023'
heroImage: '/lec10_na.png'
---


**Readings:** "The Atlas for the Aspiring Network Scientist", Chapters 20 to 22 (minus section 20.7)

#### Table of contents
- [**Link Prediction in Simple Graphs**](#link-prediction-in-simple-graphs)
  - [Preferential Attachment (PA)](#preferential-attachment-pa)
  - [Common Neighbour (CN)](#common-neighbour-cn)
  - [Adamic-Adar (AA)](#adamic-adar-aa)
  - [Resource Allocation (RA)](#resource-allocation-ra)
  - [Hierarchical Random Graphs (HRG)](#hierarchical-random-graphs-hrg)
  - [Association Rules](#association-rules)
- [**Link Prediction For Multilayer Graphs**](#link-prediction-for-multilayer-graphs)
  - [Signed Networks](#signed-networks)
    - [Social Balance Theory](#social-balance-theory)
    - [Social Status Theory](#social-status-theory)
    - [Atheoretical Sign Prediction](#atheoretical-sign-prediction)
  - [Generalised Multilayer Link Prediction](#generalised-multilayer-link-prediction)
    - [Layer Independence](#layer-independence)
    - [Blending Layers](#blending-layers)
    - [Multilayer Scores](#multilayer-scores)
- [**Designing an Experiment**](#designing-an-experiment)
  - [Splitting Network Data into Train/Test Sets](#splitting-network-data-into-traintest-sets)
    - [Specific Issues](#specific-issues)

# **Link Prediction in Simple Graphs**

Link prediction is fundamentally a task that involves making claims about the future.  
Link prediction deals with the prediction of new links in a network, so we want to see if we can predict new edges. So, here the network is fundamentally dynamic, it can change its connections.   
Link prediction in simple graphs is the simplest case for link prediction. In single layer networks, all you have to do is asking the question: “who will be the next two people to become friends with each other?” It is a little more complex with multilayer networks.  
In other words, given a network with nodes and edges, we want to know which connection/egde is the most likely to appear in the future.

📌 *Link prediction happens in **three steps. The first** thing you so is to observe the current links. **Next,** on the bases of this observation, you formulate a hypothesis on how nodes decide to link in the network. **Finally** you operationalise this hypothesis: if nodes are created via process x, you apply x to the current status of the network and that will tell you which link is most likely to appear next.*

There are a couple of classical approaches to link prediction:

- Preferential Attachment (PA)
- Common Neighbour (CN)
- Adamic-Adar (AA)
- Hierarchical Random Graph Model (HRGM)
- Graph Evolution Rules (GER)

## Preferential Attachment (PA)

We start with an example of 3 authors, 2 with a lot of collaborations and the third has a few. (Einstein, Curie and Michele). What collaboration is more likely to happen next? We expect that the two high degree hubs connect (Einstein and Curie).The hypothesis in PA is that “rich get richer”. Nodes with lots of edges will attract more edges. When you have new nodes coming into the network, they come in random (but not uniformly), it will prefer to attach to the node with the highest degree. 

If we want to predict links, we have to formulate a hypothesis and then translate it into a score of u connecting to $v$ for any pair of $u, v$ nodes: score$(u, v)$. The probability of connecting two nodes is directly proportional to their current degree: $score(u,v) = k_uk_v$, where $k_u$ and $k_v$ are u’s and v’s degrees.

## Common Neighbour (CN)

It is more likely to collaborate not only if one is good at collaborating, but also if the two people are likely to meet, namely when they have common neighbours. But for the CN model the thing that matters the most is the number of neighbours they share with them.  
CN’s basic theory is that triangles close: the more common neighbours $u$ and $v$ have, the more triangles we can close with a single edge connecting $u$ to $v$. So the likelihood of connecting two nodes is proportional to the number of shared elements in their neighbour sets: $score(u, v) = |Nu \cap Nv|$j, where $N_u$ and $N_v$ are the set of neighbours of $u$ and $v$, respectively. A variant controls for how many neighbours the two nodes have: the same number of common neighbours weighs more if it’s the total set of connections the two neighbours have. This is the Jaccard variant: $score(u, v) =
|N_u \cap N_v|/|N_u \cup N_v|.$ A way to normalise common neighbours. 

**Example:**


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-13_at_10.42.50.png" alt="dd" width="350" height="350">
</div>


The most likely link to appear in the image above is between 8 and 9, since all their neighbours are shared. This is a Jaccard score of 1, whereas the number of common neighbours is 4.

## Adamic-Adar (AA)

Common neighbour has a problem with bandwidth. It applies a social theory: it is not a strong signal to have the big hubs as a neighbour (like Einstein, since he has many connections, and it is not so special). If two nodes don’t have many connections, then it is a more strong signal to have a connection with them (it is more meaningful). You still count the common neighbours, but you weight them (then higher the degree of the common neighbour, the less they count).

> *In AA we say that common neighbours are important, but the hubs contribute less to the link predictions than two common neighbours with no other links, because the hubs do not have enough bandwidth to make the introduction.*
> 

In AA, our score function discounts the contribution of each node with the logarithm of its degree: 

$$
score(u,v)_{\text{AA}}=\sum_{z \in N_u \cap N_v} \frac{1}{\log k_z}
$$

The formula says that, for each common neighbour, instead of counting one – as we do in Common Neighbour when we look at the intersection – we count one over the common neighbour’s degree (log-transformed). So there is not much of a difference between common neighbours with 100 and 101 connections, but a big difference between common neighbours with 1 and 2 connections. 

## Resource Allocation (RA)

This is almost identical to Adamic-Adar. The only difference is that the scaling is assumed to be linear rather than logarithmic (like in AA). Thus Resource Allocation punishes the high-degree common neighbours more heavily than AA.

$$
score(u,v)_{\text{RA}}=\sum_{z \in N_u \cap N_v} \frac{1}{ k_z}
$$

## Hierarchical Random Graphs (HRG)

The main difference with this link prediction approach, as opposed to the other we have read about so far, is that Hierarchical Random Graphs (HRG) look at the entire network, instead of just nodes and their neighbours. 

In practise, we want to group nodes in the same part of the hierarchy if they have a high chance of connecting.

> *In HRG we’re basically saying that communities matter: it is more likely for nodes in the same community to connect.*
> 

Thus we fit the hierarchy and then we say that the likelihood of nodes to connect is proportional to the edge density of the group in which they both are.

If the nodes are far apart, the group containing both nodes might be just the entire network.

## Association Rules

Just like HRG, Graph Evolution Rule Mining (GERM) is a peculiar approach to link prediction. It is a bit harder to implement. GERM looks at any possible network motif (see section 39.1) and counts how many times each appears in the network.

A crucial difference between GERM and whatever we saw so far is that it doesn’t directly assign a similarity score to any two particular nodes. In GERM, rather than iterating over each pair of unconnected nodes to estimate their score, we iterate over some rules and identify which pairs should be connected. It about learning some patterns and make rules from that in order to predict new edges.  
In fact, one could use GERM not only as a link predictor but also as a graph generator.


<div style="text-align: center;">
    <img src="/Screenshot_2023-11-06_at_13.31.13.png" alt="dd" width="650" height="350">
</div>



Last thing about GERM is that we can classify new links coming into a network into three groups:

- **Old-old**  
  - An “old-old” link appearing at time t + 1 connected two nodes that were already present in the network at time t. These are two “old” nodes.  
- **Old-new**
  - You can expect what an “old-new” link is: a link connecting an old node with a node that was not present at time t – a “new” node. 
- **New-new**
  - New nodes can also connect to each other in a “new-new” link.

# **Link Prediction For Multilayer Graphs**

Here we both want to know which two nodes connect, and at the same time we want to know *how*.

## Signed Networks

First, signed networks are a subtype of multilayer networks with strong constraints on the edges. You can only have two edge types: positive and negative. Moreover, these edge types are exclusive: if you have a positive edge between u and v, you cannot have also a negative one – unless the network is directed and the edge direction flows in the opposite way. 

### Social Balance Theory

According to the **Social Balance Theory** positive and negative relationships are balanced.


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-12_at_13.28.28.png" alt="dd" width="550" height="350">
</div>


Here in the figure, (a) and (b) are balanced triangles because they have an odd number of positive relationships.  
You can calculate a **summary statistics** telling how much, on average, you whole network is balanced.  
One way to do this *frustration.* It is a bit complex to calculate:


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-12_at_13.30.30.png" alt="dd" width="350" height="350">
</div>


In the figure above, there are two unbalanced triangles: (1, 2, 3) and (2, 3, 6). Both triangles would turn balanced if we were to flip the sign of edge (2, 3) – or, alternatively, frustration would dissipate if we were to remove the edge altogether. Thus, the frustration of this graph is 1/9, since it contains 9 edges.  
If we find a configuration with three nodes connected by two positive edges, it is overwhelmingly more likely that, in the future, the triangle will close with a positive relationship rather than with a negative one.

### Social Status Theory

There is a competing theory to social balance, which is the status theory. This arises from a different interpretation of the sign. A positive sign in a social setting might mean that the user originating the link feels to be lower status than – and thus giving social credit to – whomever receives the link. Conversely, a negative link is a way for a higher status node to shoot down a lower status one. Note that here we started talking about the direction of an edge, meaning that we have more than four types of triangles. In fact, we have 32. *You can’t have social status theory if you don’t have direction.*

### Atheoretical Sign Prediction

It is similar to GERM. First, using a variation of GERM we free ourselves from the tyranny of the triangles. Second, what social balance and status have in common is that they will make the same prediction no matter the network you are going to have as input.

## Generalised Multilayer Link Prediction

Generalised multilayer link prediction is the task of estimating the likelihood of observing a new link in the network, given the two nodes we want to connect and the layer we want to connect them through. It is not only which nodes will connect now, but also how will they connect?

### Layer Independence

Given the input network, perform single layer link prediction on each of the layer separately. In this case, we count the number of common neighbours (CNs) between pairs of nodes. We then predict the one with the overall highest score. Then, you can create a single ranking table by merging all these predictions.

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-12_at_13.36.13.png" alt="dd" width="550" height="350">
</div>

This is practically a baseline: it will work as long as we have an assumption of independence between the layers.

### Blending Layers

A slightly more sophisticated alternative is to consider the multilayer network as a single structure and perform the estimations on it.  
**Explanation from book:**  
This is based on the estimation of the number of steps required for a random walker starting on $u$ to visit $v$. We can allow the random walker to, at any time, use the inter layer coupling links exactly as if they were normal edges in the network. At that point, a random walker starting from $u$ in layer $l_1$ can and will visit node $v$ in layer $l_2$.  
The information from these meta-paths can be used directly as we just saw, informing a multilayer hitting time. Or we can feed them to a classifier, which is trying to put potential edges in one of two categories: future existing and future non-existing links.

### Multilayer Scores

**The concept of layer relevance:**

- That is a way to tell you that a node u has a strong tendency of connecting through a specific layer. If a layer exclusively hosts many neighbours of $u$, that might mean that it is its preferred channel of connection.

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-12_at_13.40.04.png" alt="dd" width="450" height="350">
</div>


Consider the figure above. We see that the two nodes have many common neighbours in the blue layer. They only have one common neighbour in the red layer. However the blue layer, for both nodes, has a very low exclusive layer relevance. There is no neighbour that we can reach using exclusively blue edges. In fact, in this case, the exclusive  layer relevance is zero.

# **Designing an Experiment**

**Machine Learning in Network Analysis.**

## Splitting Network Data into Train/Test Sets

Recall that you can never ever use the data that trained your model, to test it. What this means in link prediction is that you cannot claim to have predicted a link that was already in your data.

*So how do you divide a network into training and test sets?* If you have temporal information in your edges (time data) you can use earlier edges to predict the later ones. Meaning that your train set only contains links up to time $t$, and the test set only contains links from time $t+1$ on.

If you don’t have time data, you have to do $k$-fold cross validation.

### Specific Issues

Link predictions comes with some peculiarities, that might not be a problem in other machine learning tasks. Here we focus on **size of the search space** and **sparsity of positives.**

**Size of the search space:**  
When you perform link prediction you will often sample your outputs. You will not calculate $score(u, v)$  for every possible $u$, $v$ pair, but you will sample the pairs according to some expectation criterion. Such criterion can be as hard to pin down as the link prediction problem itself.  
Also, when you sample randomly, the test set will not look like the real network, it will be very sparse. The search space i huge $\sim |V|^2$

**Sparsity of positives:**  
We can build a link prediction method that will tell us that no new link will ever appear. If we do so, we would be right 99.99% of the times. The accuracy of the “always negative” model would be 85%…
The usual fix to this problem is building you test set in a balanced way.

**My own notes on this:**  
Since I did my Network Analysis project with link prediction, I have some comments about our approach. We had an undirected and unweighted network of collaborations in research papers. We wanted to see if we could explain the underlying structure of the network via link prediction. In order to test for this, we removed 10% of our edges (our test set) and then tried two methods; Preferential Attachment and Adamic-Adar. Note that we also wanted true negatives (and not only true positives) so we added random edges to the test-set also, in order to have 50/50 existing edges and random non-existing edges. We then applied the link-prediction to the reduced network, in order to see if we could regenerate the edges we had removed. 

> “*Since real networks are sparse, there are more non-edges than edges. Thus a link prediction always predicting non-edge would have high performance. That is why you should balance your test sets, having an equal number of edges and non-edges.”*
>