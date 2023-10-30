---
title: 'Lecture 9'
description: 'Spreading processes, SI, SIS, SIR, Complex contagion & Catastrophic failures'
pubDate: 'Oct 23 2023'
heroImage: '/lec9_na.png'
---


**Readings:** The Atlas for the Aspiring Network Scientist, Chapters 17 to 19

#### Table of contents
- [**Epidemics**](#epidemics)
  - [SI](#si)
    - [Homogenous mixing](#homogenous-mixing)
    - [Without homogenous mixing](#without-homogenous-mixing)
  - [SIS](#sis)
  - [SIR](#sir)
- [**Complex Contagion**](#complex-contagion)
  - [Triggers](#triggers)
    - [The threshold model](#the-threshold-model)
    - [The cascade model](#the-cascade-model)
  - [Limited Infection Chances](#limited-infection-chances)
  - [Interventions](#interventions)
  - [Controllability](#controllability)
- [**Catastrophic Failures**](#catastrophic-failures)
  - [Random failures](#random-failures)
  - [Targeted attacks](#targeted-attacks)
  - [Chain effects](#chain-effects)
  - [Interdependent Networks](#interdependent-networks)

# **Epidemics**

This section deals with a different type og dynamics in a network, where the edges doesn’t change, but the state of the node does. The easiest metaphor for understanding these processes, is via disease.

There are different sates in which a person might find themselves in, and like these individuals, nodes can change too as time passes.

## SI

Researchers have developed simple models to describe the dynamics of diseases. They are also know as **Compartmental models** or **State models**.

Here the $S$ stands for Susceptible, which is not infected, but in a state that is susceptible to contract a disease. And $I$ stands for infected. The model allows only for one possible transition between states. $S$  → $I$. So an $SI$ model is only for diseases with no recovery (like herpes). 

There is one assumption underlying the traditional $SI$ model, homogenous mixing.

### Homogenous mixing

Here we assume that each susceptible individual has the same probability to come into contact with an infected person. It is determined by the current fraction of the population in the infected state. $\frac{|S|}{|V|}$($|V|$ is number of nodes, and $|I|$ are infected individuals) with $\bar k$ meeting (average degree).

$$
\text{Total number of meetings} = \bar k\frac{|I||S|}{|V|}
$$

The probability of a susceptible person will transition into an infected after being in contact with an infected is a parameter of the model, normally indicated with $\beta$. If $\beta = 1$, then any contact with an infected till transmit the disease, while $\beta = 0.2$, then you have a 20% risk to contract the disease. 

If you have a couple of sick people randomly put into a society then you can track the ratio of sick people as time goes by:

$$
|I|/(|I| + |S|)
$$

At first the ratio grows slowly, then as soon as $I$ expands a little, we see an exponential growth. After a critical point, the growth slows down, because there are not many healthy people left. 

<div style="text-align: center;">
    <img src="/Screenshot_2023-10-30_at_13.57.26.png" alt="dd" width="650" height="350">
</div>

No matter what value $\beta$ is, all SI models will end up in complete infection. $\beta$  only determines the speed of the system. 

Since each meeting has a probability $\beta$ of passing the disease, at each time step there are:

$$
\text{New infected peiple in I} = \beta \bar k\frac{|I||S|}{|V|}
$$

This can be simplified into:

$$
i_{t+1} = \beta \bar k i_t (1-i_t)
$$

Where $t$ is the current time step.

The mathematical solution to the SI model with homogenous mixing integrated over time:

$$
i = \frac{i_0e^{\beta \bar k t}}{1-i_0+i_0e^{\beta \bar k t}}
$$

In **homogenous mixing** we go with the assumption that the global network is a lattice (everybody is the same) a regular grid where each node is connected only to its immediate neighbours. Here ********hubs******** will allow diseases to reach the rest of the network quickly. 

### Without homogenous mixing

Here we have to group the nodes by their degree k.

$$
\text{Network-aware SI model} = \beta kf_k(1-i_{k,t})
$$

The two differences of the formulas:

1. We replace the average degree $\bar k$ with the actual node’s degree $k$.
2. Rather than using $i_{k,t}$ we use $f_k$ a function of the degree $k$.

Though it has the same functional form as the homogenous mixing, the exponential rises faster at the beginning (due to outliers) but the rising and falling off the infection rates is still exponential. 

But we also need to take degree distributions into account.

The hubs of the network contributes enormously to the speed of the infection. Networks with a degree distribution characterised by a low $\alpha$ exponent are infected more quickly.

## SIS

The states are the same as in the $SI$ model, but here we have a possibility of recovering again. Here we have $(S$ → $I)$ and $(I$ → $S)$ an infection rate: $\beta$, and a recovery rate: $\mu$.

Here we need a new parameter. Recovery rate $\mu$. If you get recovered, you can get the disease again. 

********************Be aware:******************** For SIS models, not every note will be status $I$ eventually (like in $SI$). The rate at which infected people recover and the infection rate are perfectly balanced (the endemic state). 

$$
\text{SIS model} = i_{t+1}= \beta \bar k i_t(1-i_t) -\mu i_t
$$

Eventually $\beta \bar k i_t (1-i_t) = \mu i_t$ , and that is when the share of infected nodes $i$ doesn’t grow any more. We reached the endemic state. 

It is possible that: $\beta \bar k i_t (1-i_t) < \mu i_t$ where people are recovering faster than the new infected pop-up. *We can get rid of the disease!*

<div style="text-align: center;">
    <img src="/Screenshot_2022-09-07_at_09.59.23.png" alt="dd" width="650" height="350">
</div>

There must be a critical state of $\lambda$  $(\lambda = \beta / \mu$ ) that makes us transition between the endemic and non-endemic state.

## SIR

Next step is modelling epidemic on networks is by considering those diseases you can catch only once in your lifetime. $R$ stands for remove. You get removed from the outbreak if you either die or survive and now is immune. Here we only have $(S$→$I)$ and $(I$→$R)$.

The typical evolution of an $SIR$ model: after an initial exponential growth of the infected, the removed ratio takes over until it occupies the entire network.

The defining characteristic og an $SIR$ model is its lack of endemicity, either the disease kills everybody or every individual still alive has now healed from the disease.

<div style="text-align: center;">
    <img src="/Screenshot_2022-09-07_at_10.10.32.png" alt="dd" width="450" height="350">
</div>

Note for this image above, where eventually there are only people in the $R$ state, is not the only possible. If you are lucky, $I$ empties out before $S$, meaning that the disease dies out in $R$ before every susceptible individual has had the privilege of trying to be infected.

The evolution of the recovery ratio $r=|R|/|V|$ is very simple. 

# **Complex Contagion**

Every time a healthy person comes in contact with an infected person, they have the chance to become infected as well. But if this doesn’t happen, the healthy person is still in the healthy pool. This next step is a new occasion: for them to contract the disease. This limitation in $SI$, $SIS$ and $SIR$ allows them to model only rather specific types of outbreak. 

In **simple contagion** at each time step you have the same chance of getting infected of you have one or more infected neighbours. In **complex contagion** more infected neighbours reinforce the infection chances.

## Triggers

The main difference between simple and complex contagion is that, in the latter, you require reinforcement.

This means that, a single contact may or may not infect you, but if you have multiple contacts your likelihood to transition into the infected state $I$ grows. 

There are 2 types of reinforcement:

- Cascade
- Threshold

But lets start out with a **simple classical reinforcement**. Here, more infected neighbours mean more chances of infection. If you have $n$ sick friends, and you visit them one by one, at each visit you toss a coin.

If our parameter $\beta$ tells us the probability of being infected by a single contact, then $(1-\beta)$ is the probability of not being infected. Since the coin tosses are all independent, the probability of never being infected by any of the $n$ contacts is $(1-\beta)^n$. So the probability that at least one contact will infect is is $1-((1-\beta)^n)$.

So the **whole difference between simple $SI$ and classical complex $SI$** is that the latter depends on $n$, the number of your friends that are infected.

In classical reinforcement, it is easy to infect hubs. This is what generates super-exponential outbreak growth in power law models with large hubs.

### The threshold model

In the threshold model you need at least $K$ neighbours, independently of your degree, to transition. To be infected, you need multiple infected neighbours. The threshold models adds a parameter $K$. If more than $K$ of your neighbours are infected, then they pass the infection to you.

The threshold model is where epidemiology starts to blend in with sociology. This threshold assumption works well when explaining the spread of behaviour between people. The assumption is that individuals behaviour depends on the number of other individuals already engaging in that behaviour.

You can spice up the threshold model by allowing $K$ to be a node-dependent parameter, rather than a global one, which means that each node $v$ has a different $K_v$ activation threshold.

### The cascade model

In the cascade model you need at least a fraction of your neighbours to transition. You also need reinforcement from more than one neighbour to transition to the infected state $I$. However, while in the threshold model this was governed by an absolute parameter $K$, here we use a relative one. You need a fraction of your neighbours to be infected in order for you to be infected. So the size of your neighbourhood influences your likelihood to transition.

**Why do we separate the cascade and threshold? Because of hubs.** In a threshold model, hubs are the primary spreaders of the disease. In a cascade model, they are the last bastion if defence. Once the hubs fall, there is no more chance for salvation. 

$\beta$ is not $p(S$→ $I)$ anymore, but it’s the fraction of neighbours needed to infect you. 

## Limited Infection Chances

When not talking about diseases spreading, that we want to prevent, we could talk about social networks, and wanting something to spread fast. For example a product you want to sell, popularity.

## Interventions

In practice, if this was a $SIR$ model, we want to flip some people directly from the $S$ to the $R$ state, without passing by $I$. This is equivalent to vaccinate them and of this is done properly, it would stop the epidemics in its tracks. 

**So who should we vaccinate? the hubs.**

The "vaccinate a friend" strategy (or related, the "friendship paradox") has implications in disease spreading within networks. The friendship paradox highlights that in a social network, your friends, on average, tend to have more friends than you do. In terms of disease spreading and vaccination strategies, this phenomenon has interesting implications.

<div style="text-align: center;">
    <img src="/Screenshot_2023-10-30_at_14.33.13.png" alt="dd" width="450" height="350">
</div>

One effective strategy to prevent a global outbreak is to immunize the friends of randomly chosen nodes. This strategy works because the randomly picked neighbors of randomly picked nodes are more likely to be hubs

## Controllability

The controllability of complex networks, the task is: nodes can change their state freely and there can be an arbitrary number of states in the network. What we want to ensure is that all (or most nodes) nodes in the network end up in the state we desire. By targeting vaccination towards individuals' friends or highly connected nodes, the aim is to strategically contain disease spread.

# **Catastrophic Failures**

Now we change the perspective a bit, and forget about disease spreading. We want to study the conditions under which networks break down for some reason. Imagine if information can still flow through the internet if some routers go down, or how many blocked roads does it take for cars not to be able to drive around. 

Networks are usually resilient to small failures. However when nodes start breaking down in multiple components and getting isolated, then the network is failing. 

## Random failures

We are assuming that all the network’s nodes are in the same Giant Connected Component (GCC). When will the network lose its giant component? When will we need to rely on local generators rather than on the entire grid?

Having a heavy tailed degree distribution means to have vary few gigantic hubs and a vast majority of nodes of low degree. When you pick a node at random and you make it fail, you are overwhelmingly more likely to pick one of the peripheral low degree ones.

It is extremely unlikely to pick the hub, which would be catastrophic for the networks’ connectivity.

More common hubs equals higher likelihood of picking them up in a random extraction. Thus the failure functions for different $\alpha$ values follow the pattern $I$ shown in this figure:

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-05_at_15.59.42.png" alt="dd" width="450" height="350">
</div>

💡 **Power law random networks are more resilient than $G_{n,p}$ networks to random failures.**

## Targeted attacks

What if we are not observing random failures, but a deliberate attack from an hostile force? They would go after the nodes allowing them to maximise the amount of damage while minimising the effort required.

As you would might expect, the $\alpha$ exponent of the power law degree distribution has something to do with the fragility of a network or deliberate attacks. However, it is a non-linear relationship, which also depends in the minimum degree of the network $k_{min}$.

## Chain effects

In this scenario, the failure of one node propagates in a cascade and causes more correlated failures. This sort of snowball effect can turn into an avalanche and shut the entire network down. An example could be airline schedules, if one departure gets cancelled it can shut down a lot of other trips.

All nodes start at state $S$. They are characterised by a current load and by a total capacity. Think of this as road intersections: the load is how many cars pass on average through the intersection and the capacity is the maximum amount that it can pass before congestion happens. 

So when a node fails, all its load needs to be redistributed to non-failing nodes. This can and will make the failure propagate on the network in a cascade event which might end up bringing the entire network down.

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-05_at_16.13.14.png" alt="dd" width="650" height="350">
</div>

## Interdependent Networks

In independent networks we have a multilayer network whose nodes in one layer are required for the functioning of nodes in the others. Depending on the degree correlations among layers, failures can propagate across layer and bring down power law networks even under random accidents. 

If you have two interdependent networks: the power grid needs computers to work and the computers need power to work. Even if the two networks are resilient to random failures in isolation, the inter-dependencies cause them to be fragile to failures propagating back and forth between them.