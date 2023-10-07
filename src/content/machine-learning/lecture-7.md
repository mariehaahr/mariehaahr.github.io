---
title: 'Lecture 7'
description: 'Decision Theory'
pubDate: 'Oct 01 2023'
heroImage: '/lec7_ml.png'
---


**Readings**: 1.5 in PRML, 4.4, 4.5 in ISLwR

#### Table of contents 
- [1.5 Decision Theory](#15-decision-theory)
  - [1.5.1 Minimising the Misclassification Rate](#151-minimising-the-misclassification-rate)
  - [1.5.2 Minimising the Expected Loss](#152-minimising-the-expected-loss)
    - [Loss Functions](#loss-functions)
  - [1.5.3 The Reject Option](#153-the-reject-option)
  - [1.5.4 Inference and Decision](#154-inference-and-decision)
  - [1.5.5 Loss Functions for Regression](#155-loss-functions-for-regression)
    - [3 approaches to solving regression problems](#3-approaches-to-solving-regression-problems)
  - [Notes from lecture](#notes-from-lecture)
    - [Joint distribution of data](#joint-distribution-of-data)
    - [Posterior and Prior Probability](#posterior-and-prior-probability)

# 1.5 Decision Theory

**General Machine Learning:**

Suppose we have an input vector $x$ together with a corresponding vector $t$ of target variables, and our goal is to predict $t$ given a new value for $x$. For regression
problems, $t$ will comprise continuous variables, whereas for classification problems $t$ will represent class labels. The joint probability distribution $p(x, t)$ provides a complete summary of the uncertainty associated with these variables. Determination of $p(x, t)$ from a set of training data is an example of *inference.*

So the problem lies in determining the joint distribution $p(x, C_k)$, which gives us the most exact probabilistic description of the situation.
**

We are interested in the probabilities of the two classes given given the image (here determining cancer from a picture) which are given by $p(C_k|x)$. Using Bayes Theorem, these probabilities can be expressed in the form:

$$
p(C_k|x)(\text{Posterior Probability}) = \frac{p(x|C_k)p(C_k)}{p(x)}
$$

Here, $p(C_k)$  is the prior probability for the class $C_k$. In the cancer example, this would be the proability that a person has cancer, before seeing the picture (so just a general probability).

Our aim is to minimise the probability of misclassification, soo intuitively we would choose the highest posterior probability. 

## 1.5.1 Minimising the Misclassification Rate

A mistake occurs when an input vector belonging to class 1 is assigned to class 2 and vice versa. To minimise the probability of this occurring, we should arrange that each x is assigned to whichever class has the largest posterior probability $p(C_k|x)$

## 1.5.2 Minimising the Expected Loss

When misclassification happens, in situations like determining cancer, we have two types of consequences, but they are dramatically different (one is too many resources used and the other could be death). 

### Loss Functions

We can formalise such issues through the introduction of loss functions (also called cost functions). They are a single, overall measure of loss incurred in taking any of the available decisions or actions.  Sometimes referred to as the utility function, which we wish to maximise. We want a solution that minimises the loss function, but the loss function, though, depends on the true class, which we don’t know. Therefore we seek instead to minimise the average loss, where the average is computed with respect to this distribution:

$$
E(L)=\sum_k \sum_j \int_{R_j} L_{kj}p(x,C_k)dx
$$

Our goal is to choose the regions $R_j$ in order to minimise the expected loss, which mean we want to minimise $\sum_k L_{kj}p(x,C_k)$ for each $x$.

## 1.5.3 The Reject Option

In some applications, it will be appropriate to avoid making decisions on the difficult cases in anticipation of a lower error rate on those examples for which a classification decision is made. This is known as the *reject option*.

<aside>
📌 Therese’s words:
Reject option is an introduction to an artificial class.

</aside>

For example, the medical example, it may be appropriate to use an automatic system to classify those X-ray images for which there is little doubt as to the correct class, while leaving a human expert to classify the more ambiguous cases. We can achieve this by introducing a threshold $θ$ and rejecting those inputs $x$ for which the largest of the posterior probabilities $p(Ck|x)$ is less than or equal to $θ$.

Posterior probabilities allow us to determine a rejection criterion that will minimise the misclassification rate, or more generally the expected loss, for a given fraction of rejected data points.

## 1.5.4 Inference and Decision

The decision stage is where we use the posterior probabilities to make optimal class assignments. An alternative possibility would be to solve both problem (inference and decision) together, and simply just learn a function that maps inputs $x$ directly into decisions. This is called a discriminant function.

## 1.5.5 Loss Functions for Regression

### 3 approaches to solving regression problems

- First solve the inference problem of determining the joint density $p(x, t)$. Then normalise to find the conditional density $p(t|x)$, and finally marginalise to find the conditional mean given by (1.89).

$$
y(x) = \frac{\int tp(x,t)dt}{p(x)} = \int tp(t|x)dt = E_t[t|x] \ \ (1.89)
$$

- First solve the inference problem of determining the conditional density p(t|x), and then subsequently marginalise to find the conditional mean given by (1.89).
- Find a regression function y(x) directly from the training data.


## Notes from lecture

### Joint distribution of data

$p(x,y)$

The probability of drawing exactly this pair of features and outcome.
The marginal distribution.

$p(x|y)$ Take all the healthy people how do their features vary/distribute?
$p(y|x)$ Posterior probability of class y. $P(Y=k|x)$

$p(\theta)$ prior probability of the parameter

$p(\theta | y)$ posterior probability of the parameter given some data


<div style="text-align: center;">
    <img src="/Screenshot_2022-09-20_at_10.35.52.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


<div style="text-align: center;">
    <img src="/Screenshot_2022-09-20_at_10.37.54.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

How does y vary on a fixed x? The exact same as:
$P(A|B) = \frac{P(A \cap B)}{P(B)}$

Posterior expected loss:

$1-\argmax_{k \in 1...K} P(Y=k|X_0)$

<div style="text-align: center;">
    <img src="/Screenshot_2022-09-20_at_10.45.59.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


<div style="text-align: center;">
    <img src="/Screenshot_2022-09-20_at_11.04.09.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


<div style="text-align: center;">
    <img src="/Screenshot_2022-09-20_at_11.41.46.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

$p(Y=k)$ is the probability of classes. How does the classes look. 
$p(x|Y=k)$ given a particular class, how does the features vary. 

************Fig 1.24************
$\hat x$ (to the right of) is the region where i want to classify to class 2, and the left is class 1. 

The probability of classifying wrong: The green and red part is the probability of classifying wrong, when it should have been class 1.

The blue part is the probability (the integral under that) that you classify to class 1, when it should have been 2.

If you shift the boundary to $x_0$, there still will be some error, but the errors will shrink. The blue and green error is the irreducible error, because you cant get rid of it. 
$x_0$ is the optimal for **misclassification error.**

### Posterior and Prior Probability

Loss and expected loss ($E(L(y,d(x)|x)$) are often mixed up.

The posterior mean minimises the expected squared loss error:
$L(y,d(x)) = (y-d(x))^2$

**Practical example.** Imagine that you are given a sample feature record $x$. Based on this vector, you are supposed to predict whether the given person has Covid or not, i.e., $C_k \in \{1, 0\}$. Before you would look at the given sample vector, the **prior probability** for each class would be $p(C_k)$. After you would look at the sample vector, your **posterior probability for each class** would be $p(C_k|x)$. In words, the probability that person is positive/negative given $x$. Using **Bayes theorem**, this quantity can be expressed as:

$$
p(C_k|x) = \frac{p(x|C_k)p(C_k)}{p(x)}
$$

Notice that if we knew joint probability $p(C_k, x)$ we could infer all terms on the right side from it. Most importantly, intuitively, we would **choose a class with highest posterior probability**. The goal of this section is to show why this intuition is in fact correct within the **classification** setting. However, almost identically, it is done for regression which is discussed in the next section.