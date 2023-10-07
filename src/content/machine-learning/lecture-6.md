---
title: 'Lecture 6'
description: 'Classification, Bayes Clf., K-Nearest Neighbours'
pubDate: 'Sep 23 2023'
heroImage: '/lec6_ml.png'
---

**Readings**: 2.2.3 in ISLwR, and 1.5 in pattern recognition 

#### Table of contents
- [2.2.3 The Classification Setting](#223-the-classification-setting)
    - [Bayes Classifier](#bayes-classifier)
    - [K-Nearest Neighbours](#k-nearest-neighbours)
- [Notes from class](#notes-from-class)
  - [Loss function](#loss-function)
    - [Loss function in classification setting](#loss-function-in-classification-setting)
    - [Loss functions in regression setting](#loss-functions-in-regression-setting)
    - [Expected loss for a prediction mechanism](#expected-loss-for-a-prediction-mechanism)
    - [Find a good prediction Mechanism](#find-a-good-prediction-mechanism)
  - [Bayes Classifier](#bayes-classifier-1)

# 2.2.3 The Classification Setting

We now seek to estimate $f$ on the basis of training observations $\{(x_1,y_1...)\}$ where $y_1,...,y_n$ are qualitative (in classes). **The most common approach** for quantifying the accuracy of our estimate $\hat f$ is the training *error rate,* the proportion of mistakes that are made if the estimate is applied to the **training observations.**

$$
\frac{1}{n} \sum_{i=1}^nI(y_i \ne \hat y_i)
$$

Here $\hat y^i$ is the predicted class label for the ith observation using f. And
$I(y_i \ne \hat y_i)$ is an *indicator variable* that equals 1 if $y_i \ne \hat y_i$, and zero if $y_i = \hat y_i$. If $I(y_i \ne \hat y_i)$ the *i*th observation was classified correctly by our classification method.

**WE are interested in the error rates that result from applying out classifier to test observations that were NOT used in training.** Here we can use the *test error rate* associated with a set of test observations given by:
$Ave(I(y_0 \ne \hat y_0) \ \ \ \ (2.9)$, 
where $\hat y_0$ is the predicted class label that results from applying the classifier to the test observation with predictor $x_0$. A good classifier is one for which the test error (2.9) is smallest.

### Bayes Classifier

<aside>
📌 *In theory, Bayes classifier would be ideal, but for real life data we do not know the conditional distribution of $Y$ given $X$, so computing Bayes classifier is impossible.
This is where K-Nearest Neighbours comes in.*

</aside>

Bayes classifier is a simple classifier that assigns each observation to the most likely class, given its predictor values. 
We should assign a class *j* to a test observation (with predictor vector $x_0$ ) for which the probability is largest:

$Pr(Y=j|X=x_0)$

**Note that this is CONDITIONAL PROBABILITY.**
The Bayes classifier’s prediction is determined by the Bayes decision boundary.

The Bayes classifier produces the lowest possible test error rate, called the Bayes Error rate:

$$
1-E(\max_j Pr(Y=j|X)
$$

Where the expectation averages the probability over all possible values of $X.$

### K-Nearest Neighbours

Since we do now know the conditional distribution of Y given X, many approaches attempt to estimate the conditional distribution, and then classify a given observation to the class with highest *estimated* probability.

One such, is KNN. Despite the fact that it is a very simple approach, KNN can often produce classifiers that are surprisingly close to the optimal Bayes classifier. 

The choice of K has a drastic effect on the KNN classifier obtained. 
**K can be chosen by cross-validation**

<div style="text-align: center;">
    <img src="/Untitled_lec6.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

**Flexibility**

When $K=1$, the decision decision boundary is overly flexible and finds patterns in the data that don’t correspond to the Bayes. This corresponds to a classifier that has low bias but very high variance. As K grows, the method becomes less flexible and produces a decision boundary that is close to linear. This corresponds to a low-variance but high-bias classifier. 

**Often does not work well in high dimension feature space**

# Notes from class

How do we create a good decision function?

## Loss function

Loss function, it assigns a real-values loss $L(Y,\hat Y)$ to a combination of the ground truth $Y$, and our prediction $\hat Y = d(X)$.

The higher the loss, the worse the prediction.
We wish to find a decision function with minimal expected loss.

### Loss function in classification setting

Misclassification error (0-1 loss), 1 if the predicted class is not equal to the real class and 0 if they are equal.

<div style="text-align: center;">
    <img src="/Screenshot_2022-09-21_at_10.56.05.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>


### Loss functions in regression setting

**Squared error loss:**

$$
L(y,d(x)) = (y-d(x))^2
$$

**Absolute error loss:**

$$
L(y,d(x)) = |y-d(x)|
$$

### Expected loss for a prediction mechanism

Taking a new point (X,Y), what would be the expected loss associated with using the decision function d?
We can estimate it from a specific dataset by the *empirical risk*:

$$
\frac{1}{n} \sum^n_{i=1} L(Y_i, d(X_i))
$$

**Training error:** the empirical risk on the training data (OBS: generally a bad estimator of the expected loss)

**THE TRUE EXPECTED LOSS:Test error:** the empirical risk on the test data

### Find a good prediction Mechanism

We wish to construct a decision function $d(x)$ so that it minimises the expected loss $E[L(Y,d(x))]$

**For classification:**

$E[L(Y,d(x))] = \int_X \sum^k_{y=1} L(y,d(x))p(x,y)dx$ 

(You can think of $x$  as a single continuous feature. If  $x$ is a feature vector this is a multiple integral, and if some features are discrete the integral is replaced with a sum.)


<div style="text-align: center;">
    <img src="/Screenshot_2022-09-21_at_11.06.57.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

## Bayes Classifier

It minimises the expected loss under the specific choice of misclassification error

With Bayes classifier, we classify to the class $k$ with the highest posterior probability:

$$
d(x) = \argmax_y P(Y=y | X=x)
$$

KNN is a simple approximation to the Bayes Classifier 

KNN is not good for high dimension, because of the curse of high dimensionality. The more dimensions, the more diastance there will be between the neighbours youre choosing