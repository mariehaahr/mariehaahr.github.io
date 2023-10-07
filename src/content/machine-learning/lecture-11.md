---
title: 'Lecture 11'
description: 'Decision Trees'
pubDate: 'Oct 06 2023'
heroImage: '/lec11_ml.png'
---

**Readings**: ISLwR 327-331, Hands on 237-242 + video posted on LearnIT

#### Table of contents
- [Decision Trees Video](#decision-trees-video)
  - [Decision Trees in General](#decision-trees-in-general)
- [ISLwR Tree-Based Methods](#islwr-tree-based-methods)
  - [Regression Trees](#regression-trees)
    - [Prediction via Stratification of the Feature Space](#prediction-via-stratification-of-the-feature-space)
    - [Tree Pruning  (not hw)](#tree-pruning--not-hw)
- [Hands-On: Decision Trees](#hands-on-decision-trees)
  - [Training and Visualising a Decision Tree](#training-and-visualising-a-decision-tree)
    - [In code](#in-code)
  - [Making Predictions](#making-predictions)
    - [Gini Attribute](#gini-attribute)
  - [Estimating Class Probabilities](#estimating-class-probabilities)
  - [The CART Training Algorithm](#the-cart-training-algorithm)
  - [Gini Impurity or Entropy?](#gini-impurity-or-entropy)
  - [Regularisation Hyperparameters](#regularisation-hyperparameters)
  - [Regression](#regression)
  - [Instability - Limitations](#instability---limitations)

# Decision Trees Video

StatQuest video on YouTube.

Generally, a decision tree asks a question and based on that answer, we classify that. 

The classification can be either categories or numeric. 

The top of the tree is called a root node (root). The nodes down the tree is called the internal nodes (nodes). The decisions are called leaves. 

To decide which feature to select i the root node, we look at the feature that has the best fit to the classification.. (??). So to determine which separation is best, we measure and compare the impurity of the features. 

There are several way to measure this, but one popular is “Gini impurity”. The feature with the lowest Gini impurity gets to be the root. 

Then you calculate the gini impurity on the other features (other than the one at the root) and decide which feature best fits to be the next node in the tree. Do it the same way with the impurity measure. 

To wrap it up:

- First calculate all the Gini Impurity scores.
- Then, if the node itself has the lowest score, we stop there and make that node a leaf node.
- If separating the data results in an improvement in the impurity score, then we pick the separation with a feature with the lowest impurity score.

## Decision Trees in General

DTs are non-parametric, supervised machine learning models. The thing that DTs do is predict a value of a target (can be both classification and regression) by learning some decision rules from the features. DTs approximate posterior class probabilities by maximising class separation at each level.

So prediction happens, the tree is traversed such that you start at the root of the tree, and the next node in the tree is then chosen based on the given threshold.

Tree-based approaches in general are based on the idea of *stratifying* or *segmenting* the feature space into a number of simple regions (hyper-rectangles) The prediction is then based on identifying which region a new data point belongs to and predicts the mean of those values.

Visualising a decision tree makes everything easier, and this also shows that DTs are white box model

# ISLwR Tree-Based Methods

Each of these approaches involves producing multiple trees, and then combine them into a single prediction.

## Regression Trees

DT’s can be applied to both regression and classification problems.

The regions $R_1, R_2, ...,R_J$ are known as **terminal nodes** (or leaves) of the tree. 

The points along the tree where the predictor space is split, are referred to as **internal nodes.**

The segments of the trees that connect the nodes are called **branches.**

### Prediction via Stratification of the Feature Space

There are 2 steps in building a regression tree.

- We divide the predictor space - that is, the set of possible values for $X_1,X_2,...,X_p$ - into $J$ distinct and non-overlapping regions, $R_1, R_2, ...,R_J$.
- For every observation that falls into the region $R_j$, we make the same prediction, which is simply the mean of the response values for the training observations in $R_j$.

In theory, the regions could have any shape, however we choose to divide the predictor space in to rectangles (boxes). And now the goal is to find boxes $R_1,...,R_J$ that minimise the RSS given by:

$$
\sum_{j=1}^J \sum_{i \in R_j}(y_i - \hat y_{R_j})^2 \ \ \ \ \ \ \ (8.1)
$$

Here, $\hat y_{R_j}$ is the mean response for the training observations within the $j$th box.

Unfortunately this is computationally infeasible to consider every possible partition of the feature space into $J$ boxes, so we take a greedy approach to this:

The recursive binary splitting:

- We first select the predictor $X_j$ and the cut-point $s$ such that splitting the predictor space into the regions $\{X|X_j < s\}$ and $\{X|X_j \ge s \}$ leads to the greatest possible reduction in RSS.
- Next we repeat the process, looking for the best predictor and best cut-point in order to split the data further so as to minimise the RSS within each of the resulting regions. However, this time, instead of splitting the entire predictor space, we split one of the two previously identified regions.
- The process continues until a stopping criterion is reached; for instance, we may continue until no region contains more than five observations.

**This process may product good predictions on the training set, but is very likely to overfit the data!!**

### Tree Pruning  (not hw)

A smaller tree with fewer splits (fewer regions) might lead to lower variance and better interpretation at the cost of a little bias.

Therefore, a better strategy is to grow a very large tree, and then ‘prune’ it back in order to obtain a **subtree.** Our goal is then to select a subtree that leads to the lowest *test error rate.* Given a subtree, we can estimate its test error using **cross-validation**.

Since we don’t want to estimate the error for every possible subtree, we instead do **Cost Complexity Pruning** - also known as *weakest link pruning.* Rather than considering every possible subtree, we consider a sequence of trees indexed by a nonnegative tuning parameter $\alpha$.

# Hands-On: Decision Trees

Decision trees are intuitive, and easy to interpret. These models are called white box models, since you know what the decision are based on compared to black box models like neural networks and random forests.

Decision Trees make very few assumption about the training data. (as opposed to linear models that assumes the data is linear). 

## Training and Visualising a Decision Tree

### In code

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```

## Making Predictions

One of the many qualities of decision trees is that they require very little data representation. They don’t require feature scaling or centring at all. 

### Gini Attribute

It measures a nodes impurity. A node is “pure” (gini=0) if all training instances it applies to belong to the same class.

$$
\text{Gini Impurity} = G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2
$$

Here $p_{i,k}$ is the ratio of class $k$ instances among the training instances in the $i$th node.

## Estimating Class Probabilities

```python
>>> tree_clf.predict_proba([[5, 1.5]])
# array([[0. , 0.90740741, 0.09259259]])

>>> tree_clf.predict([[5, 1.5]])
# array([1])
```

## The CART Training Algorithm

The Classification and Regression Tree (CART) algorithm:

- First splitting the training set into two subsets using a single feature $k$ and a threshold $t_k$.
- When the training set is split, it splits the subsets using the same logic, then the sub-subsets, and so on, recursively. It stops recursing once it reaches the maximum depth (often specified) or if ot cannot find a split that will reduce the impurity.

The CART algorithm is a greedy algorithm, meaning that it often produces a solutions that’s reasonably good, but not guaranteed to be optimal. 

## Gini Impurity or Entropy?

In Machine learning, entropy is frequently used as an impurity measure: a set’s entropy is 0 when it contains instances of only 1 class. 

$$
\text{Entropy} = H_i = \sum_{k = 1, \ p_{i,k} \ne 0}^{n} p_{i,k} \log_2(p_{i,k})
$$

Most of the time it does not make a big difference if you use Gini or entropy, they lead to similar trees. Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees. 

## Regularisation Hyperparameters

If the Decision tree is left unconstrained, the structure will adapt itself to the training data, fitting it very closely. 

This is called a **nonparametric model**, because the number of parameters is not determined beforehand, like the parametric models are, i.e logistic regression where you have determined a number of parameters.

This means, that the model structure is free to stick closely to the data. To avoid overfitting the training data, we need to restrict the freedom of the DT during training. This is called **regularisation**. 

The regularisation hyper-parameters depend on the algorithm used, but normally you can restrict the maximum depth of the DT, and thus reduce the risk of overfitting.

The `DecisionTreeClassifier` class has some other parameters that restrict the shape of the DT:

- `min_samples_split` - the minimum number of samples a node must have before it can be split.
- `min_samples_leaf` - the minimum number of samples a leaf node must have.
- `min_weight_fraction_leaf` - same as the one above, but expressed as a fraction of the total number of weighted instances.
- `max_leaf_nodes` - the maximum number of leaf nodes.
- `max_features` - the maximum number of features that are evaluated for splitting at each node.

## Regression

DT performing regression tasks.

The main difference is that instead of predicting a class in each node, it predicts a value.

Just like classification tasks, DT’s are prone to overfitting when dealing with regression tasks.

## Instability - Limitations

DT’s are simple and powerful, but have a few limitations:

- First, DT’s love orthogonal decision boundaries, which makes them sensitive to training a set rotation.
- A main issue with DT’s is that they are very sensitive to small variations in the training data.
- OBS - you may get very different models on the same training data when using `Scikit-Learn` training algorithm, so you need the `random_state` hyper-parameter.

Solution: Random Forests can limit this instability by averaging predictions over many trees.