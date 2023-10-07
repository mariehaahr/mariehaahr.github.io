---
title: 'Lecture 13'
description: 'Ensemble Methods continued'
pubDate: 'Oct 08 2023'
heroImage: '/lec13_ml.png'
---


**Readings**: ISL_wR Chapter 8.2 pages 340-345, Hands On: Chapter 7

#### Table of contents
- [Ensemble Methods](#ensemble-methods)
- [8.2 Bagging, Random Forests and Boosting](#82-bagging-random-forests-and-boosting)
  - [Bagging](#bagging)
    - [How can bagging be extended to a classification problem where Y is qualitative?](#how-can-bagging-be-extended-to-a-classification-problem-where-y-is-qualitative)
    - [Error Estimation using OOB (out-of-bag)](#error-estimation-using-oob-out-of-bag)
    - [Advantages of bagging](#advantages-of-bagging)
    - [Variable Importance Measures](#variable-importance-measures)
    - [Pasting (bagging without replacement)](#pasting-bagging-without-replacement)
  - [Random Forests](#random-forests)
    - [Advantages of Random Forests](#advantages-of-random-forests)
  - [Boosting](#boosting)
    - [Extra Trees](#extra-trees)

# Ensemble Methods

# 8.2 Bagging, Random Forests and Boosting

## Bagging

Decision trees has a tendency to suffer from high variance. Bagging (bootstrap Aggregation) is a general-purpose procedure for reducing the variance of a statistical learning method. Bagging is very good when you have very much variance in the data. 

Given a set of $n$ independent observations $Z_1,...,Z_n$ each with variance $\sigma^2$, the variance of the mean $\bar Z$ of the observations is then given by $\frac{\sigma^2}{n}$. This means that averaging a set of observations reduces the variance! This then lead to, a natural way to reduce the variance and increase the test set accuracy (generally of a statistical learning method) is to:

- Take many training sets from the same population
- Build a separate prediction model using each training set.
- Then average over the resulting predictions.

So in other words, we could calculate $\hat f^1(x),...,\hat f^B(x)$ using $B$ separate training sets, and average them in order to obtain a single low-variance statistical learning model.

BUT since we normally don’t have access to several data sets, we bootstrap the same training set and aggregate those, which is called bagging.

We generate $B$ different bootstrapped training data sets, and train our method on the $b$-th bootstrapped training set in order to get $\hat f^{*b}(x)$, and finally average all the predictions, to obtain:

$$
\hat f_{bag}(x) = \frac{1}{B} \sum^B_{b=1} \hat f^{*b} (x)
$$

<aside>
📌 Unfortunately, averaging many highly correlated quantities does not lead to as large of a reduction in variance. Bagging will not lead to a substantial reduction in variance over a single tree in this setting.  

Random Forests overcome this problem, though.

</aside>

**Bagging shortly explained**

To sum up, bagging involves creating multiple copies of the original training data set using the bootstrap, fitting a separate decision tree to each copy, and then combining all of the trees in order to create a single predictive model.

Make bootstrap subsets of the training data and train model on these different subsets in parallel. 

### How can bagging be extended to a classification problem where Y is qualitative?

For a given test observation, we can record the class predicted by each of the $B$ trees, and take a majority vote (the overall prediction is the most commonly occurring class among the $B$ prediction).

The probability of an observation being in a bootstrapped dataset:

$$
(1-\frac{1}{N})^N = \frac{1}{e} = 0.37
$$

The probability of a data point not being in a bootstrapped dataset is then 1 minus the thing above.

### Error Estimation using OOB (out-of-bag)

<aside>
📌 The OOB approach for estimating the test error is particularly convenient when performing bagging on large data sets.

</aside>

We don’t have to use cross-validation or validation set when estimating the error. A much less computationally expensive way to estimate the test error is using OOB.

The key to bagging is that trees are repeatedly fit to bootstrapped subsets of the observations. Here, only around $2/3$ of the observation is used in each bagged tree (because the bootstrapping method is with replacement, so some of the observations are never used). The remaining $1/3$ is then referred to as the out-of-bag observations.

We can predict the response for the $i$-th observation using each of the trees in which that observation was out of bag.

Sampling with replacement: 
- Not all data points will be in a given bag (bootstrapped dataset)
- Different data points are included/excluded in different bags. 

<aside>
📌 OOB error is an unbiased estimation of the test error, because data points in OOB are not used in training. OOB can replace the use of cross-validation or separate validation set.

</aside>

**To calculate the OOB error:**

1. Find all trees that are not trained by the OOB observation.
2. Take the majority vote of these trees’ result for the OOB observation, compared to the true value of the OOB observation.
3. Compile the OOB error for all observations in the OOB dataset (around a third of the dataset).

To validate an ensemble using OOB:
• Each predictor predicts on its OOB data points
• The results from all predictors are aggregated

### Advantages of bagging

Easy to implement

Reduces Variance → beneficial for high variance predictors

We can use OOB error for validation

The prediction is an average of many predictors’ outputs.  We can also obtain variance of outputs (score variance) to be interpreted as the uncertainty of the prediction.

### Variable Importance Measures

Even though bagging has a higher prediction accuracy than a single tree, we lose some of the interpretability when bagging a large number of trees.

Though, there is a way of obtaining a overall summary of the importance of each predictor using the Residual Sum og Squares (for regression) or the Gini index (for classification).

**Bagging Regression**

In the case of regression, we can record the total amount of that the RSS is decreased due to splits over a given predictor, averaged over all $B$ trees. The larger the value, the more important the predictor.

$$
RSS = \sum_{j=1}^J \sum_{i \in R_j}(y_i - \hat y_{R_j})^2
$$

Where $\hat y_{R_j}$ is the mean response for the training observations within the $j$th region. 

**Bagging Classification**

In the case of classification, we can add up the total amount that the Gini index is decreased by splits over a given predictor, averaged over all $B$ trees.

$$
\text{Gini Index} = \sum_{k=1}^K \hat p_{mk}(1-\hat p_{mk})
$$

Where $\hat p_{mk}$ represents the proportion of training observation in the $m$-th region that are from the $k$-th class.

### Pasting (bagging without replacement)

Both methods scale well:

If you want to have a smaller bootstrapped dataset. You can try both bagging and pasting and see which one is better
- training in parallel
- predicting in parallel

## Random Forests

<aside>
📌 The main difference between bagging and random forests is the choice of predictor subset size $m$.

</aside>

Random Forests provide an improvement over bagging trees, because is decorrelates the trees. Each time a split in a tree is considered, a random sample of $m$ predictors is chosen as split candidates from the full set $p$  predictors. At each split, a fresh sample of the $m$ predictors is taken. It is typically $m \approx \sqrt{p}$.

**Why do we do this?**

Suppose that there is one very strong predictor in the data set, along with a number of other moderately strong predictors. Then in the collection of bagged trees, most (or all) of the trees will use this strong predictor in the top split, hence the predictors from the bagged trees will be highly correlated. 

Averaging many uncorrelated quantities will lead to a substantial reduction in variance over a single tree (compared to bagging).

Random forest overcome this problem, by forcing each split to consider only a subset of the predictors. When the trees are decorrelated, is making the average of the resulting trees less variable, and hence, more reliable.

Using a small value of $m$  (subset of predictors) when building a random forest will typically be helpful when we have a large number of correlated predictors.

Random Forests are essentially a bagged decision tree, but with a slightly modified splitting criterion.

- Make bootstrapped data sets from training set.
- For each data set, train a full DT with a modified split strategy.
      - before each split:
• randomly choose some of the features
• only consider those for the split
• The output of random forest is the aggregated output of all the trees.

### Advantages of Random Forests

The Random Forests (RF) are nice, because:

- Only few hyper-parameters:
- $m$, number of bootstrapped datasets, and 
$- k$, the number of randomly selected features.
- For classifiers, usually a good choice for k is $k = \sqrt{\text{number of features}}$. You can set m as large as you can afford.
- It is based on decision trees and they do not require a lot of preprocessing. e.g, the features can be of different scales and types.
- Like single decision trees, a feature’s importance can be estimated by its (aggregated) closeness to the root – average depth

## Boosting

<div style="text-align: center;">
    <img src="/Screenshot_2022-10-11_at_11.07.13.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

<aside>
📌 Unlike fitting a single large decision tree to the data (which amounts to *fitting the data hard* and potentially overfitting), the boosting approach instead *learns slowly.*

In general, statistical learning approaches that *learn slowly* tend to perform well.

</aside>

Boosting is a general approach that can be applied to many statistical learning methods for regression or classification. 

Bagging and its associated methods are very good at reducing variance, but does not reduce bias. Boosting though, reduces bias. 

**The main idea (Own notes of confusion):**

Boosting works in a similar way, except that the trees are grown *sequentially:* ****each tree is grown using information from previously grown trees. **Boosting does not involve bootstrap sampling,** instead each tree is fit on a modified version of the original data set. Given the current model, we fit a decision tree to the residuals from the model, That is, we fit a tree using the current residuals, rather than the outcome $Y$, as the response.

When you need to take into account for the weights after training a model and you need to train a new one with the updated weights, it is up to you how you want to do it. Ludek mentions that you can bootstrap the data (or sample the data with replacement) but as a hyper-parameter for the sampling, you have the weights to the data-points. At first, all the data points have the same weights, you sample from the data with replacement like every other time. Then you train a tree, reweight the misclassified points, and now you have to train a new model with the new weights. 

**Important**: for all the weights to sum to one, you need to normalise all the weights after readjusting the misclassified ones. Here you divide all the current weights with the sum of all weights (which now is not 1 anymore, but something above). This should normalise all the weights, so you now can continue.

Now you have to sample the data again, now with the new weights. Here you give python `np.random.choice` the hyper-parameter `p` the weights, because when the probability (or the weight) of the misclassified point is higher, the chance of us sampling that point for the next model is higher. This means that the next model will take into account all the weight it couldn’t handle, and we would learn from the previous models mistakes. 

At the end you have a lot of models through the sequence that you sum up. All the bad ones and all the good ones. 

**Boosting classification trees**

This is quite similar, but a little more complex.

There are 3 tuning parameters, that define the end condition for building a new tree. They are usually tuned to increase accuracy and prevent overfitting:

1. Max depth (usually we want < 10)
2. Max features (the tree doesn’t have to use all features)
3. Minimum samples per leaf (usually we want < 1% of the data)


<div style="text-align: center;">
    <img src="/Screenshot_2022-10-08_at_12.31.47.png" alt="dd" width="600" height="350" style="text-align: center;" >
</div>

### Extra Trees

It is possible to make trees even more random by also using random thresholds for each feature rather than searching for the best possible thresholds (split point).

When you do this, it is called an **Extremely Randomised Trees Ensemble.** This technique trades more bias for a lower variance.

The thing that makes this method not 100% random is that you choose the best split of these random splits, and that way you random forest becomes better and better. 

There is no bootstrapping, you just use the original dataset.

If you choose $m=3$ you choose 3 random features, split the datapoints at random for those 3 feautes, calculate the impurity for each of the splits and then choose the best one (aka the lowest one). 

<aside>
📌 Extra-Trees is much faster to train than regular Random Forest, since finding the best possible threshold for each feature is one of the most time-consuming tasks of growing a tree.

</aside>

Unlike bagging and random forest, the Extra Trees fit each decision tree on the whole
training dataset (i.e., no bootstrapped datasets).

- Like random forest, the Extra Trees algorithm randomly samples the features at each
decision node of each decision tree
- Unlike random forest, the Extra Trees algorithm selects a split point at random from
each of the features, then picks the best of them.